//! DuckDB table functions that *tee* one read-once Arrow stream to every
//! scan of it in a query, the DuckDB counterpart of
//! `batchcorder-datafusion`'s `TeeTableProvider`.
//!
//! [`register`] adds three functions to a database:
//!
//! - `CALL tee_register(name, stream_ptr)` takes ownership of the
//!   `ArrowArrayStream` at address `stream_ptr` and wraps it in a
//!   `StreamCache` stored under `name`.
//! - `SELECT * FROM tee_scan(name)` replays the stream from its first batch;
//!   every scan reads the same cache, so the upstream is read once.
//! - `CALL tee_drop(name)` forgets the name.  Scans still running keep the
//!   cache alive until they finish.
//!
//! The registry is per database: each [`register`] call creates one, shared
//! by the three functions it adds and freed with the database.
//!
//! # Stream ownership
//!
//! `stream_ptr` is the address of a caller-allocated `ArrowArrayStream`.
//! When `tee_register` *runs* (not when it is bound, so `EXPLAIN` and
//! re-binding never consume it), it moves the struct out and leaves the
//! source with `release = NULL`.  From then on the stream is the cache's,
//! whether registration succeeds or fails; the caller must not release it
//! again and still frees the struct's own memory.  A NULL address, or a
//! stream whose `release` is already NULL, is an error.
//!
//! # DuckDB facts this relies on
//!
//! `tests/probes.rs` pins how DuckDB drives a C-API table function: every
//! table-function reference gets its own `bind` (a `NOT MATERIALIZED` CTE
//! adds one unused bind), and the bind data of a scan the optimizer prunes,
//! or never uses, is freed before the remaining scans execute.

mod bridge;

use std::collections::HashMap;
use std::error::Error;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use arrow_array::RecordBatchReader as _;
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use batchcorder_core::{CacheConfig, StreamCache, StreamCacheReader};
use duckdb::Connection;
use duckdb::arrow::array::{ArrayRef, MutableArrayData, RecordBatch, make_array};
use duckdb::arrow::compute::cast;
use duckdb::arrow::datatypes::{DataType, Field, FieldRef, Schema, SchemaRef};
use duckdb::arrow::error::ArrowError;
use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{
    BindInfo, InitInfo, TableFunctionInfo, VTab, Value, record_batch_to_duckdb_data_chunk,
    to_duckdb_logical_type_for_field,
};

use crate::bridge::BoxError;

/// Add `tee_register`, `tee_scan` and `tee_drop` to `con`'s database, with a
/// fresh registry of tees.
pub fn register(con: &Connection) -> duckdb::Result<()> {
    let registry = Registry::default();
    con.register_table_function_with_extra_info::<TeeRegister, _>("tee_register", &registry)?;
    con.register_table_function_with_extra_info::<TeeScan, _>("tee_scan", &registry)?;
    con.register_table_function_with_extra_info::<TeeDrop, _>("tee_drop", &registry)?;
    Ok(())
}

/// A registered stream: the cache, and its schema in duckdb-rs's arrow.
#[derive(Clone)]
struct Tee {
    cache: StreamCache,
    schema: SchemaRef,
}

/// The tees of one database, by name.
#[derive(Clone, Default)]
struct Registry(Arc<Mutex<HashMap<String, Tee>>>);

impl Registry {
    /// The map, even after a panic elsewhere poisoned its lock: entries are
    /// only inserted and removed whole, so it is always valid.
    fn tees(&self) -> MutexGuard<'_, HashMap<String, Tee>> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The registry a function was registered with.
    fn of(bind: &BindInfo) -> Self {
        // SAFETY: `register` registers every function with a `Registry` as
        // its extra info, which DuckDB keeps alive as long as the function.
        unsafe { &*bind.get_extra_info::<Registry>() }.clone()
    }

    /// Take ownership of the stream at `ptr` and store it under `name`.
    /// Returns the number of columns.
    fn add(&self, name: &str, ptr: u64) -> Result<usize, BoxError> {
        // A u64 wider than a pointer cannot be an address on this platform;
        // truncating it would point somewhere else entirely.
        let raw = usize::try_from(ptr)
            .map_err(|_| format!("tee_register: stream_ptr {ptr:#x} is not an address"))?
            as *mut FFI_ArrowArrayStream;
        if raw.is_null() {
            return Err("tee_register: stream_ptr is NULL".into());
        }
        // SAFETY: the caller promises `ptr` addresses a live
        // ArrowArrayStream (the documented contract).  A NULL `release`
        // marks a stream that was already moved or released.
        if unsafe { (*raw).release.is_none() } {
            return Err("tee_register: stream already consumed or released".into());
        }
        // SAFETY: as above; `from_raw` moves the struct out and leaves the
        // source with `release = NULL`, so it is released exactly once.
        let stream = unsafe { FFI_ArrowArrayStream::from_raw(raw) };
        let reader = ArrowArrayStreamReader::try_new(stream)?;
        let arrow_schema = reader.schema();
        let schema = bridge::schema(&arrow_schema)?;
        let cache = StreamCache::try_new(Box::new(reader), arrow_schema, CacheConfig::default())?;
        let mut tees = self.tees();
        if tees.contains_key(name) {
            // The stream is ours by now; dropping `cache` releases it.
            return Err(format!(
                "tee_register: a tee named '{name}' already exists; tee_drop it first"
            )
            .into());
        }
        let columns = schema.fields().len();
        tees.insert(name.to_owned(), Tee { cache, schema });
        Ok(columns)
    }

    fn get(&self, name: &str) -> Result<Tee, BoxError> {
        self.tees().get(name).cloned().ok_or_else(|| {
            format!("tee_scan: no tee named '{name}'; register it with tee_register").into()
        })
    }

    fn remove(&self, name: &str) -> bool {
        self.tees().remove(name).is_some()
    }
}

fn varchar() -> LogicalTypeHandle {
    LogicalTypeHandle::from(LogicalTypeId::Varchar)
}

/// Parameter `index` of `function`, which must not be NULL.  Reading a NULL
/// value as VARCHAR or UBIGINT throws a C++ exception that aborts the
/// process, so NULL is rejected before any conversion.
fn parameter(bind: &BindInfo, function: &str, index: u64, what: &str) -> Result<Value, BoxError> {
    let value = bind.get_parameter(index);
    if value.is_null() {
        return Err(format!("{function}: {what} is NULL").into());
    }
    Ok(value)
}

/// The tee name, parameter 0 of every function.
fn name_parameter(bind: &BindInfo, function: &str) -> Result<String, BoxError> {
    Ok(parameter(bind, function, 0, "name")?.to_string())
}

/// Init data of a function that emits one row and then stops.
#[derive(Default)]
struct Once {
    done: AtomicBool,
}

impl Once {
    /// True the first time only.
    fn first(&self) -> bool {
        !self.done.swap(true, Ordering::Relaxed)
    }
}

// ── tee_register ────────────────────────────────────────────────────────────

struct TeeRegister;

struct RegisterBind {
    registry: Registry,
    name: String,
    ptr: u64,
}

impl VTab for TeeRegister {
    type InitData = Once;
    type BindData = RegisterBind;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("name", varchar());
        bind.add_result_column(
            "column_count",
            LogicalTypeHandle::from(LogicalTypeId::UBigint),
        );
        Ok(RegisterBind {
            registry: Registry::of(bind),
            name: name_parameter(bind, "tee_register")?,
            ptr: parameter(bind, "tee_register", 1, "stream_ptr")?.to_uint64(),
        })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(Once::default())
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        if !func.get_init_data().first() {
            output.set_len(0);
            return Ok(());
        }
        let bind = func.get_bind_data();
        // Ownership is taken here, at execution, never in `bind`.
        let columns = bind.registry.add(&bind.name, bind.ptr)?;
        output.flat_vector(0).insert(0, bind.name.as_str());
        // SAFETY: column 1 is UBIGINT and the chunk holds at least one row.
        unsafe { output.flat_vector(1).as_mut_slice::<u64>()[0] = columns as u64 };
        output.set_len(1);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![
            varchar(),
            LogicalTypeHandle::from(LogicalTypeId::UBigint),
        ])
    }
}

// ── tee_drop ────────────────────────────────────────────────────────────────

struct TeeDrop;

struct DropBind {
    registry: Registry,
    name: String,
}

impl VTab for TeeDrop {
    type InitData = Once;
    type BindData = DropBind;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        bind.add_result_column("name", varchar());
        bind.add_result_column("dropped", LogicalTypeHandle::from(LogicalTypeId::Boolean));
        Ok(DropBind {
            registry: Registry::of(bind),
            name: name_parameter(bind, "tee_drop")?,
        })
    }

    fn init(_: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        Ok(Once::default())
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        if !func.get_init_data().first() {
            output.set_len(0);
            return Ok(());
        }
        let bind = func.get_bind_data();
        let dropped = bind.registry.remove(&bind.name);
        output.flat_vector(0).insert(0, bind.name.as_str());
        // SAFETY: column 1 is BOOLEAN and the chunk holds at least one row.
        unsafe { output.flat_vector(1).as_mut_slice::<bool>()[0] = dropped };
        output.set_len(1);
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![varchar()])
    }
}

// ── tee_scan ────────────────────────────────────────────────────────────────

struct TeeScan;

/// Where one scan is in the stream.
struct Cursor {
    reader: StreamCacheReader,
    /// Rows per output chunk: DuckDB's vector size.
    vector_size: usize,
    /// The batch being emitted, and how many of its rows are out already.
    current: Option<(RecordBatch, usize)>,
}

impl Cursor {
    /// The next slice of at most `vector_size` rows, or `None` at the end of
    /// the stream.
    fn next_slice(&mut self) -> Result<Option<RecordBatch>, BoxError> {
        loop {
            if let Some((batch, offset)) = &mut self.current
                && *offset < batch.num_rows()
            {
                let len = self.vector_size.min(batch.num_rows() - *offset);
                let slice = if len == batch.num_rows() {
                    batch.clone()
                } else {
                    compact_nested(&batch.slice(*offset, len))?
                };
                *offset += len;
                return Ok(Some(slice));
            }
            self.current = match self.reader.next() {
                None => return Ok(None),
                Some(batch) => Some((decode_dictionaries(bridge::batch(&batch?)?)?, 0)),
            };
        }
    }
}

/// `slice` with its nested columns copied into arrays of their own.
///
/// duckdb-rs writes a list (or map) column by copying its whole child array
/// and pointing each row at its offsets.  A slice keeps the parent's child
/// array, so without this every chunk of a large batch would copy the child
/// values of the entire batch: quadratic in the batch's length.
fn compact_nested(slice: &RecordBatch) -> Result<RecordBatch, BoxError> {
    if !slice
        .schema()
        .fields()
        .iter()
        .any(|f| f.data_type().is_nested())
    {
        return Ok(slice.clone());
    }
    let columns = slice
        .columns()
        .iter()
        .map(|column| {
            if !column.data_type().is_nested() {
                return Ok(column.clone());
            }
            let data = column.to_data();
            let mut copy = MutableArrayData::new(vec![&data], false, column.len());
            copy.try_extend(0, 0, column.len())?;
            Ok(make_array(copy.freeze()))
        })
        .collect::<Result<Vec<_>, ArrowError>>()?;
    Ok(RecordBatch::try_new(slice.schema(), columns)?)
}

/// `data_type` with every dictionary, at any depth, replaced by its value
/// type.
fn without_dictionaries(data_type: &DataType) -> DataType {
    let field = |f: &FieldRef| {
        Arc::new(
            f.as_ref()
                .clone()
                .with_data_type(without_dictionaries(f.data_type())),
        )
    };
    match data_type {
        DataType::Dictionary(_, value) => without_dictionaries(value),
        DataType::List(f) => DataType::List(field(f)),
        DataType::LargeList(f) => DataType::LargeList(field(f)),
        DataType::FixedSizeList(f, n) => DataType::FixedSizeList(field(f), *n),
        DataType::Map(f, sorted) => DataType::Map(field(f), *sorted),
        DataType::Struct(fields) => DataType::Struct(fields.iter().map(field).collect()),
        other => other.clone(),
    }
}

/// `batch` with every dictionary, top-level or nested, decoded to its value
/// type.  duckdb-rs binds a dictionary as its value type but cannot write
/// dictionary arrays into a chunk.
fn decode_dictionaries(batch: RecordBatch) -> Result<RecordBatch, BoxError> {
    let schema = batch.schema();
    let decoded: Vec<DataType> = schema
        .fields()
        .iter()
        .map(|f| without_dictionaries(f.data_type()))
        .collect();
    if schema
        .fields()
        .iter()
        .zip(&decoded)
        .all(|(f, d)| f.data_type() == d)
    {
        return Ok(batch);
    }
    let (fields, columns): (Vec<Field>, Vec<ArrayRef>) = schema
        .fields()
        .iter()
        .zip(batch.columns())
        .zip(decoded)
        .map(|((field, column), data_type)| {
            if field.data_type() == &data_type {
                return Ok((field.as_ref().clone(), column.clone()));
            }
            let column = cast(column, &data_type)?;
            Ok((field.as_ref().clone().with_data_type(data_type), column))
        })
        .collect::<Result<Vec<_>, ArrowError>>()?
        .into_iter()
        .unzip();
    Ok(RecordBatch::try_new(
        Arc::new(Schema::new(fields)),
        columns,
    )?)
}

struct ScanInit {
    cursor: Mutex<Cursor>,
}

impl VTab for TeeScan {
    type InitData = ScanInit;
    type BindData = Tee;

    fn bind(bind: &BindInfo) -> Result<Self::BindData, Box<dyn Error>> {
        let tee = Registry::of(bind).get(&name_parameter(bind, "tee_scan")?)?;
        for field in tee.schema.fields() {
            bind.add_result_column(field.name(), to_duckdb_logical_type_for_field(field)?);
        }
        Ok(tee)
    }

    fn init(init: &InitInfo) -> Result<Self::InitData, Box<dyn Error>> {
        // Each scan reads one ordered stream.
        init.set_max_threads(1);
        // SAFETY: DuckDB hands init the bind data `bind` returned, which
        // outlives this call.
        let tee = unsafe { &*init.get_bind_data::<Tee>() };
        // SAFETY: a plain getter of DuckDB's compile-time vector size.
        let vector_size = usize::try_from(unsafe { duckdb::ffi::duckdb_vector_size() })?;
        if vector_size == 0 {
            return Err("tee_scan: DuckDB reports a vector size of 0".into());
        }
        Ok(ScanInit {
            cursor: Mutex::new(Cursor {
                reader: tee.cache.reader(true)?,
                vector_size,
                current: None,
            }),
        })
    }

    fn func(
        func: &TableFunctionInfo<Self>,
        output: &mut DataChunkHandle,
    ) -> Result<(), Box<dyn Error>> {
        let mut cursor = func
            .get_init_data()
            .cursor
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        match cursor.next_slice()? {
            Some(slice) => record_batch_to_duckdb_data_chunk(&slice, output)?,
            None => output.set_len(0),
        }
        Ok(())
    }

    fn parameters() -> Option<Vec<LogicalTypeHandle>> {
        Some(vec![varchar()])
    }
}

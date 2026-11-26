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
use duckdb::arrow::array::{ArrayRef, RecordBatch};
use duckdb::arrow::compute::cast;
use duckdb::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use duckdb::arrow::error::ArrowError;
use duckdb::core::{DataChunkHandle, Inserter, LogicalTypeHandle, LogicalTypeId};
use duckdb::vtab::{
    BindInfo, InitInfo, TableFunctionInfo, VTab, record_batch_to_duckdb_data_chunk,
    to_duckdb_logical_type_for_field,
};

use crate::bridge::BoxError;

/// Rows DuckDB accepts per output chunk (its standard vector size).  Cached
/// batches larger than this are emitted in slices.
const VECTOR_SIZE: usize = 2048;

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
        let raw = ptr as *mut FFI_ArrowArrayStream;
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
            name: bind.get_parameter(0).to_string(),
            ptr: bind.get_parameter(1).to_uint64(),
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
            name: bind.get_parameter(0).to_string(),
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
    /// The batch being emitted, and how many of its rows are out already.
    current: Option<(RecordBatch, usize)>,
}

impl Cursor {
    /// The next slice of at most [`VECTOR_SIZE`] rows, or `None` at the end
    /// of the stream.
    fn next_slice(&mut self) -> Result<Option<RecordBatch>, BoxError> {
        loop {
            if let Some((batch, offset)) = &mut self.current
                && *offset < batch.num_rows()
            {
                let len = VECTOR_SIZE.min(batch.num_rows() - *offset);
                let slice = batch.slice(*offset, len);
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

/// `batch` with each top-level dictionary column decoded to its value type.
/// duckdb-rs binds a dictionary column as its value type but cannot write
/// dictionary arrays into a chunk.
fn decode_dictionaries(batch: RecordBatch) -> Result<RecordBatch, BoxError> {
    let schema = batch.schema();
    if !schema
        .fields()
        .iter()
        .any(|f| matches!(f.data_type(), DataType::Dictionary(..)))
    {
        return Ok(batch);
    }
    let (fields, columns): (Vec<Field>, Vec<ArrayRef>) = schema
        .fields()
        .iter()
        .zip(batch.columns())
        .map(|(field, column)| match field.data_type() {
            DataType::Dictionary(_, value) => Ok((
                field
                    .as_ref()
                    .clone()
                    .with_data_type(value.as_ref().clone()),
                cast(column, value)?,
            )),
            _ => Ok((field.as_ref().clone(), column.clone())),
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
        let tee = Registry::of(bind).get(&bind.get_parameter(0).to_string())?;
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
        Ok(ScanInit {
            cursor: Mutex::new(Cursor {
                reader: tee.cache.reader(true)?,
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

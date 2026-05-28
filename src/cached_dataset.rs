//! Cached Arrow dataset backed by an in-memory Vec or an on-disk IPC file.
//!
//! # Overview
//!
//! [`PyStreamCache`] accepts any upstream Arrow stream source (anything that
//! exposes `__arrow_c_stream__` in Python) and stores each `RecordBatch` in a
//! cache keyed by a monotonic `u64` batch index.  Two storage modes:
//!
//! - **Memory-only** (`disk_path` / `disk_capacity` omitted): batches are kept
//!   as `Arc<RecordBatch>` in a `Vec`.  Reads are zero-copy Arc clones; no IPC
//!   serialisation happens at all.
//! - **Disk** (`disk_path` + `disk_capacity` both provided): batches are
//!   serialised to Arrow IPC stream format and appended to a single temp file.
//!   A hot in-memory layer (`memory_capacity` bytes) avoids disk reads for
//!   recently ingested batches; entries that exceed the hot budget are read
//!   back from disk via positional I/O (no seek lock needed).
//!
//! Multiple independent [`PyStreamCacheReader`] handles can be obtained from a
//! single dataset, each maintaining its own read position.

use std::collections::VecDeque;
use std::io::{Cursor, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};

// ── system memory detection ───────────────────────────────────────────────────

fn total_system_memory() -> usize {
    let sys = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    );
    sys.total_memory() as usize
}

use arrow_array::RecordBatch;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_pyarrow::{IntoPyArrow, PyArrowType};
use arrow_schema::{ArrowError, Schema, SchemaRef};
use pyo3::exceptions::{PyIOError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_stub_gen::derive::*;
use std::ffi::CString;

// ── dataset counter ───────────────────────────────────────────────────────────

static DATASET_COUNTER: AtomicU64 = AtomicU64::new(0);

// ── cache tiers ──────────────────────────────────────────────────────────────

/// In-memory cache: batches stored as `Arc<RecordBatch>`, zero-copy reads.
struct MemoryCacheTier {
    batches: RwLock<Vec<Arc<RecordBatch>>>,
    /// Byte budget for all cached batches combined.
    capacity: usize,
    /// Bytes currently held across all cached batches.
    used: AtomicUsize,
}

/// Disk-backed write state (mutated only during ingestion).
struct DiskWriteState {
    file: std::fs::File,
    offset: u64,
}

/// FIFO in-memory read cache for the disk tier.
///
/// Entries are keyed by a monotonic batch index.  When the byte budget is
/// exceeded the oldest entry is evicted first — optimal for sequential replay
/// because batch N is always read before batch N+1.
struct HotLayer {
    /// `entries[0]` corresponds to batch `head_batch_idx`.
    entries: VecDeque<Option<(Arc<RecordBatch>, usize)>>,
    /// Global batch index of `entries[0]`.
    head_batch_idx: u64,
    /// Bytes currently held (sum of ipc_byte_len across all `Some` entries).
    used: usize,
    /// Byte budget.
    capacity: usize,
}

impl HotLayer {
    fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            head_batch_idx: 0,
            used: 0,
            capacity,
        }
    }

    /// Insert `batch` at `batch_idx`, evicting from the front until there is room.
    /// Returns `false` when the batch is larger than the entire budget (not cached,
    /// but callers must still write it to disk — returning false is not an error).
    fn try_insert(&mut self, batch_idx: u64, batch: Arc<RecordBatch>, ipc_len: usize) -> bool {
        while self.used + ipc_len > self.capacity {
            match self.entries.pop_front() {
                None => return false,
                Some(evicted) => {
                    if let Some((_, len)) = evicted {
                        self.used -= len;
                    }
                    self.head_batch_idx += 1;
                }
            }
        }
        debug_assert_eq!(batch_idx, self.head_batch_idx + self.entries.len() as u64);
        self.entries.push_back(Some((batch, ipc_len)));
        self.used += ipc_len;
        true
    }

    fn get(&self, batch_idx: u64) -> Option<Arc<RecordBatch>> {
        if batch_idx < self.head_batch_idx {
            return None;
        }
        let slot = (batch_idx - self.head_batch_idx) as usize;
        self.entries.get(slot)?.as_ref().map(|(arc, _)| arc.clone())
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.used = 0;
    }
}

/// Combined index + hot layer for the disk tier.
struct DiskIndex {
    /// `(file_offset, ipc_byte_length)` for each ingested batch.
    entries: Vec<(u64, usize)>,
    hot: HotLayer,
}

/// Disk-backed cache: append-only IPC file + optional hot in-memory layer.
struct DiskCacheTier {
    /// Subdirectory created under the caller's `disk_path`; removed on cleanup.
    dir_path: PathBuf,
    /// Serialised write position (locked separately; never contended — ingestion
    /// is always serialised by the outer `DatasetInner` mutex).
    write_state: Mutex<DiskWriteState>,
    /// Batch locations and hot-cache entries (RwLock: many concurrent readers,
    /// one writer at a time during ingestion).
    index: RwLock<DiskIndex>,
    /// File descriptor used for positional reads (pread-style, no seek lock).
    read_file: std::fs::File,
    /// Hard byte limit on the on-disk cache file.
    disk_capacity: u64,
    /// Bytes written to disk so far.  Relaxed ordering is safe: single writer
    /// (serialised by `DatasetInner` mutex), readers never inspect this field.
    disk_used: AtomicU64,
}

enum CacheTier {
    Memory(MemoryCacheTier),
    Disk(DiskCacheTier),
}

impl CacheTier {
    /// Store a batch.  Called while `DatasetInner`'s mutex is held.
    fn insert(&self, batch: RecordBatch) -> Result<(), ArrowError> {
        match self {
            CacheTier::Memory(m) => {
                let batch_size: usize = batch
                    .columns()
                    .iter()
                    .map(|c| c.get_array_memory_size())
                    .sum();
                let used = m.used.load(Ordering::Relaxed);
                if used + batch_size > m.capacity {
                    return Err(ArrowError::MemoryError(format!(
                        "Memory cache capacity ({} bytes) exceeded",
                        m.capacity
                    )));
                }
                m.used.fetch_add(batch_size, Ordering::Relaxed);
                m.batches.write().unwrap().push(Arc::new(batch));
                Ok(())
            }
            CacheTier::Disk(d) => {
                let bytes = serialize_batch(&batch)?;
                let length = bytes.len();
                // Enforce disk capacity before touching the file.
                let disk_prev = d.disk_used.load(Ordering::Relaxed);
                if disk_prev
                    .checked_add(length as u64)
                    .is_none_or(|end| end > d.disk_capacity)
                {
                    return Err(ArrowError::MemoryError(format!(
                        "Disk cache capacity ({} bytes) exceeded: {} bytes already written, \
                         cannot fit {} more bytes",
                        d.disk_capacity, disk_prev, length
                    )));
                }
                // Write to file and advance the offset.
                let offset = {
                    let mut ws = d.write_state.lock().unwrap();
                    let off = ws.offset;
                    ws.file
                        .write_all(&bytes)
                        .map_err(|e| other_arrow_err(format!("Disk write failed: {e}")))?;
                    // Flush so subsequent pread calls on the read_file FD see
                    // the written bytes (kernel buffer cache shared between FDs).
                    ws.file
                        .flush()
                        .map_err(|e| other_arrow_err(format!("Disk flush failed: {e}")))?;
                    ws.offset = ws
                        .offset
                        .checked_add(length as u64)
                        .ok_or_else(|| other_arrow_err("Cache file offset overflowed"))?;
                    off
                };
                d.disk_used.fetch_add(length as u64, Ordering::Relaxed);
                let batch_arc = Arc::new(batch);
                let mut idx = d.index.write().unwrap();
                let batch_idx = idx.entries.len() as u64;
                idx.hot.try_insert(batch_idx, batch_arc, length);
                idx.entries.push((offset, length));
                Ok(())
            }
        }
    }

    /// Retrieve batch `idx`, or `None` if not yet ingested.
    fn get(&self, idx: u64) -> Result<Option<Arc<RecordBatch>>, ArrowError> {
        match self {
            CacheTier::Memory(m) => Ok(m.batches.read().unwrap().get(idx as usize).cloned()),
            CacheTier::Disk(d) => {
                // Grab offset + hot copy while holding the read lock (brief).
                let (offset, length, maybe_hot) = {
                    let index = d.index.read().unwrap();
                    match index.entries.get(idx as usize) {
                        None => return Ok(None),
                        Some(&(off, len)) => (off, len, index.hot.get(idx)),
                    }
                }; // read lock released before any I/O

                if let Some(arc) = maybe_hot {
                    return Ok(Some(arc));
                }

                // Fall through to disk: positional read, no seek required.
                let mut buf = vec![0u8; length];
                pread_exact(&d.read_file, &mut buf, offset)
                    .map_err(|e| other_arrow_err(format!("Disk read failed: {e}")))?;
                deserialize_batch(&buf).map(|b| Some(Arc::new(b)))
            }
        }
    }

    /// Release in-memory data (hot layer and, for memory tiers, all batches).
    fn clear(&self) {
        match self {
            CacheTier::Memory(m) => m.batches.write().unwrap().clear(),
            CacheTier::Disk(d) => {
                d.index.write().unwrap().hot.clear();
            }
        }
    }

    /// Delete the on-disk subdirectory (no-op for memory tiers).
    fn cleanup_disk(&self) {
        if let CacheTier::Disk(d) = self {
            let _ = std::fs::remove_dir_all(&d.dir_path);
        }
    }
}

// ── platform-portable positional read ────────────────────────────────────────

/// Read exactly `buf.len()` bytes from `file` starting at `offset` without
/// modifying the file's seek position.  Concurrent callers are safe because
/// neither Unix `pread(2)` nor Windows `ReadFile` with OVERLAPPED updates the
/// file descriptor's offset.
fn pread_exact(file: &std::fs::File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut remaining = buf;
        let mut off = offset;
        while !remaining.is_empty() {
            let n = file.seek_read(remaining, off)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "unexpected EOF reading cache file",
                ));
            }
            off += n as u64;
            remaining = &mut remaining[n..];
        }
        Ok(())
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = (file, buf, offset);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "positional reads not supported on this platform",
        ))
    }
}

// ── error helpers ────────────────────────────────────────────────────────────

fn other_arrow_err(msg: impl std::fmt::Display) -> ArrowError {
    ArrowError::ExternalError(Box::new(std::io::Error::other(msg.to_string())))
}

/// Semantic error type for `without_gil` closures, so each kind maps to the
/// right Python exception at the boundary rather than everything becoming OSError.
enum BoundaryError {
    Value(String),
    Io(String),
    Memory(String),
    Runtime(String),
}

impl From<BoundaryError> for PyErr {
    fn from(e: BoundaryError) -> PyErr {
        match e {
            BoundaryError::Value(s) => PyValueError::new_err(s),
            BoundaryError::Io(s) => PyIOError::new_err(s),
            BoundaryError::Memory(s) => PyMemoryError::new_err(s),
            BoundaryError::Runtime(s) => PyRuntimeError::new_err(s),
        }
    }
}

/// Classify an [`ArrowError`] into the right [`BoundaryError`] variant.
fn arrow_to_boundary(e: ArrowError) -> BoundaryError {
    match e {
        ArrowError::MemoryError(msg) => BoundaryError::Memory(msg),
        ArrowError::InvalidArgumentError(msg) => BoundaryError::Value(msg),
        _ => BoundaryError::Io(e.to_string()),
    }
}

/// Convert an [`ArrowError`] into the appropriate Python exception.
fn arrow_to_py_err(e: ArrowError) -> PyErr {
    match e {
        ArrowError::MemoryError(msg) => PyMemoryError::new_err(msg),
        ArrowError::InvalidArgumentError(msg) => PyValueError::new_err(msg),
        _ => arrow_pyarrow::PyArrowException::new_err(e.to_string()),
    }
}

/// Export `schema` as an `"arrow_schema"` [`PyCapsule`] via Arrow FFI —
/// no pyarrow Python object is allocated.
fn to_schema_pycapsule<'py>(py: Python<'py>, schema: &Schema) -> PyResult<Bound<'py, PyCapsule>> {
    let ffi_schema =
        FFI_ArrowSchema::try_from(schema).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let name = CString::new("arrow_schema").map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyCapsule::new(py, ffi_schema, Some(name))
}

// ── IPC serialization ────────────────────────────────────────────────────────

fn serialize_batch(batch: &RecordBatch) -> Result<Vec<u8>, ArrowError> {
    let mut buf = Vec::new();
    {
        let mut writer = arrow_ipc::writer::StreamWriter::try_new(&mut buf, batch.schema_ref())?;
        writer.write(batch)?;
        writer.finish()?;
    }
    Ok(buf)
}

// Arrow's practical per-batch limit (int32 offset arrays cap at ~2^31 elements).
const MAX_BATCH_BYTES: usize = 2 * 1024 * 1024 * 1024;

fn deserialize_batch(bytes: &[u8]) -> Result<RecordBatch, ArrowError> {
    if bytes.len() > MAX_BATCH_BYTES {
        return Err(other_arrow_err(format!(
            "Batch size {} exceeds maximum of {} bytes",
            bytes.len(),
            MAX_BATCH_BYTES
        )));
    }
    let mut reader = arrow_ipc::reader::StreamReader::try_new(Cursor::new(bytes), None)?;
    reader
        .next()
        .ok_or_else(|| other_arrow_err("Empty IPC stream in cache entry"))?
}

// ── GIL management ───────────────────────────────────────────────────────────

fn without_gil<T, F: FnOnce() -> T>(_py: Python<'_>, f: F) -> T {
    struct RestoreGuard(*mut pyo3::ffi::PyThreadState);
    impl Drop for RestoreGuard {
        fn drop(&mut self) {
            unsafe { pyo3::ffi::PyEval_RestoreThread(self.0) };
        }
    }
    #[cfg(not(Py_GIL_DISABLED))]
    let _guard = RestoreGuard(unsafe { pyo3::ffi::PyEval_SaveThread() });
    f()
}

fn with_gil_acquired<T, F: FnOnce() -> T>(f: F) -> T {
    #[cfg(not(Py_GIL_DISABLED))]
    {
        struct ReleaseGuard(pyo3::ffi::PyGILState_STATE);
        impl Drop for ReleaseGuard {
            fn drop(&mut self) {
                unsafe { pyo3::ffi::PyGILState_Release(self.0) };
            }
        }
        let _guard = ReleaseGuard(unsafe { pyo3::ffi::PyGILState_Ensure() });
        f()
    }
    #[cfg(Py_GIL_DISABLED)]
    f()
}

// ── shared ingestion state ───────────────────────────────────────────────────

struct DatasetInner {
    cache: Arc<CacheTier>,
    upstream: Option<Box<dyn arrow_array::RecordBatchReader + Send>>,
    ingested_count: u64,
    upstream_exhausted: bool,
    closed: bool,
}

impl DatasetInner {
    fn ingest_up_to(&mut self, target_index: u64) -> Result<bool, ArrowError> {
        if self.closed {
            return Err(ArrowError::InvalidArgumentError(
                "Dataset has been closed".into(),
            ));
        }
        while self.ingested_count <= target_index {
            if self.upstream_exhausted {
                return Ok(false);
            }
            let next_result = with_gil_acquired(|| self.upstream.as_mut().and_then(|u| u.next()));
            let batch = match next_result {
                None => {
                    self.upstream_exhausted = true;
                    return Ok(false);
                }
                Some(Err(e)) => return Err(e),
                Some(Ok(b)) => b,
            };
            self.cache.insert(batch)?;
            self.ingested_count += 1;
        }
        Ok(true)
    }
}

// ── StreamCacheReaderImpl ──────────────────────────────────────────────────

struct StreamCacheReaderImpl {
    schema: SchemaRef,
    inner: Arc<Mutex<DatasetInner>>,
    current_index: u64,
}

impl Iterator for StreamCacheReaderImpl {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current_index;

        // Ensure the batch has been ingested and clone the cache handle.
        let cache = {
            let mut inner = match self.inner.lock() {
                Ok(g) => g,
                Err(e) => return Some(Err(other_arrow_err(format!("Mutex poisoned: {e}")))),
            };
            if inner.closed {
                return Some(Err(ArrowError::InvalidArgumentError(
                    "Dataset has been closed".into(),
                )));
            }
            match inner.ingest_up_to(idx) {
                Err(e) => return Some(Err(e)),
                Ok(false) => return None,
                Ok(true) => {}
            }
            inner.cache.clone()
        }; // mutex released before I/O

        // Fetch from cache; no lock held — safe for concurrent readers.
        match cache.get(idx) {
            Err(e) => Some(Err(e)),
            Ok(None) => {
                let closed = self.inner.lock().map(|g| g.closed).unwrap_or(false);
                if closed {
                    Some(Err(ArrowError::InvalidArgumentError(
                        "Dataset has been closed".into(),
                    )))
                } else {
                    // Should not happen: ingest_up_to returned Ok(true).
                    Some(Err(other_arrow_err(format!(
                        "Batch {idx} disappeared from the cache unexpectedly"
                    ))))
                }
            }
            Ok(Some(arc)) => {
                self.current_index += 1;
                // Clone the RecordBatch (cheap: clones Arc pointers to buffers).
                Some(Ok((*arc).clone()))
            }
        }
    }
}

impl arrow_array::RecordBatchReader for StreamCacheReaderImpl {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

// ── PyStreamCacheReader ────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "StreamCacheReader", frozen)]
pub struct PyStreamCacheReader(Mutex<Option<StreamCacheReaderImpl>>);

impl PyStreamCacheReader {
    fn new(impl_: StreamCacheReaderImpl) -> Self {
        Self(Mutex::new(Some(impl_)))
    }

    fn to_stream_pycapsule<'py>(
        py: Python<'py>,
        reader: StreamCacheReaderImpl,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let boxed: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(reader);
        match requested_schema {
            None => {
                // Fast path: export directly via Arrow C Stream FFI, no Python allocation.
                let ffi_stream = FFI_ArrowArrayStream::new(boxed);
                let name = CString::new("arrow_array_stream")
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
                PyCapsule::new(py, ffi_stream, Some(name))
            }
            Some(schema) => {
                // Schema-casting path: let pyarrow handle the requested_schema negotiation.
                let py_reader = boxed.into_pyarrow(py)?;
                py_reader
                    .call_method1("__arrow_c_stream__", (schema,))?
                    .cast_into::<PyCapsule>()
                    .map_err(|_| {
                        PyTypeError::new_err(
                            "pyarrow __arrow_c_stream__ did not return a PyCapsule",
                        )
                    })
            }
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStreamCacheReader {
    #[pyo3(signature = (requested_schema = None))]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let reader = self
            .0
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        Self::to_stream_pycapsule(py, reader, requested_schema.map(|c| c.into_any()))
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let inner = self.0.lock().unwrap();
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        to_schema_pycapsule(py, reader.schema.as_ref())
    }

    #[gen_stub(override_return_type(type_repr = "pa.Schema", imports = ("pyarrow as pa",)))]
    #[getter]
    fn schema(&self) -> PyResult<PyArrowType<Schema>> {
        let inner = self.0.lock().unwrap();
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        Ok(PyArrowType((*reader.schema).clone()))
    }

    #[getter]
    fn closed(&self) -> bool {
        self.0.lock().unwrap().is_none()
    }

    fn __iter__<'py>(slf: PyRef<'py, Self>) -> PyRef<'py, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn cast<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        target_schema: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let impl_ = self
            .0
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        let new_reader = PyStreamCacheReader::new(impl_);
        let kwargs = pyo3::types::PyDict::new(py);
        kwargs.set_item("schema", target_schema)?;
        let py_reader = Bound::new(py, new_reader)?;
        py.import("pyarrow")?
            .getattr("RecordBatchReader")?
            .getattr("from_stream")?
            .call((py_reader,), Some(&kwargs))
    }

    #[gen_stub(override_return_type(type_repr = "pa.RecordBatch", imports = ("pyarrow as pa",)))]
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyArrowType<RecordBatch>>> {
        let mut guard = self.0.lock().unwrap();
        let impl_ = match guard.as_mut() {
            None => {
                return Err(PyValueError::new_err("Reader already consumed"));
            }
            Some(r) => r,
        };
        let result = without_gil(py, || impl_.next());
        match result {
            None => Ok(None),
            Some(Err(e)) => Err(arrow_to_py_err(e)),
            Some(Ok(batch)) => Ok(Some(PyArrowType(batch))),
        }
    }
}

// ── PyCastingStreamCache ──────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "CastingStreamCache", frozen)]
pub struct PyCastingStreamCache {
    inner: Arc<Mutex<DatasetInner>>,
    source_schema: SchemaRef,
    target_schema: SchemaRef,
}

impl PyCastingStreamCache {
    fn make_reader_impl(&self, py: Python<'_>) -> PyResult<StreamCacheReaderImpl> {
        without_gil(py, || {
            let inner = self
                .inner
                .lock()
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))?;
            if inner.closed {
                return Err(BoundaryError::Value("Dataset has been closed".into()));
            }
            Ok::<_, BoundaryError>(StreamCacheReaderImpl {
                schema: self.source_schema.clone(),
                inner: self.inner.clone(),
                current_index: 0,
            })
        })
        .map_err(PyErr::from)
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCastingStreamCache {
    #[gen_stub(override_return_type(type_repr = "pa.Schema", imports = ("pyarrow as pa",)))]
    #[getter]
    pub fn schema(&self) -> PyResult<PyArrowType<Schema>> {
        Ok(PyArrowType((*self.target_schema).clone()))
    }

    #[pyo3(signature = (requested_schema = None))]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let impl_ = self.make_reader_impl(py)?;
        let effective_schema: Option<Bound<'py, PyAny>> = if requested_schema.is_some() {
            requested_schema.map(|c| c.into_any())
        } else {
            Some(to_schema_pycapsule(py, self.target_schema.as_ref())?.into_any())
        };
        PyStreamCacheReader::to_stream_pycapsule(py, impl_, effective_schema)
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, self.target_schema.as_ref())
    }

    #[gen_stub(override_return_type(type_repr = "CastingStreamCache", imports = ()))]
    pub fn cast(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        target_schema: PyArrowType<Schema>,
    ) -> PyResult<PyCastingStreamCache> {
        Ok(PyCastingStreamCache {
            inner: self.inner.clone(),
            source_schema: self.source_schema.clone(),
            target_schema: Arc::new(target_schema.0),
        })
    }
}

// ── PyStreamCache ─────────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "StreamCache", frozen)]
pub struct PyStreamCache {
    schema: SchemaRef,
    inner: Arc<Mutex<DatasetInner>>,
}

impl Drop for PyStreamCache {
    fn drop(&mut self) {
        if let Ok(inner) = self.inner.lock()
            && !inner.closed
        {
            let cache = inner.cache.clone();
            drop(inner);
            cache.cleanup_disk();
            cache.clear();
        }
    }
}

#[gen_stub_pymethods]
#[pymethods]
impl PyStreamCache {
    #[new]
    #[pyo3(signature = (reader, memory_capacity = None, disk_path = None, disk_capacity = None))]
    pub fn new(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        reader: PyArrowType<ArrowArrayStreamReader>,
        memory_capacity: Option<usize>,
        disk_path: Option<String>,
        disk_capacity: Option<u64>,
    ) -> PyResult<Self> {
        let upstream: Box<dyn arrow_array::RecordBatchReader + Send> = Box::new(reader.0);
        let schema = upstream.schema();

        let cache = match (disk_path, disk_capacity) {
            (Some(path), Some(capacity)) => {
                let id = DATASET_COUNTER.fetch_add(1, Ordering::Relaxed);
                let dir_path = PathBuf::from(&path).join(format!("_{id}"));

                // Create the subdirectory and both file descriptors while the
                // GIL is released (pure OS I/O — no Python objects touched).
                without_gil(py, || {
                    std::fs::create_dir_all(&dir_path).map_err(|e| {
                        BoundaryError::Io(format!("Failed to create cache directory: {e}"))
                    })?;
                    let file_path = dir_path.join("cache.arrow");
                    let mut open_opts = std::fs::OpenOptions::new();
                    open_opts.write(true).create_new(true);
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::OpenOptionsExt;
                        open_opts.mode(0o600);
                    }
                    let write_file = open_opts.open(&file_path).map_err(|e| {
                        BoundaryError::Io(format!("Failed to create cache file: {e}"))
                    })?;
                    let read_file = std::fs::File::open(&file_path).map_err(|e| {
                        BoundaryError::Io(format!("Failed to open cache file for reading: {e}"))
                    })?;

                    let hot_capacity = memory_capacity.unwrap_or_else(|| {
                        // Cap at half of system RAM so multiple caches don't
                        // collectively exhaust memory; floor at 64 MiB.
                        (total_system_memory() / 2).max(64 * 1024 * 1024)
                    });
                    Ok::<_, BoundaryError>(CacheTier::Disk(DiskCacheTier {
                        dir_path,
                        write_state: Mutex::new(DiskWriteState {
                            file: write_file,
                            offset: 0,
                        }),
                        index: RwLock::new(DiskIndex {
                            entries: Vec::new(),
                            hot: HotLayer::new(hot_capacity),
                        }),
                        read_file,
                        disk_capacity: capacity,
                        disk_used: AtomicU64::new(0),
                    }))
                })
                .map_err(PyErr::from)?
            }
            (None, None) => {
                let capacity = memory_capacity.unwrap_or_else(|| {
                    // Default to 10% of system RAM, floor at 64 MiB.
                    (total_system_memory() / 10).max(64 * 1024 * 1024)
                });
                CacheTier::Memory(MemoryCacheTier {
                    batches: RwLock::new(Vec::new()),
                    capacity,
                    used: AtomicUsize::new(0),
                })
            }
            _ => {
                return Err(PyValueError::new_err(
                    "disk_path and disk_capacity must both be provided, or both omitted",
                ));
            }
        };

        let inner = DatasetInner {
            cache: Arc::new(cache),
            upstream: Some(upstream),
            ingested_count: 0,
            upstream_exhausted: false,
            closed: false,
        };

        Ok(Self {
            schema,
            inner: Arc::new(Mutex::new(inner)),
        })
    }

    #[gen_stub(override_return_type(type_repr = "pa.Schema", imports = ("pyarrow as pa",)))]
    #[getter]
    pub fn schema(&self) -> PyResult<PyArrowType<Schema>> {
        Ok(PyArrowType((*self.schema).clone()))
    }

    #[pyo3(signature = (from_start = true))]
    pub fn reader(&self, py: Python<'_>, from_start: bool) -> PyResult<PyStreamCacheReader> {
        without_gil(py, || {
            let inner = self
                .inner
                .lock()
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))?;
            if inner.closed {
                return Err(BoundaryError::Value("Dataset has been closed".into()));
            }
            let start_index = if from_start { 0 } else { inner.ingested_count };
            Ok::<_, BoundaryError>(PyStreamCacheReader::new(StreamCacheReaderImpl {
                schema: self.schema.clone(),
                inner: self.inner.clone(),
                current_index: start_index,
            }))
        })
        .map_err(PyErr::from)
    }

    pub fn __iter__(&self, py: Python<'_>) -> PyResult<PyStreamCacheReader> {
        self.reader(py, true)
    }

    #[pyo3(signature = (requested_schema = None))]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let reader = self.reader(py, true)?;
        let impl_ = reader
            .0
            .lock()
            .unwrap()
            .take()
            .expect("freshly created reader cannot be closed");
        PyStreamCacheReader::to_stream_pycapsule(py, impl_, requested_schema.map(|c| c.into_any()))
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, self.schema.as_ref())
    }

    pub fn cast(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        target_schema: PyArrowType<Schema>,
    ) -> PyResult<PyCastingStreamCache> {
        Ok(PyCastingStreamCache {
            inner: self.inner.clone(),
            source_schema: self.schema.clone(),
            target_schema: Arc::new(target_schema.0),
        })
    }

    pub fn ingest_all(&self, py: Python<'_>) -> PyResult<u64> {
        without_gil(py, || {
            let mut inner = self
                .inner
                .lock()
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))?;
            inner.ingest_up_to(u64::MAX).map_err(arrow_to_boundary)?;
            Ok::<_, BoundaryError>(inner.ingested_count)
        })
        .map_err(PyErr::from)
    }

    #[getter]
    pub fn ingested_count(&self, py: Python<'_>) -> PyResult<u64> {
        without_gil(py, || {
            self.inner
                .lock()
                .map(|g| g.ingested_count)
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))
        })
        .map_err(PyErr::from)
    }

    #[getter]
    pub fn upstream_exhausted(&self, py: Python<'_>) -> PyResult<bool> {
        without_gil(py, || {
            self.inner
                .lock()
                .map(|g| g.upstream_exhausted)
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))
        })
        .map_err(PyErr::from)
    }

    pub fn close(&self, py: Python<'_>) -> PyResult<()> {
        without_gil(py, || {
            let mut inner = self
                .inner
                .lock()
                .map_err(|e| BoundaryError::Runtime(format!("Internal mutex error: {e}")))?;
            if inner.closed {
                return Ok(());
            }
            inner.closed = true;
            let cache = inner.cache.clone();
            drop(inner);
            cache.cleanup_disk();
            cache.clear();
            Ok::<_, BoundaryError>(())
        })
        .map_err(PyErr::from)
    }
}

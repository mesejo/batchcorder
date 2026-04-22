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

use arrow_array::{Array, ArrayRef, RecordBatch, StructArray};
use arrow_schema::{ArrowError, Field, SchemaRef};
use pyo3::exceptions::{PyIOError, PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_arrow::error::{PyArrowError, PyArrowResult};
use pyo3_arrow::export::{Arro3RecordBatch, Arro3Schema};
use pyo3_arrow::ffi::{ArrayIterator, to_schema_pycapsule, to_stream_pycapsule};
use pyo3_arrow::{PyRecordBatchReader, PySchema};
use pyo3_stub_gen::derive::*;

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

/// Combined index + hot layer for the disk tier.
struct DiskIndex {
    /// `(file_offset, byte_length)` for each ingested batch.
    entries: Vec<(u64, usize)>,
    /// In-memory copy of a batch, or `None` when the hot budget was exceeded.
    hot: Vec<Option<Arc<RecordBatch>>>,
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
    /// Byte budget for the hot in-memory layer.
    hot_capacity: usize,
    /// Bytes currently held in the hot layer.
    hot_used: AtomicUsize,
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
                // Try to keep a hot copy if the budget allows.
                let hot = if d.hot_used.load(Ordering::Relaxed) + length <= d.hot_capacity {
                    d.hot_used.fetch_add(length, Ordering::Relaxed);
                    Some(Arc::new(batch))
                } else {
                    None
                };
                let mut idx = d.index.write().unwrap();
                idx.entries.push((offset, length));
                idx.hot.push(hot);
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
                        Some(&(off, len)) => {
                            let hot = index
                                .hot
                                .get(idx as usize)
                                .and_then(|o| o.as_ref())
                                .cloned();
                            (off, len, hot)
                        }
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
                let mut idx = d.index.write().unwrap();
                idx.hot.iter_mut().for_each(|h| *h = None);
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

/// Convert an [`ArrowError`] into the right [`PyArrowError`] variant so
/// iterator callers receive a meaningful Python exception type.
fn arrow_to_pyarrow_error(e: ArrowError) -> PyArrowError {
    match e {
        ArrowError::MemoryError(msg) => PyArrowError::PyErr(PyMemoryError::new_err(msg)),
        ArrowError::InvalidArgumentError(msg) => PyArrowError::PyErr(PyValueError::new_err(msg)),
        _ => PyArrowError::ArrowError(e),
    }
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
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let schema = reader.schema.clone();
        let array_iter = reader.map(|maybe_batch| {
            let arr: ArrayRef = Arc::new(StructArray::from(maybe_batch?));
            Ok(arr)
        });
        let array_reader = Box::new(ArrayIterator::new(
            array_iter,
            Field::new_struct("", schema.fields().clone(), false)
                .with_metadata(schema.metadata.clone())
                .into(),
        ));
        to_stream_pycapsule(py, array_reader, requested_schema)
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
    ) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let reader =
            self.0.lock().unwrap().take().ok_or_else(|| {
                PyArrowError::PyErr(PyValueError::new_err("Reader already consumed"))
            })?;
        Self::to_stream_pycapsule(py, reader, requested_schema)
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let inner = self.0.lock().unwrap();
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyArrowError::PyErr(PyValueError::new_err("Reader already consumed")))?;
        to_schema_pycapsule(py, reader.schema.as_ref())
    }

    #[gen_stub(override_return_type(type_repr = "arro3.core.Schema", imports = ("arro3.core",)))]
    #[getter]
    fn schema(&self) -> PyResult<Arro3Schema> {
        let inner = self.0.lock().unwrap();
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        Ok(PySchema::new(reader.schema.clone()).into())
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

    #[gen_stub(override_return_type(type_repr = "arro3.core.RecordBatch", imports = ("arro3.core",)))]
    fn __next__(&self, py: Python<'_>) -> PyArrowResult<Option<Arro3RecordBatch>> {
        let mut guard = self.0.lock().unwrap();
        let impl_ = match guard.as_mut() {
            None => {
                return Err(PyArrowError::PyErr(PyValueError::new_err(
                    "Reader already consumed",
                )));
            }
            Some(r) => r,
        };
        let result = without_gil(py, || impl_.next());
        match result {
            None => Ok(None),
            Some(Err(e)) => Err(arrow_to_pyarrow_error(e)),
            Some(Ok(batch)) => Ok(Some(Arro3RecordBatch::from(batch))),
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
    #[gen_stub(override_return_type(type_repr = "arro3.core.Schema", imports = ("arro3.core",)))]
    #[getter]
    pub fn schema(&self) -> PyResult<Arro3Schema> {
        Ok(PySchema::new(self.target_schema.clone()).into())
    }

    #[pyo3(signature = (requested_schema = None))]
    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let impl_ = self.make_reader_impl(py).map_err(PyArrowError::PyErr)?;
        let effective_schema = if requested_schema.is_some() {
            requested_schema
        } else {
            Some(to_schema_pycapsule(py, self.target_schema.as_ref())?)
        };
        PyStreamCacheReader::to_stream_pycapsule(py, impl_, effective_schema)
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyArrowResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, self.target_schema.as_ref())
    }

    #[gen_stub(override_return_type(type_repr = "CastingStreamCache", imports = ()))]
    pub fn cast(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        target_schema: PySchema,
    ) -> PyResult<PyCastingStreamCache> {
        Ok(PyCastingStreamCache {
            inner: self.inner.clone(),
            source_schema: self.source_schema.clone(),
            target_schema: target_schema.into_inner(),
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
        reader: PyRecordBatchReader,
        memory_capacity: Option<usize>,
        disk_path: Option<String>,
        disk_capacity: Option<u64>,
    ) -> PyResult<Self> {
        let upstream = reader.into_reader()?;
        let schema = upstream.schema();

        let cache = match (disk_path, disk_capacity) {
            (Some(path), Some(_capacity)) => {
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
                            hot: Vec::new(),
                        }),
                        read_file,
                        hot_capacity,
                        hot_used: AtomicUsize::new(0),
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

    #[gen_stub(override_return_type(type_repr = "arro3.core.Schema", imports = ("arro3.core",)))]
    #[getter]
    pub fn schema(&self) -> PyResult<Arro3Schema> {
        Ok(PySchema::new(self.schema.clone()).into())
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
    ) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let reader = self.reader(py, true).map_err(PyArrowError::PyErr)?;
        let impl_ = reader
            .0
            .lock()
            .unwrap()
            .take()
            .expect("freshly created reader cannot be closed");
        PyStreamCacheReader::to_stream_pycapsule(py, impl_, requested_schema)
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyArrowResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, self.schema.as_ref())
    }

    pub fn cast(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        target_schema: PySchema,
    ) -> PyResult<PyCastingStreamCache> {
        Ok(PyCastingStreamCache {
            inner: self.inner.clone(),
            source_schema: self.schema.clone(),
            target_schema: target_schema.into_inner(),
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

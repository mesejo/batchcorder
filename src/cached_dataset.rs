//! Cached Arrow dataset backed by Foyer.
//!
//! # Overview
//!
//! [`PyStreamCache`] accepts any upstream Arrow stream source (anything that
//! exposes `__arrow_c_stream__` in Python) and stores each `RecordBatch` in a
//! Foyer cache keyed by a monotonic `u64` batch index.  The IPC stream format
//! is used for on-cache serialization so the data is schema-agnostic.
//!
//! Two storage modes are supported:
//!
//! - **Memory-only** (`disk_path` / `disk_capacity` omitted): batches are kept
//!   in a Foyer in-memory cache only.  No files are created on disk.
//! - **Hybrid** (`disk_path` + `disk_capacity` both provided): batches that
//!   are evicted from the memory tier are written to a Foyer block-device
//!   backed by the filesystem, allowing the working set to exceed available
//!   RAM.
//!
//! Multiple independent [`PyStreamCacheReader`] handles can be obtained from
//! a single dataset, each maintaining its own read position.  A reader that
//! requests a batch not yet ingested will trigger lazy ingestion up to that
//! index.
//!
//! # Eviction caveat
//!
//! Foyer evicts cache entries under memory/disk pressure.  If an entry is
//! evicted before a reader reaches it, that reader will return an error.
//! Size the cache to hold at least as many batches as the span between the
//! slowest and fastest concurrent reader.

use std::collections::HashMap;
use std::io::Cursor;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

// ── system memory detection ───────────────────────────────────────────────────

/// Return the total physical RAM of this machine in bytes.
///
/// Used as the default `memory_capacity` for memory-only caches so they fail
/// only when a `tee`-style copy would also exhaust available memory.
fn total_system_memory() -> usize {
    let sys = sysinfo::System::new_with_specifics(
        sysinfo::RefreshKind::nothing()
            .with_memory(sysinfo::MemoryRefreshKind::nothing().with_ram()),
    );
    sys.total_memory() as usize
}

use arrow_array::{ArrayRef, RecordBatch, StructArray};
use arrow_schema::{ArrowError, Field, SchemaRef};
use foyer::{
    BlockEngineConfig, Cache, CacheBuilder, DeviceBuilder, FsDeviceBuilder, HybridCache,
    HybridCacheBuilder, HybridCachePolicy,
};
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_arrow::error::{PyArrowError, PyArrowResult};
use pyo3_arrow::export::{Arro3RecordBatch, Arro3Schema};
use pyo3_arrow::ffi::{ArrayIterator, to_schema_pycapsule, to_stream_pycapsule};
use pyo3_arrow::{PyRecordBatchReader, PySchema};
use pyo3_stub_gen::derive::*;
use tokio::runtime::Runtime;

// ── dataset counter ───────────────────────────────────────────────────────────

/// Monotonically increasing counter used to give each [`PyStreamCache`] a
/// unique Foyer device subdirectory under the caller-supplied `disk_path`.
static DATASET_COUNTER: AtomicU64 = AtomicU64::new(0);

// ── cache tier ───────────────────────────────────────────────────────────────

/// Abstraction over memory-only and hybrid memory+disk Foyer caches.
///
/// Both variants expose the same insert/get/clear interface so the rest of the
/// code does not need to branch on the storage mode.
enum CacheTier {
    Memory(Cache<u64, Vec<u8>>),
    Hybrid(HybridCache<u64, Vec<u8>>),
}

impl CacheTier {
    /// Insert a batch into the cache.  Synchronous for both variants.
    fn insert(&self, key: u64, value: Vec<u8>) {
        match self {
            CacheTier::Memory(c) => {
                c.insert(key, value);
            }
            CacheTier::Hybrid(c) => {
                c.insert(key, value);
            }
        }
    }

    /// Fetch the raw IPC bytes for `key`, or `None` if evicted / not yet
    /// inserted.  Returns `Err` only when the hybrid disk tier fails.
    async fn get_bytes(&self, key: &u64) -> Result<Option<Vec<u8>>, String> {
        match self {
            CacheTier::Memory(c) => Ok(c.get(key).map(|e| e.value().clone())),
            CacheTier::Hybrid(c) => c
                .get(key)
                .await
                .map(|opt| opt.map(|e| e.value().clone()))
                .map_err(|e| e.to_string()),
        }
    }

    /// Clear all entries from the cache.
    async fn clear(&self) {
        match self {
            CacheTier::Memory(c) => c.clear(),
            CacheTier::Hybrid(c) => {
                let _ = c.clear().await;
            }
        }
    }
}

// ── error helpers ────────────────────────────────────────────────────────────

/// Wrap an arbitrary `Display` message as an [`ArrowError`].
fn other_arrow_err(msg: impl std::fmt::Display) -> ArrowError {
    ArrowError::ExternalError(Box::new(std::io::Error::other(msg.to_string())))
}

// ── IPC serialization ────────────────────────────────────────────────────────

/// Serialize a [`RecordBatch`] to Arrow IPC stream-format bytes.
///
/// The schema is embedded in the IPC stream so each cached blob is
/// self-contained and can be deserialized without external metadata.
fn serialize_batch(batch: &RecordBatch) -> Result<Vec<u8>, ArrowError> {
    let mut buf = Vec::new();
    {
        let mut writer = arrow_ipc::writer::StreamWriter::try_new(&mut buf, batch.schema_ref())?;
        writer.write(batch)?;
        writer.finish()?;
    }
    Ok(buf)
}

/// Deserialize the first (and only) [`RecordBatch`] from Arrow IPC stream bytes.
fn deserialize_batch(bytes: &[u8]) -> Result<RecordBatch, ArrowError> {
    let mut reader = arrow_ipc::reader::StreamReader::try_new(Cursor::new(bytes), None)?;
    reader
        .next()
        .ok_or_else(|| other_arrow_err("Empty IPC stream in cache entry"))?
}

// ── GIL management ───────────────────────────────────────────────────────────

/// Release the GIL, run `f`, then re-acquire it.
///
/// `pyo3::Python::allow_threads` is not available under `abi3` targets because
/// pyo3 cannot guarantee the GIL exists at compile time when targeting
/// Python 3.13+, which supports a no-GIL build.  This helper calls
/// `PyEval_SaveThread` / `PyEval_RestoreThread` directly; both are part of
/// the stable C API (`Py_LIMITED_API`) and work correctly under abi3.
///
/// The calling code must already hold the GIL (ensured by the `Python<'_>`
/// token).  Any Python calls made inside `f` must re-acquire the GIL via
/// `Python::with_gil` — `pyo3_arrow`'s upstream reader wrapper does this
/// automatically.
fn without_gil<T, F: FnOnce() -> T>(_py: Python<'_>, f: F) -> T {
    struct RestoreGuard(*mut pyo3::ffi::PyThreadState);
    impl Drop for RestoreGuard {
        fn drop(&mut self) {
            unsafe { pyo3::ffi::PyEval_RestoreThread(self.0) };
        }
    }
    // On free-threaded Python (Py_GIL_DISABLED) there is no GIL to release.
    #[cfg(not(Py_GIL_DISABLED))]
    let _guard = RestoreGuard(unsafe { pyo3::ffi::PyEval_SaveThread() });
    f()
}

/// Acquire the GIL, run `f`, then release it back to its previous state.
///
/// This is the inverse of [`without_gil`]: safe to call from any thread,
/// including C extension threads that have no Python thread state yet.
/// Uses `PyGILState_Ensure`/`PyGILState_Release` (stable C API, abi3-safe)
/// which handle the "already held" case correctly via a saved state cookie.
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

/// Mutable state shared between a [`PyStreamCache`] and all of its readers.
///
/// Protected by `Arc<Mutex<DatasetInner>>` so readers and the dataset can
/// share it across threads.
struct DatasetInner {
    /// Foyer cache: `batch_index → IPC bytes`.
    cache: Arc<CacheTier>,
    /// Upstream source; `None` once it is fully exhausted or has never been set.
    upstream: Option<Box<dyn arrow_array::RecordBatchReader + Send>>,
    /// How many batches have been pulled from `upstream` and inserted into the cache.
    ingested_count: u64,
    /// `true` once `upstream.next()` returns `None`.
    upstream_exhausted: bool,
    /// Current read position of every live reader: `consumer_id → batch_index`.
    consumer_positions: HashMap<u64, u64>,
    /// Monotonically increasing counter used to assign unique consumer IDs.
    next_consumer_id: u64,
    /// Path of the Foyer device subdirectory (a child of the caller-supplied
    /// `disk_path`).  `None` for memory-only caches.  Only this subdirectory
    /// is removed on close/drop; the parent directory is left untouched.
    foyer_path: Option<PathBuf>,
    /// `true` once [`PyStreamCache::close`] has been called or the dataset
    /// has been dropped.  Prevents double-cleanup and signals active readers
    /// that the dataset is no longer usable.
    closed: bool,
}

impl DatasetInner {
    /// Pull batches from the upstream reader into the cache until `target_index`
    /// is available (inclusive).
    ///
    /// Returns `true` if `target_index` is now in the cache, or `false` if the
    /// upstream was exhausted before reaching it.
    fn ingest_up_to(&mut self, target_index: u64) -> Result<bool, ArrowError> {
        if self.closed {
            return Err(other_arrow_err("Dataset has been closed"));
        }
        while self.ingested_count <= target_index {
            if self.upstream_exhausted {
                return Ok(false);
            }
            // The upstream is backed by a Python C stream whose get_next function
            // requires the GIL.  Acquire it here so this call is safe from any
            // thread context — including DuckDB scanner threads that have no
            // Python thread state (the Arrow C stream get_next path bypasses
            // #[pymethods] and our without_gil wrapper entirely).
            let next_result = with_gil_acquired(|| self.upstream.as_mut().and_then(|u| u.next()));
            let batch = match next_result {
                None => {
                    self.upstream_exhausted = true;
                    return Ok(false);
                }
                Some(Err(e)) => return Err(e),
                Some(Ok(b)) => b,
            };
            let idx = self.ingested_count;
            let bytes = serialize_batch(&batch)?;
            // Both cache variants insert synchronously.  For the hybrid tier
            // this writes to memory immediately and schedules disk persistence
            // in the background.
            self.cache.insert(idx, bytes);
            self.ingested_count += 1;
        }
        Ok(true)
    }
}

// ── StreamCacheReaderImpl ──────────────────────────────────────────────────

/// The concrete Rust iterator that backs [`PyStreamCacheReader`].
///
/// Implements [`arrow_array::RecordBatchReader`] so it can be handed to any
/// Rust API or exported via the Arrow C Stream interface.
struct StreamCacheReaderImpl {
    /// Schema shared with the owning dataset (immutable, cheap to clone).
    schema: SchemaRef,
    /// Shared dataset state (ingestion, cache handle, consumer registry).
    inner: Arc<Mutex<DatasetInner>>,
    /// Index of the next batch this reader will return.
    current_index: u64,
    /// Unique ID registered in `DatasetInner::consumer_positions`.
    consumer_id: u64,
    /// Tokio runtime used to drive `HybridCache::get`.
    runtime: Arc<Runtime>,
}

impl Iterator for StreamCacheReaderImpl {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current_index;

        // ── Step 1: ensure the batch has been ingested ──────────────────────
        let cache = {
            let mut inner = match self.inner.lock() {
                Ok(g) => g,
                Err(e) => return Some(Err(other_arrow_err(format!("Mutex poisoned: {e}")))),
            };
            if inner.closed {
                return Some(Err(other_arrow_err("Dataset has been closed")));
            }
            match inner.ingest_up_to(idx) {
                Err(e) => return Some(Err(e)),
                Ok(false) => return None, // upstream exhausted before idx
                Ok(true) => {}
            }
            // Update this reader's position in the registry.
            inner.consumer_positions.insert(self.consumer_id, idx);
            inner.cache.clone()
        };

        // ── Step 2: fetch from cache and deserialize (lock released) ─────────
        let bytes = self.runtime.block_on(async { cache.get_bytes(&idx).await });

        match bytes {
            Err(e) => Some(Err(other_arrow_err(format!("Cache get failed: {e}")))),
            Ok(None) => {
                // Distinguish a post-close access from a genuine capacity eviction.
                let closed = self.inner.lock().map(|g| g.closed).unwrap_or(false);
                if closed {
                    Some(Err(other_arrow_err("Dataset has been closed")))
                } else {
                    Some(Err(other_arrow_err(format!(
                        "Batch {idx} was evicted from the cache before it could be read. \
                         Increase cache capacity so it can hold all live consumer positions."
                    ))))
                }
            }
            // ── Step 3: deserialize and advance position ──────────────────────
            Ok(Some(b)) => {
                self.current_index += 1;
                Some(deserialize_batch(&b))
            }
        }
    }
}

impl arrow_array::RecordBatchReader for StreamCacheReaderImpl {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Drop for StreamCacheReaderImpl {
    /// Deregister this reader from the consumer registry on drop so the dataset
    /// can track the true minimum read frontier across live consumers.
    fn drop(&mut self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.consumer_positions.remove(&self.consumer_id);
        }
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

    /// Consume the inner reader and produce an Arrow C Stream PyCapsule.
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
                PyArrowError::PyErr(PyIOError::new_err("Reader already consumed"))
            })?;
        Self::to_stream_pycapsule(py, reader, requested_schema)
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let inner = self.0.lock().unwrap();
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyArrowError::PyErr(PyIOError::new_err("Reader already consumed")))?;
        to_schema_pycapsule(py, reader.schema.as_ref())
    }

    #[gen_stub(override_return_type(type_repr = "arro3.core.Schema", imports = ("arro3.core",)))]
    #[getter]
    fn schema(&self) -> PyResult<Arro3Schema> {
        let inner = self.0.lock().unwrap();
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyIOError::new_err("Reader already consumed"))?;
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
            .ok_or_else(|| PyIOError::new_err("Reader already consumed"))?;
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
                return Err(PyArrowError::PyErr(PyIOError::new_err(
                    "Reader already consumed",
                )));
            }
            Some(r) => r,
        };
        // Release the GIL while doing Rust I/O so other Python threads (e.g.
        // DuckDB scanner threads) are not blocked.  The upstream Python reader
        // re-acquires the GIL internally via Python::with_gil when it needs it.
        let result = without_gil(py, || impl_.next());
        match result {
            None => Ok(None),
            Some(Err(e)) => Err(PyArrowError::ArrowError(e)),
            Some(Ok(batch)) => Ok(Some(Arro3RecordBatch::from(batch))),
        }
    }
}

// ── PyCastingStreamCache ──────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "CastingStreamCache", frozen)]
pub struct PyCastingStreamCache {
    inner: Arc<Mutex<DatasetInner>>,
    runtime: Arc<Runtime>,
    source_schema: SchemaRef,
    target_schema: SchemaRef,
}

impl PyCastingStreamCache {
    fn make_reader_impl(&self, py: Python<'_>) -> PyResult<StreamCacheReaderImpl> {
        without_gil(py, || {
            let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
            let consumer_id = inner.next_consumer_id;
            inner.next_consumer_id += 1;
            inner.consumer_positions.insert(consumer_id, 0);
            Ok::<_, String>(StreamCacheReaderImpl {
                schema: self.source_schema.clone(),
                inner: self.inner.clone(),
                current_index: 0,
                consumer_id,
                runtime: self.runtime.clone(),
            })
        })
        .map_err(PyIOError::new_err)
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
            runtime: self.runtime.clone(),
            source_schema: self.source_schema.clone(),
            target_schema: target_schema.into_inner(),
        })
    }
}

#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "StreamCache", frozen)]
pub struct PyStreamCache {
    /// Arrow schema stored outside the cache for O(1) access.
    schema: SchemaRef,
    /// Shared ingestion/cache state accessible by all readers.
    inner: Arc<Mutex<DatasetInner>>,
    /// Dedicated Tokio runtime for driving Foyer's async interface.
    runtime: Arc<Runtime>,
}

impl Drop for PyStreamCache {
    fn drop(&mut self) {
        if let Ok(inner) = self.inner.lock()
            && !inner.closed
        {
            // Do NOT set closed = true here.  Readers hold their own
            // Arc<Mutex<DatasetInner>> and can outlive the dataset handle;
            // marking closed would break them.  Readers that hit the disk
            // tier after the files are gone will receive an "evicted" error,
            // which is acceptable for the implicit-drop path.  Only explicit
            // close() sets closed = true to aggressively terminate readers.
            let foyer_path = inner.foyer_path.clone();
            let cache = inner.cache.clone();
            drop(inner);

            // Remove only the Foyer subdirectory we created; the caller's
            // disk_path directory is left untouched.  Nothing to do for
            // memory-only caches (foyer_path is None).
            if let Some(path) = foyer_path {
                let _ = std::fs::remove_dir_all(path);
            }
            // Best-effort async flush/clear.  May not complete if this is
            // the last Arc<Runtime> reference, but the subdirectory is
            // already gone so data integrity is not at risk.
            // block_on guarantees the clear completes before Drop returns,
            // unlike spawn which may not execute if the runtime is torn down.
            self.runtime.block_on(async move {
                cache.clear().await;
            });
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
        let memory_capacity = memory_capacity.unwrap_or_else(total_system_memory);

        // Build a dedicated multi-thread Tokio runtime.  A fresh runtime is
        // used rather than `Handle::current()` so this works both inside and
        // outside an existing async context.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyIOError::new_err(format!("Failed to create Tokio runtime: {e}")))?;

        let (cache, foyer_path) = match (disk_path, disk_capacity) {
            (Some(path), Some(capacity)) => {
                // Create a unique subdirectory for Foyer's device files so that only
                // that subdirectory is deleted on close/drop, leaving the caller's
                // `disk_path` directory intact.
                let id = DATASET_COUNTER.fetch_add(1, Ordering::Relaxed);
                let foyer_path = PathBuf::from(&path).join(format!("_{id}"));
                std::fs::create_dir_all(&foyer_path).map_err(|e| {
                    PyIOError::new_err(format!("Failed to create Foyer device directory: {e}"))
                })?;

                let disk_cap_usize = usize::try_from(capacity).unwrap_or(usize::MAX);

                // Release the GIL while building the cache: Foyer opens disk
                // files and initialises its async engine, which is pure Rust I/O.
                let cache = without_gil(py, || {
                    rt.block_on(async {
                        let device = FsDeviceBuilder::new(&foyer_path)
                            .with_capacity(disk_cap_usize)
                            .build()?;
                        HybridCacheBuilder::new()
                            // Only write a batch to disk when it is evicted from the
                            // memory tier (i.e. when memory is full).  The alternative,
                            // WriteOnInsertion, would write every batch to disk
                            // immediately regardless of memory pressure.
                            .with_policy(HybridCachePolicy::WriteOnEviction)
                            // Do not flush in-memory entries to disk when the cache is
                            // dropped.  Without this, every StreamCache that fits
                            // entirely in memory would still produce disk writes on drop.
                            .with_flush_on_close(false)
                            .memory(memory_capacity)
                            .storage()
                            .with_engine_config(BlockEngineConfig::new(device))
                            .build()
                            .await
                            .map(CacheTier::Hybrid)
                    })
                })
                .map_err(|e| {
                    PyIOError::new_err(format!("Failed to build Foyer hybrid cache: {e}"))
                })?;

                (cache, Some(foyer_path))
            }
            (None, None) => {
                let cache = CacheBuilder::new(memory_capacity).build();
                (CacheTier::Memory(cache), None)
            }
            _ => {
                return Err(PyIOError::new_err(
                    "disk_path and disk_capacity must both be provided, or both omitted",
                ));
            }
        };

        let inner = DatasetInner {
            cache: Arc::new(cache),
            upstream: Some(upstream),
            ingested_count: 0,
            upstream_exhausted: false,
            consumer_positions: HashMap::new(),
            next_consumer_id: 0,
            foyer_path,
            closed: false,
        };

        Ok(Self {
            schema,
            inner: Arc::new(Mutex::new(inner)),
            runtime: Arc::new(rt),
        })
    }

    #[gen_stub(override_return_type(type_repr = "arro3.core.Schema", imports = ("arro3.core",)))]
    #[getter]
    pub fn schema(&self) -> PyResult<Arro3Schema> {
        Ok(PySchema::new(self.schema.clone()).into())
    }

    #[pyo3(signature = (from_start = true))]
    pub fn reader(&self, py: Python<'_>, from_start: bool) -> PyResult<PyStreamCacheReader> {
        // Release the GIL before locking `inner` to prevent ABBA deadlock:
        // scanner threads can hold `inner.lock()` while waiting to acquire the
        // GIL (via `with_gil_acquired` in `ingest_up_to`), so we must never
        // hold the GIL while waiting for `inner.lock()`.
        without_gil(py, || {
            let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
            if inner.closed {
                return Err("Dataset has been closed".to_string());
            }
            let consumer_id = inner.next_consumer_id;
            inner.next_consumer_id += 1;
            let start_index = if from_start { 0 } else { inner.ingested_count };
            inner.consumer_positions.insert(consumer_id, start_index);
            Ok::<_, String>(PyStreamCacheReader::new(StreamCacheReaderImpl {
                schema: self.schema.clone(),
                inner: self.inner.clone(),
                current_index: start_index,
                consumer_id,
                runtime: self.runtime.clone(),
            }))
        })
        .map_err(PyIOError::new_err)
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
            runtime: self.runtime.clone(),
            source_schema: self.schema.clone(),
            target_schema: target_schema.into_inner(),
        })
    }

    pub fn ingest_all(&self, py: Python<'_>) -> PyResult<u64> {
        // Release the GIL before locking `inner` (same ordering rule as `reader`).
        without_gil(py, || {
            let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
            // u64::MAX as sentinel: the loop inside ingest_up_to exits via
            // upstream_exhausted long before ingested_count could approach it.
            inner.ingest_up_to(u64::MAX).map_err(|e| e.to_string())?;
            Ok::<_, String>(inner.ingested_count)
        })
        .map_err(PyIOError::new_err)
    }

    #[getter]
    pub fn ingested_count(&self, py: Python<'_>) -> PyResult<u64> {
        without_gil(py, || {
            self.inner
                .lock()
                .map(|g| g.ingested_count)
                .map_err(|e| e.to_string())
        })
        .map_err(PyIOError::new_err)
    }

    #[getter]
    pub fn upstream_exhausted(&self, py: Python<'_>) -> PyResult<bool> {
        without_gil(py, || {
            self.inner
                .lock()
                .map(|g| g.upstream_exhausted)
                .map_err(|e| e.to_string())
        })
        .map_err(PyIOError::new_err)
    }

    pub fn close(&self, py: Python<'_>) -> PyResult<()> {
        // Release the GIL before locking `inner` and destroying storage.
        without_gil(py, || {
            let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
            if inner.closed {
                return Ok(()); // idempotent — safe to call more than once
            }
            inner.closed = true;
            let cache = inner.cache.clone();
            let foyer_path = inner.foyer_path.clone();
            drop(inner); // release the lock before I/O

            // Flush/clear the cache (best-effort; errors ignored).
            self.runtime.block_on(async move {
                cache.clear().await;
            });
            // Remove only the Foyer subdirectory we created; errors are
            // ignored because the directory may already be gone.
            // Nothing to do for memory-only caches (foyer_path is None).
            if let Some(path) = foyer_path {
                let _ = std::fs::remove_dir_all(path);
            }
            Ok::<_, String>(())
        })
        .map_err(PyIOError::new_err)
    }
}

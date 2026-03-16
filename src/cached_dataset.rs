//! Hybrid memory+disk cached Arrow dataset backed by Foyer.
//!
//! # Overview
//!
//! [`PyCachedDataset`] accepts any upstream Arrow stream source (anything that
//! exposes `__arrow_c_stream__` in Python) and stores each `RecordBatch` in a
//! Foyer hybrid cache keyed by a monotonic `u64` batch index.  The IPC stream
//! format is used for on-cache serialization so the data is schema-agnostic.
//!
//! Multiple independent [`PyCachedDatasetReader`] handles can be obtained from
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
use std::sync::{Arc, Mutex};

use arrow_array::{ArrayRef, RecordBatch, StructArray};
use arrow_schema::{ArrowError, Field, SchemaRef};
use foyer::{BlockEngineConfig, DeviceBuilder, FsDeviceBuilder, HybridCache, HybridCacheBuilder};
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_arrow::error::{PyArrowError, PyArrowResult};
use pyo3_arrow::export::{Arro3RecordBatch, Arro3Schema};
use pyo3_arrow::ffi::{to_schema_pycapsule, to_stream_pycapsule, ArrayIterator};
use pyo3_arrow::{PyRecordBatchReader, PySchema};
use pyo3_stub_gen::derive::*;
use tokio::runtime::Runtime;

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

/// Mutable state shared between a [`PyCachedDataset`] and all of its readers.
///
/// Protected by `Arc<Mutex<DatasetInner>>` so readers and the dataset can
/// share it across threads.
struct DatasetInner {
    /// Foyer hybrid cache: `batch_index → IPC bytes`.
    cache: Arc<HybridCache<u64, Vec<u8>>>,
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
}

impl DatasetInner {
    /// Pull batches from the upstream reader into the cache until `target_index`
    /// is available (inclusive).
    ///
    /// Returns `true` if `target_index` is now in the cache, or `false` if the
    /// upstream was exhausted before reaching it.
    fn ingest_up_to(&mut self, runtime: &Runtime, target_index: u64) -> Result<bool, ArrowError> {
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
            let cache = self.cache.clone();
            // `HybridCache::insert` is synchronous (memory tier) and schedules
            // disk persistence in the background.
            runtime.block_on(async move {
                cache.insert(idx, bytes);
            });
            self.ingested_count += 1;
        }
        Ok(true)
    }
}

// ── CachedDatasetReaderImpl ──────────────────────────────────────────────────

/// The concrete Rust iterator that backs [`PyCachedDatasetReader`].
///
/// Implements [`arrow_array::RecordBatchReader`] so it can be handed to any
/// Rust API or exported via the Arrow C Stream interface.
struct CachedDatasetReaderImpl {
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

impl Iterator for CachedDatasetReaderImpl {
    type Item = Result<RecordBatch, ArrowError>;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current_index;

        // ── Step 1: ensure the batch has been ingested ──────────────────────
        let cache = {
            let mut inner = match self.inner.lock() {
                Ok(g) => g,
                Err(e) => return Some(Err(other_arrow_err(format!("Mutex poisoned: {e}")))),
            };
            match inner.ingest_up_to(&self.runtime, idx) {
                Err(e) => return Some(Err(e)),
                Ok(false) => return None, // upstream exhausted before idx
                Ok(true) => {}
            }
            // Update this reader's position in the registry.
            inner.consumer_positions.insert(self.consumer_id, idx);
            inner.cache.clone()
        };

        // ── Step 2: fetch IPC bytes from cache (lock released) ───────────────
        let maybe_bytes = self.runtime.block_on(async {
            cache
                .get(&idx)
                .await
                .map(|opt| opt.map(|entry| entry.value().clone()))
        });

        let bytes = match maybe_bytes {
            Err(e) => return Some(Err(other_arrow_err(format!("Cache get failed: {e}")))),
            Ok(None) => {
                return Some(Err(other_arrow_err(format!(
                    "Batch {idx} was evicted from the cache before it could be read. \
                     Increase cache capacity so it can hold all live consumer positions."
                ))))
            }
            Ok(Some(b)) => b,
        };

        // ── Step 3: deserialize and advance position ─────────────────────────
        self.current_index += 1;
        Some(deserialize_batch(&bytes))
    }
}

impl arrow_array::RecordBatchReader for CachedDatasetReaderImpl {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
}

impl Drop for CachedDatasetReaderImpl {
    /// Deregister this reader from the consumer registry on drop so the dataset
    /// can track the true minimum read frontier across live consumers.
    fn drop(&mut self) {
        if let Ok(mut inner) = self.inner.lock() {
            inner.consumer_positions.remove(&self.consumer_id);
        }
    }
}

// ── PyCachedDatasetReader ────────────────────────────────────────────────────

/// A single-use iterator handle for a :class:`CachedDataset`.
///
/// Maintains an independent read position.  Multiple handles backed by the
/// same dataset share the underlying cache; the upstream source is ingested
/// lazily as needed.
///
/// Once consumed via ``__arrow_c_stream__`` or by exhausting iteration the
/// reader is marked closed and raises an error on further use.
///
/// Notes
/// -----
/// Obtain a handle from :meth:`CachedDataset.reader` rather than constructing
/// one directly.
#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "CachedDatasetReader", frozen)]
pub struct PyCachedDatasetReader(Mutex<Option<CachedDatasetReaderImpl>>);

impl PyCachedDatasetReader {
    fn new(impl_: CachedDatasetReaderImpl) -> Self {
        Self(Mutex::new(Some(impl_)))
    }

    /// Consume the inner reader and produce an Arrow C Stream PyCapsule.
    fn to_stream_pycapsule<'py>(
        py: Python<'py>,
        reader: CachedDatasetReaderImpl,
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
impl PyCachedDatasetReader {
    /// Export this reader as an Arrow C Stream PyCapsule.
    ///
    /// Consumes the reader; subsequent calls raise an error.
    ///
    /// Parameters
    /// ----------
    /// requested_schema : object, optional
    ///     Schema capsule to cast the stream to, or ``None``.
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the reader has already been consumed.
    #[gen_stub(skip)]
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let reader =
            self.0.lock().unwrap().take().ok_or_else(|| {
                PyArrowError::PyErr(PyIOError::new_err("Reader already consumed"))
            })?;
        Self::to_stream_pycapsule(py, reader, requested_schema)
    }

    /// Export the schema as an Arrow C Schema PyCapsule.
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the reader has already been consumed.
    #[gen_stub(skip)]
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let inner = self.0.lock().unwrap();
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyArrowError::PyErr(PyIOError::new_err("Reader already consumed")))?;
        to_schema_pycapsule(py, reader.schema.as_ref())
    }

    /// Arrow schema of batches produced by this reader.
    ///
    /// Returns
    /// -------
    /// arro3.core.Schema
    ///
    /// Raises
    /// ------
    /// IOError
    ///     If the reader has already been consumed.
    #[gen_stub(skip)]
    #[getter]
    fn schema(&self) -> PyResult<Arro3Schema> {
        let inner = self.0.lock().unwrap();
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyIOError::new_err("Reader already consumed"))?;
        Ok(PySchema::new(reader.schema.clone()).into())
    }

    /// ``True`` if this reader has been consumed.
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    fn closed(&self) -> bool {
        self.0.lock().unwrap().is_none()
    }

    fn __iter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    #[gen_stub(skip)]
    fn __next__(&self, py: Python<'_>) -> PyArrowResult<Option<Arro3RecordBatch>> {
        let mut guard = self.0.lock().unwrap();
        let impl_ = match guard.as_mut() {
            None => {
                return Err(PyArrowError::PyErr(PyIOError::new_err(
                    "Reader already consumed",
                )))
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

// ── PyCachedDataset ──────────────────────────────────────────────────────────

/// A hybrid memory+disk cached Arrow dataset.
///
/// Wraps any Arrow stream source and caches each ``RecordBatch`` in a Foyer
/// hybrid cache keyed by a monotonic batch index.  Multiple independent
/// :class:`CachedDatasetReader` handles can replay the full stream from any
/// position; the upstream source is ingested lazily on demand.
///
/// Parameters
/// ----------
/// reader : object
///     Any object implementing ``__arrow_c_stream__`` (e.g.
///     :class:`pyarrow.Table`, :class:`pyarrow.RecordBatchReader`,
///     :class:`arro3.core.RecordBatchReader`).
/// memory_capacity : int
///     In-memory cache tier size in bytes.
/// disk_path : str
///     Directory for the on-disk cache tier.  Created on first use.
/// disk_capacity : int
///     On-disk cache tier size in bytes.
///
/// Notes
/// -----
/// Foyer may evict cache entries under memory/disk pressure.  If a batch is
/// evicted before a reader reaches it the reader raises an error.  Size both
/// tiers so they can hold all live reader positions simultaneously.
///
/// Examples
/// --------
/// >>> import tempfile
/// >>> import pyarrow as pa
/// >>> from batchcorder import CachedDataset
/// >>> table = pa.table({"id": [1, 2, 3], "val": [0.5, 1.0, 1.5]})
/// >>> tmp = tempfile.mkdtemp()
/// >>> ds = CachedDataset(table, memory_capacity=16 << 20, disk_path=tmp, disk_capacity=64 << 20)
/// >>> pa.RecordBatchReader.from_stream(ds).read_all().equals(table)
/// True
/// >>> ds.upstream_exhausted
/// True
#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "CachedDataset", frozen)]
pub struct PyCachedDataset {
    /// Arrow schema stored outside the cache for O(1) access.
    schema: SchemaRef,
    /// Shared ingestion/cache state accessible by all readers.
    inner: Arc<Mutex<DatasetInner>>,
    /// Dedicated Tokio runtime for driving Foyer's async interface.
    runtime: Arc<Runtime>,
}

#[gen_stub_pymethods]
#[pymethods]
impl PyCachedDataset {
    /// Create a new :class:`CachedDataset`.
    ///
    /// Foyer opens (or creates) the disk cache under ``disk_path`` during
    /// construction.  The call blocks until the cache is ready.
    ///
    /// Parameters
    /// ----------
    /// reader : object
    ///     Any object implementing ``__arrow_c_stream__``.
    /// memory_capacity : int
    ///     In-memory cache tier size in bytes.
    /// disk_path : str
    ///     Directory for the on-disk cache tier.
    /// disk_capacity : int
    ///     On-disk cache tier size in bytes.
    #[gen_stub(skip)]
    #[new]
    #[pyo3(signature = (reader, memory_capacity, disk_path, disk_capacity))]
    pub fn new(
        py: Python<'_>,
        reader: PyRecordBatchReader,
        memory_capacity: usize,
        disk_path: String,
        disk_capacity: u64,
    ) -> PyResult<Self> {
        let upstream = reader.into_reader()?;
        let schema = upstream.schema();

        // Build a dedicated multi-thread Tokio runtime.  A fresh runtime is
        // used rather than `Handle::current()` so this works both inside and
        // outside an existing async context.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .map_err(|e| PyIOError::new_err(format!("Failed to create Tokio runtime: {e}")))?;

        let disk_cap_usize = usize::try_from(disk_capacity).unwrap_or(usize::MAX);

        // Release the GIL while building the cache: Foyer opens disk files
        // and initialises its async engine, which is pure Rust I/O.
        let cache: HybridCache<u64, Vec<u8>> = without_gil(py, || {
            rt.block_on(async {
                let device = FsDeviceBuilder::new(std::path::Path::new(&disk_path))
                    .with_capacity(disk_cap_usize)
                    .build()?;
                HybridCacheBuilder::new()
                    .memory(memory_capacity)
                    .storage()
                    .with_engine_config(BlockEngineConfig::new(device))
                    .build()
                    .await
            })
        })
        .map_err(|e| PyIOError::new_err(format!("Failed to build Foyer hybrid cache: {e}")))?;

        let inner = DatasetInner {
            cache: Arc::new(cache),
            upstream: Some(upstream),
            ingested_count: 0,
            upstream_exhausted: false,
            consumer_positions: HashMap::new(),
            next_consumer_id: 0,
        };

        Ok(Self {
            schema,
            inner: Arc::new(Mutex::new(inner)),
            runtime: Arc::new(rt),
        })
    }

    /// Arrow schema of this dataset.
    ///
    /// Returns
    /// -------
    /// arro3.core.Schema
    #[gen_stub(skip)]
    #[getter]
    pub fn schema(&self) -> PyResult<Arro3Schema> {
        Ok(PySchema::new(self.schema.clone()).into())
    }

    /// Return a new :class:`CachedDatasetReader` handle.
    ///
    /// Parameters
    /// ----------
    /// from_start : bool, optional
    ///     If ``True`` (default), the reader starts at batch 0 and replays the
    ///     full stream.  If ``False``, it starts at the current ingestion
    ///     frontier and yields only batches ingested after this call.
    ///
    /// Returns
    /// -------
    /// CachedDatasetReader
    ///
    /// Examples
    /// --------
    /// >>> import tempfile, pyarrow as pa
    /// >>> from batchcorder import CachedDataset
    /// >>> table = pa.table({"x": [1, 2, 3]})
    /// >>> tmp = tempfile.mkdtemp()
    /// >>> ds = CachedDataset(table, 16 << 20, tmp, 64 << 20)
    /// >>> r1 = ds.reader()
    /// >>> r2 = ds.reader()
    /// >>> r1.closed, r2.closed
    /// (False, False)
    #[pyo3(signature = (from_start = true))]
    pub fn reader(&self, py: Python<'_>, from_start: bool) -> PyResult<PyCachedDatasetReader> {
        // Release the GIL before locking `inner` to prevent ABBA deadlock:
        // scanner threads can hold `inner.lock()` while waiting to acquire the
        // GIL (via `with_gil_acquired` in `ingest_up_to`), so we must never
        // hold the GIL while waiting for `inner.lock()`.
        without_gil(py, || {
            let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
            let consumer_id = inner.next_consumer_id;
            inner.next_consumer_id += 1;
            let start_index = if from_start { 0 } else { inner.ingested_count };
            inner.consumer_positions.insert(consumer_id, start_index);
            Ok::<_, String>(PyCachedDatasetReader::new(CachedDatasetReaderImpl {
                schema: self.schema.clone(),
                inner: self.inner.clone(),
                current_index: start_index,
                consumer_id,
                runtime: self.runtime.clone(),
            }))
        })
        .map_err(PyIOError::new_err)
    }

    /// Iterate over all batches from the start.
    ///
    /// Creates a fresh :class:`CachedDatasetReader` starting at batch 0 and
    /// returns it as the iterator.
    ///
    /// Returns
    /// -------
    /// CachedDatasetReader
    pub fn __iter__(&self, py: Python<'_>) -> PyResult<PyCachedDatasetReader> {
        self.reader(py, true)
    }

    /// Export this dataset as an Arrow C Stream PyCapsule.
    ///
    /// Creates a fresh reader starting at batch 0.  Allows the dataset to be
    /// consumed directly by PyArrow, DuckDB, DataFusion, and any other
    /// Arrow-compatible library.
    ///
    /// Parameters
    /// ----------
    /// requested_schema : object, optional
    ///     Schema capsule to cast the stream to, or ``None``.
    #[gen_stub(skip)]
    #[pyo3(signature = (requested_schema = None))]
    pub fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let reader = self.reader(py, true).map_err(PyArrowError::PyErr)?;
        let impl_ = reader
            .0
            .lock()
            .unwrap()
            .take()
            .expect("freshly created reader cannot be closed");
        PyCachedDatasetReader::to_stream_pycapsule(py, impl_, requested_schema)
    }

    /// Export the schema as an Arrow C Schema PyCapsule.
    ///
    /// Returns
    /// -------
    /// object
    ///     Arrow C Schema capsule.
    #[gen_stub(skip)]
    pub fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyArrowResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, self.schema.as_ref())
    }

    /// Eagerly ingest all batches from the upstream source into the cache.
    ///
    /// After this call ``upstream_exhausted`` is ``True`` and the upstream
    /// reference is released.  Subsequent reads are served entirely from cache.
    /// Calling this method more than once is safe and idempotent.
    ///
    /// Returns
    /// -------
    /// int
    ///     Total number of batches ingested (including any ingested previously).
    ///
    /// Examples
    /// --------
    /// >>> import tempfile, pyarrow as pa
    /// >>> from batchcorder import CachedDataset
    /// >>> table = pa.table({"x": [1, 2, 3]})
    /// >>> tmp = tempfile.mkdtemp()
    /// >>> ds = CachedDataset(table, 16 << 20, tmp, 64 << 20)
    /// >>> ds.ingest_all()
    /// 1
    /// >>> ds.upstream_exhausted
    /// True
    pub fn ingest_all(&self, py: Python<'_>) -> PyResult<u64> {
        // Release the GIL before locking `inner` (same ordering rule as `reader`).
        without_gil(py, || {
            let mut inner = self.inner.lock().map_err(|e| e.to_string())?;
            inner
                .ingest_up_to(&self.runtime, u64::MAX)
                .map_err(|e| e.to_string())?;
            Ok::<_, String>(inner.ingested_count)
        })
        .map_err(PyIOError::new_err)
    }

    /// Number of batches pulled from the upstream source so far.
    ///
    /// Increments lazily as readers consume batches.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    pub fn ingested_count(&self, py: Python<'_>) -> PyResult<u64> {
        Ok(without_gil(py, || {
            self.inner.lock().unwrap().ingested_count
        }))
    }

    /// ``True`` once the upstream source has been fully consumed.
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    pub fn upstream_exhausted(&self, py: Python<'_>) -> PyResult<bool> {
        Ok(without_gil(py, || {
            self.inner.lock().unwrap().upstream_exhausted
        }))
    }
}

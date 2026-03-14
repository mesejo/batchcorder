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
use pyo3_arrow::ffi::{to_schema_pycapsule, to_stream_pycapsule, ArrayIterator};
use pyo3_arrow::export::Arro3Schema;
use pyo3_arrow::{PyRecordBatchReader, PySchema};
use tokio::runtime::Runtime;

// ── error helpers ────────────────────────────────────────────────────────────

/// Wrap an arbitrary `Display` message as an [`ArrowError`].
fn other_arrow_err(msg: impl std::fmt::Display) -> ArrowError {
    ArrowError::ExternalError(Box::new(std::io::Error::new(
        std::io::ErrorKind::Other,
        msg.to_string(),
    )))
}

// ── IPC serialization ────────────────────────────────────────────────────────

/// Serialize a [`RecordBatch`] to Arrow IPC stream-format bytes.
///
/// The schema is embedded in the IPC stream so each cached blob is
/// self-contained and can be deserialized without external metadata.
fn serialize_batch(batch: &RecordBatch) -> Result<Vec<u8>, ArrowError> {
    let mut buf = Vec::new();
    {
        let mut writer =
            arrow_ipc::writer::StreamWriter::try_new(&mut buf, batch.schema_ref())?;
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
            let batch = match self.upstream.as_mut().and_then(|u| u.next()) {
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
                Err(e) => {
                    return Some(Err(other_arrow_err(format!("Mutex poisoned: {e}"))))
                }
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

/// A single-use reader handle for a [`PyCachedDataset`].
///
/// Maintains an independent read position.  Multiple readers backed by the
/// same dataset share the underlying cache; the upstream is ingested lazily as
/// needed.
///
/// The inner reader is wrapped in `Mutex<Option<…>>` following the arro3
/// pattern: it can be consumed once (by `__arrow_c_stream__`) and is marked
/// closed afterwards.
#[pyclass(module = "multirecord", name = "CachedDatasetReader", frozen)]
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

#[pymethods]
impl PyCachedDatasetReader {
    /// Export this reader as an Arrow C Stream PyCapsule.
    ///
    /// Consumes the reader; subsequent calls will return an error.
    #[pyo3(signature = (requested_schema = None))]
    fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let reader = self
            .0
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| PyArrowError::PyErr(PyIOError::new_err("Reader already consumed")))?;
        Self::to_stream_pycapsule(py, reader, requested_schema)
    }

    /// Export the schema as an Arrow C Schema PyCapsule.
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let inner = self.0.lock().unwrap();
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyArrowError::PyErr(PyIOError::new_err("Reader already consumed")))?;
        to_schema_pycapsule(py, reader.schema.as_ref())
    }

    /// The Arrow schema of this reader.
    #[getter]
    fn schema(&self) -> PyResult<Arro3Schema> {
        let inner = self.0.lock().unwrap();
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyIOError::new_err("Reader already consumed"))?;
        Ok(PySchema::new(reader.schema.clone()).into())
    }

    /// `True` if the underlying stream has been consumed.
    #[getter]
    fn closed(&self) -> bool {
        self.0.lock().unwrap().is_none()
    }
}

// ── PyCachedDataset ──────────────────────────────────────────────────────────

/// A hybrid memory+disk cached Arrow dataset.
///
/// Wraps an upstream Arrow stream source (any Python object implementing
/// `__arrow_c_stream__`) and stores each `RecordBatch` in a Foyer hybrid
/// cache keyed by a monotonic `u64` batch index.  Batches are ingested
/// lazily: the upstream is only read when a reader requests a batch that has
/// not been ingested yet.
///
/// Multiple independent [`PyCachedDatasetReader`] handles can be obtained via
/// [`PyCachedDataset::reader`], each starting at a configurable position.
///
/// # Python interface
///
/// ```python
/// import pyarrow as pa
/// from multirecord import CachedDataset
///
/// table = pa.table({"x": [1, 2, 3]})
/// ds = CachedDataset(table, memory_capacity=64*1024*1024,
///                   disk_path="/tmp/cache", disk_capacity=512*1024*1024)
///
/// # Use directly as an Arrow stream:
/// result = pa.RecordBatchReader.from_stream(ds).read_all()
///
/// # Or get independent replay handles:
/// r1 = ds.reader()
/// r2 = ds.reader()
/// ```
#[pyclass(module = "multirecord", name = "CachedDataset", frozen)]
pub struct PyCachedDataset {
    /// Arrow schema stored outside the cache for O(1) access.
    schema: SchemaRef,
    /// Shared ingestion/cache state accessible by all readers.
    inner: Arc<Mutex<DatasetInner>>,
    /// Dedicated Tokio runtime for driving Foyer's async interface.
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PyCachedDataset {
    /// Create a new `CachedDataset`.
    ///
    /// # Arguments
    ///
    /// * `reader` – any Python object implementing `__arrow_c_stream__`
    ///   (e.g. a PyArrow table/reader, an arro3 RecordBatchReader, etc.)
    /// * `memory_capacity` – in-memory cache tier size in bytes
    /// * `disk_path` – directory in which Foyer stores the disk cache files
    /// * `disk_capacity` – disk cache tier size in bytes
    ///
    /// Foyer opens (or creates) the disk cache under `disk_path` during
    /// construction.  The call blocks until the cache is ready.
    #[new]
    #[pyo3(signature = (reader, memory_capacity, disk_path, disk_capacity))]
    pub fn new(
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

        let cache: HybridCache<u64, Vec<u8>> = rt
            .block_on(async {
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
            .map_err(|e| {
                PyIOError::new_err(format!("Failed to build Foyer hybrid cache: {e}"))
            })?;

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

    /// Return the Arrow schema of this dataset.
    #[getter]
    pub fn schema(&self) -> PyResult<Arro3Schema> {
        Ok(PySchema::new(self.schema.clone()).into())
    }

    /// Create a new [`PyCachedDatasetReader`] handle.
    ///
    /// # Arguments
    ///
    /// * `from_start` – if `True` (default), the reader starts at batch index
    ///   0 and replays the full stream.  If `False`, the reader starts at the
    ///   current ingestion frontier (the number of batches ingested so far).
    #[pyo3(signature = (from_start = true))]
    pub fn reader(&self, from_start: bool) -> PyResult<PyCachedDatasetReader> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| PyIOError::new_err(format!("Mutex poisoned: {e}")))?;

        let consumer_id = inner.next_consumer_id;
        inner.next_consumer_id += 1;

        let start_index = if from_start {
            0
        } else {
            inner.ingested_count
        };
        inner.consumer_positions.insert(consumer_id, start_index);

        Ok(PyCachedDatasetReader::new(CachedDatasetReaderImpl {
            schema: self.schema.clone(),
            inner: self.inner.clone(),
            current_index: start_index,
            consumer_id,
            runtime: self.runtime.clone(),
        }))
    }

    /// Export this dataset as an Arrow C Stream PyCapsule.
    ///
    /// Creates a fresh reader starting at batch 0 and exports it as a single-
    /// use Arrow C Stream.  This allows the dataset to be consumed directly by
    /// PyArrow, DataFusion, DuckDB, and any other Arrow-compatible library.
    #[pyo3(signature = (requested_schema = None))]
    pub fn __arrow_c_stream__<'py>(
        &self,
        py: Python<'py>,
        requested_schema: Option<Bound<'py, PyCapsule>>,
    ) -> PyArrowResult<Bound<'py, PyCapsule>> {
        let reader = self
            .reader(true)
            .map_err(|e| PyArrowError::PyErr(e))?;
        let impl_ = reader
            .0
            .lock()
            .unwrap()
            .take()
            .expect("freshly created reader cannot be closed");
        PyCachedDatasetReader::to_stream_pycapsule(py, impl_, requested_schema)
    }

    /// Export the schema as an Arrow C Schema PyCapsule.
    pub fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyArrowResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, self.schema.as_ref())
    }

    /// Eagerly ingest all remaining batches from the upstream reader into the cache.
    ///
    /// After this call, `upstream_exhausted` is `true` and the upstream object
    /// is released.  Subsequent readers will be served entirely from the cache.
    pub fn ingest_all(&self) -> PyResult<u64> {
        let mut inner = self
            .inner
            .lock()
            .map_err(|e| PyIOError::new_err(format!("Mutex poisoned: {e}")))?;
        // Drive ingestion to a very high index; ingest_up_to stops naturally
        // when the upstream is exhausted.
        let _ = inner
            .ingest_up_to(&self.runtime, u64::MAX)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(inner.ingested_count)
    }

    /// Number of batches ingested from the upstream so far.
    #[getter]
    pub fn ingested_count(&self) -> PyResult<u64> {
        Ok(self.inner.lock().unwrap().ingested_count)
    }

    /// `True` if the upstream reader has been fully consumed.
    #[getter]
    pub fn upstream_exhausted(&self) -> PyResult<bool> {
        Ok(self.inner.lock().unwrap().upstream_exhausted)
    }
}

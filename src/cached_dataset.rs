//! Python bindings for [`batchcorder_core::StreamCache`].
//!
//! # Overview
//!
//! [`PyStreamCache`] accepts any upstream Arrow stream source (anything that
//! exposes `__arrow_c_stream__` in Python) and caches its batches in memory or
//! on disk; see `batchcorder-core` for the storage and eviction machinery.
//! Multiple independent [`PyStreamCacheReader`] handles can be obtained from a
//! single dataset, each maintaining its own read position.
//!
//! This module only adds what the Python boundary needs: GIL handling around
//! blocking calls, mapping errors to Python exceptions, and Arrow PyCapsule
//! export.

use std::ffi::CString;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_pyarrow::{IntoPyArrow, PyArrowType};
use arrow_schema::{ArrowError, Schema, SchemaRef};
use batchcorder_core::{
    CacheConfig, DiskConfig, StreamCache, StreamCacheReader, Upstream, WritePolicy,
};
use pyo3::exceptions::{PyIOError, PyMemoryError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_stub_gen::derive::*;

// ── error helpers ────────────────────────────────────────────────────────────

fn poisoned_lock_pyerr<T>(_: std::sync::PoisonError<T>) -> PyErr {
    PyRuntimeError::new_err("Internal lock poisoned by a prior panic")
}

/// Map an [`ArrowError`] from a cache-level operation (construction, reader
/// creation, ingestion) to the matching Python exception.
fn arrow_to_boundary(e: ArrowError) -> PyErr {
    match e {
        ArrowError::MemoryError(msg) => PyMemoryError::new_err(msg),
        ArrowError::InvalidArgumentError(msg) => PyValueError::new_err(msg),
        ArrowError::IoError(msg, _) => PyIOError::new_err(msg),
        _ => arrow_pyarrow::PyArrowException::new_err(e.to_string()),
    }
}

/// Map an [`ArrowError`] raised while iterating a reader.  Unlike
/// [`arrow_to_boundary`], I/O errors surface as `ArrowException`.
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

// ── GIL management ───────────────────────────────────────────────────────────
//
// We use raw CPython FFI instead of PyO3's safe wrappers because:
//  - `py.allow_threads(f)` requires `F: Ungil`, which forbids reacquiring the
//    GIL inside `f`.  `PyUpstream::next_batch` needs to reacquire it.
//  - `Python::with_gil()` was removed in PyO3 0.28.
//
// Safety contract: `without_gil` saves the thread state and restores it on
// drop; `with_gil_acquired` ensures/releases via `PyGILState_Ensure/Release`.
// Both are no-ops under the free-threaded build (`Py_GIL_DISABLED`).

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

// ── PyUpstream ───────────────────────────────────────────────────────────────

/// An Arrow C stream imported from Python.  The producer may be implemented
/// in Python (e.g. a pyarrow reader over a generator), so every pull holds
/// the GIL.  Readers call `next_batch` with the GIL released, so this
/// reacquires it rather than assuming it is held.
struct PyUpstream(ArrowArrayStreamReader);

impl Upstream for PyUpstream {
    fn next_batch(&mut self) -> Option<Result<RecordBatch, ArrowError>> {
        with_gil_acquired(|| self.0.next())
    }
}

// ── PyStreamCacheReader ────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "StreamCacheReader", frozen)]
pub struct PyStreamCacheReader(Mutex<Option<StreamCacheReader>>);

impl PyStreamCacheReader {
    fn new(reader: StreamCacheReader) -> Self {
        Self(Mutex::new(Some(reader)))
    }

    fn to_stream_pycapsule<'py>(
        py: Python<'py>,
        reader: StreamCacheReader,
        requested_schema: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let boxed: Box<dyn RecordBatchReader + Send> = Box::new(reader);
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
            .map_err(poisoned_lock_pyerr)?
            .take()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        Self::to_stream_pycapsule(py, reader, requested_schema.map(|c| c.into_any()))
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        let inner = self.0.lock().map_err(poisoned_lock_pyerr)?;
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        to_schema_pycapsule(py, reader.schema().as_ref())
    }

    #[gen_stub(override_return_type(type_repr = "pa.Schema", imports = ("pyarrow as pa",)))]
    #[getter]
    fn schema(&self) -> PyResult<PyArrowType<Schema>> {
        let inner = self.0.lock().map_err(poisoned_lock_pyerr)?;
        let reader = inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        Ok(PyArrowType((*reader.schema()).clone()))
    }

    #[getter]
    fn closed(&self) -> PyResult<bool> {
        Ok(self.0.lock().map_err(poisoned_lock_pyerr)?.is_none())
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
        let reader = self
            .0
            .lock()
            .map_err(poisoned_lock_pyerr)?
            .take()
            .ok_or_else(|| PyValueError::new_err("Reader already consumed"))?;
        let new_reader = PyStreamCacheReader::new(reader);
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
        let mut guard = self.0.lock().map_err(poisoned_lock_pyerr)?;
        let reader = match guard.as_mut() {
            None => {
                return Err(PyValueError::new_err("Reader already consumed"));
            }
            Some(r) => r,
        };
        let result = without_gil(py, || reader.next());
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
    cache: StreamCache,
    target_schema: SchemaRef,
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
        let reader = without_gil(py, || self.cache.reader(true)).map_err(arrow_to_boundary)?;
        let effective_schema: Option<Bound<'py, PyAny>> = if requested_schema.is_some() {
            requested_schema.map(|c| c.into_any())
        } else {
            Some(to_schema_pycapsule(py, self.target_schema.as_ref())?.into_any())
        };
        PyStreamCacheReader::to_stream_pycapsule(py, reader, effective_schema)
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
            cache: self.cache.clone(),
            target_schema: Arc::new(target_schema.0),
        })
    }
}

// ── PyStreamCache ─────────────────────────────────────────────────────────────

#[gen_stub_pyclass]
#[pyclass(module = "batchcorder", name = "StreamCache", frozen)]
pub struct PyStreamCache {
    cache: StreamCache,
}

// No `Drop` impl: readers share the cache's state and may outlive this
// handle.  Cached batches stay available until the last owner (handle or
// reader) drops, at which point the disk tier removes its subdirectory.
// Explicit teardown is still available via `close()`.

#[gen_stub_pymethods]
#[pymethods]
impl PyStreamCache {
    #[new]
    #[pyo3(signature = (reader, memory_capacity = None, disk_path = None, disk_capacity = None, write_policy = "on_insertion".to_string(), max_readers = None))]
    pub fn new(
        py: Python<'_>,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        reader: PyArrowType<ArrowArrayStreamReader>,
        memory_capacity: Option<usize>,
        disk_path: Option<String>,
        disk_capacity: Option<u64>,
        write_policy: String,
        max_readers: Option<u64>,
    ) -> PyResult<Self> {
        // Validated here as well as in the core so argument errors keep their
        // original precedence (max_readers, then write_policy, then disk).
        if max_readers == Some(0) {
            return Err(PyValueError::new_err("max_readers must be at least 1"));
        }
        let schema = reader.0.schema();
        let upstream: Box<dyn Upstream> = Box::new(PyUpstream(reader.0));

        // Parse the write policy up front (applies to the disk tier only; ignored
        // for memory-only caches).
        let write_policy = match write_policy.as_str() {
            "on_insertion" => WritePolicy::OnInsertion,
            "on_eviction" => WritePolicy::OnEviction,
            other => {
                return Err(PyValueError::new_err(format!(
                    "write_policy must be 'on_insertion' or 'on_eviction', got {other:?}"
                )));
            }
        };

        let disk = match (disk_path, disk_capacity) {
            (Some(path), Some(capacity)) => Some(DiskConfig {
                path: PathBuf::from(path),
                capacity,
            }),
            (None, None) => None,
            _ => {
                return Err(PyValueError::new_err(
                    "disk_path and disk_capacity must both be provided, or both omitted",
                ));
            }
        };

        let config = CacheConfig {
            memory_capacity,
            disk,
            write_policy,
            max_readers,
        };
        let cache = if config.disk.is_some() {
            // Creating the disk tier is pure OS I/O — release the GIL.
            without_gil(py, || StreamCache::try_new(upstream, schema, config))
        } else {
            StreamCache::try_new(upstream, schema, config)
        }
        .map_err(arrow_to_boundary)?;

        Ok(Self { cache })
    }

    #[gen_stub(override_return_type(type_repr = "pa.Schema", imports = ("pyarrow as pa",)))]
    #[getter]
    pub fn schema(&self) -> PyResult<PyArrowType<Schema>> {
        Ok(PyArrowType((*self.cache.schema()).clone()))
    }

    #[pyo3(signature = (from_start = true))]
    pub fn reader(&self, py: Python<'_>, from_start: bool) -> PyResult<PyStreamCacheReader> {
        without_gil(py, || self.cache.reader(from_start))
            .map(PyStreamCacheReader::new)
            .map_err(arrow_to_boundary)
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
        let reader = without_gil(py, || self.cache.reader(true)).map_err(arrow_to_boundary)?;
        PyStreamCacheReader::to_stream_pycapsule(py, reader, requested_schema.map(|c| c.into_any()))
    }

    #[gen_stub(override_return_type(type_repr = "typing.Any", imports = ("typing",)))]
    pub fn __arrow_c_schema__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        to_schema_pycapsule(py, self.cache.schema().as_ref())
    }

    pub fn cast(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Any", imports = ("typing",)))]
        target_schema: PyArrowType<Schema>,
    ) -> PyResult<PyCastingStreamCache> {
        Ok(PyCastingStreamCache {
            cache: self.cache.clone(),
            target_schema: Arc::new(target_schema.0),
        })
    }

    pub fn ingest_all(&self, py: Python<'_>) -> PyResult<u64> {
        without_gil(py, || self.cache.ingest_all()).map_err(arrow_to_boundary)
    }

    #[getter]
    pub fn ingested_count(&self, py: Python<'_>) -> PyResult<u64> {
        without_gil(py, || self.cache.ingested_count()).map_err(arrow_to_boundary)
    }

    #[getter]
    pub fn upstream_exhausted(&self, py: Python<'_>) -> PyResult<bool> {
        without_gil(py, || self.cache.upstream_exhausted()).map_err(arrow_to_boundary)
    }

    pub fn close(&self, py: Python<'_>) -> PyResult<()> {
        without_gil(py, || self.cache.close()).map_err(arrow_to_boundary)
    }
}

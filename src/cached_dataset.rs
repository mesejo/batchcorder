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

use std::sync::{Arc, Mutex};

use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::ffi::FFI_ArrowSchema;
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use arrow_pyarrow::{IntoPyArrow, PyArrowType};
use arrow_schema::{Schema, SchemaRef};
use batchcorder_core::{Bounding, StreamCache, StreamCacheReader};
use batchcorder_pyo3::{
    arrow_to_boundary, arrow_to_py_err, build_cache, parse_disk_config, parse_write_policy,
    without_gil,
};
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use pyo3_stub_gen::derive::*;

// ── error helpers ────────────────────────────────────────────────────────────

fn poisoned_lock_pyerr<T>(_: std::sync::PoisonError<T>) -> PyErr {
    PyRuntimeError::new_err("Internal lock poisoned by a prior panic")
}

/// Export `schema` as an `"arrow_schema"` [`PyCapsule`] via Arrow FFI —
/// no pyarrow Python object is allocated.
fn to_schema_pycapsule<'py>(py: Python<'py>, schema: &Schema) -> PyResult<Bound<'py, PyCapsule>> {
    let ffi_schema =
        FFI_ArrowSchema::try_from(schema).map_err(|e| PyValueError::new_err(e.to_string()))?;
    PyCapsule::new_with_value(py, ffi_schema, c"arrow_schema")
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
                PyCapsule::new_with_value(py, ffi_stream, c"arrow_array_stream")
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
        // Parse the write policy up front (applies to the disk tier only; ignored
        // for memory-only caches).
        let write_policy = parse_write_policy(&write_policy)?;
        let disk = parse_disk_config(disk_path, disk_capacity)?;
        let bounding = match max_readers {
            Some(max) => Bounding::Counted { max },
            None => Bounding::Unbounded,
        };
        let cache = build_cache(py, reader.0, memory_capacity, disk, write_policy, bounding)?;
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

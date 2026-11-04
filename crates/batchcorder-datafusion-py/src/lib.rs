//! Python bindings for `batchcorder-datafusion`, exported to
//! datafusion-python over `datafusion-ffi`.
//!
//! [`PyTeeTableProvider`] owns its own [`StreamCache`]: Rust objects cannot be
//! shared with the separate `batchcorder` extension, so the provider takes the
//! upstream Arrow stream directly.  It implements the datafusion-python
//! `__datafusion_table_provider__(session)` protocol, so it can be passed to
//! `SessionContext.register_table`.

use std::sync::{Arc, LazyLock};

use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::ArrowArrayStreamReader;
use arrow_pyarrow::PyArrowType;
use arrow_schema::Schema;
use batchcorder_core::{Bounding, CacheConfig, StreamCache, Upstream};
use batchcorder_datafusion::TeeTableProvider;
use batchcorder_pyo3::{
    PyUpstream, arrow_to_boundary, parse_disk_config, parse_write_policy, wait_without_gil,
    without_gil,
};
use datafusion::catalog::TableProvider;
use datafusion_ffi::proto::logical_extension_codec::FFI_LogicalExtensionCodec;
use datafusion_ffi::table_provider::FFI_TableProvider;
use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use tokio::runtime::{Handle, Runtime};

// ── runtime ──────────────────────────────────────────────────────────────────

/// The Tokio runtime that runs this library's FFI futures and blocking reader
/// tasks.  The host's runtime lives in another copy of Tokio (statically
/// linked into datafusion-python), which this library cannot see.  Readers
/// run on its blocking pool, so a couple of async workers are enough.
static RUNTIME: LazyLock<Result<Runtime, String>> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("batchcorder-datafusion")
        .enable_all()
        .build()
        .map_err(|e| e.to_string())
});

fn runtime_handle() -> PyResult<Handle> {
    match &*RUNTIME {
        Ok(rt) => Ok(rt.handle().clone()),
        Err(e) => Err(PyRuntimeError::new_err(format!(
            "Failed to start the batchcorder-datafusion Tokio runtime: {e}"
        ))),
    }
}

// ── FFI helpers ──────────────────────────────────────────────────────────────

/// Extract the host's logical extension codec from the `session` argument of
/// `__datafusion_table_provider__`.  datafusion-python passes either an
/// object exposing `__datafusion_logical_extension_codec__()` (a session) or
/// the codec capsule itself (`Schema.register_table`), so both are accepted.
fn ffi_logical_codec(session: &Bound<'_, PyAny>) -> PyResult<FFI_LogicalExtensionCodec> {
    let obj = if session.hasattr("__datafusion_logical_extension_codec__")? {
        session.call_method0("__datafusion_logical_extension_codec__")?
    } else {
        session.clone()
    };
    let capsule = obj.cast::<PyCapsule>().map_err(|_| {
        PyTypeError::new_err(
            "expected a datafusion session or a 'datafusion_logical_extension_codec' capsule",
        )
    })?;
    let ptr = capsule.pointer_checked(Some(c"datafusion_logical_extension_codec"))?;
    // SAFETY: the capsule name was checked above, and by the datafusion-python
    // protocol a capsule with that name holds an `FFI_LogicalExtensionCodec`.
    // It is cloned (bumping the codec's own reference count) while the
    // capsule is alive, so the borrow does not outlive its owner.
    let codec = unsafe { ptr.cast::<FFI_LogicalExtensionCodec>().as_ref() };
    Ok(codec.clone())
}

fn parse_bounding(bounding: &str) -> PyResult<Bounding> {
    match bounding {
        "sealed" => Ok(Bounding::Sealed),
        "unbounded" => Ok(Bounding::Unbounded),
        other => Err(PyValueError::new_err(format!(
            "bounding must be 'sealed' or 'unbounded', got {other:?}"
        ))),
    }
}

// ── TeeTableProvider ─────────────────────────────────────────────────────────

/// DataFusion table provider over a read-once Arrow stream.
///
/// Every scan of the table in a query replays the same upstream stream, which
/// is read at most once.
///
/// Parameters
/// ----------
/// reader
///     Any object exposing ``__arrow_c_stream__`` (a ``pa.RecordBatchReader``,
///     a ``pa.Table``, ...).
/// memory_capacity
///     Byte budget of the in-memory cache (or of the hot layer when spilling
///     to disk).  Defaults to a fraction of system RAM.
/// disk_path, disk_capacity
///     Spill batches to an Arrow IPC file under ``disk_path``, up to
///     ``disk_capacity`` bytes.  Give both or neither.
/// write_policy
///     ``"on_insertion"`` or ``"on_eviction"``: when batches reach disk.
/// bounding
///     ``"sealed"`` (default): readers are counted from the query plan and
///     batches are evicted once every scan has read them; one query per
///     provider.  ``"unbounded"``: never evict; the table can be queried
///     repeatedly.
#[pyclass(module = "batchcorder_datafusion", name = "TeeTableProvider", frozen)]
pub struct PyTeeTableProvider {
    provider: Arc<TeeTableProvider>,
}

impl PyTeeTableProvider {
    fn cache(&self) -> &StreamCache {
        self.provider.cache()
    }
}

#[pymethods]
impl PyTeeTableProvider {
    #[new]
    #[pyo3(signature = (reader, *, memory_capacity = None, disk_path = None, disk_capacity = None, write_policy = "on_insertion", bounding = "sealed"))]
    fn new(
        py: Python<'_>,
        reader: PyArrowType<ArrowArrayStreamReader>,
        memory_capacity: Option<usize>,
        disk_path: Option<String>,
        disk_capacity: Option<u64>,
        write_policy: &str,
        bounding: &str,
    ) -> PyResult<Self> {
        let bounding = parse_bounding(bounding)?;
        let write_policy = parse_write_policy(write_policy)?;
        let disk = parse_disk_config(disk_path, disk_capacity)?;
        let schema = reader.0.schema();
        let upstream: Box<dyn Upstream> = Box::new(PyUpstream::new(reader.0));
        let config = CacheConfig {
            memory_capacity,
            disk,
            write_policy,
            bounding,
            wait_hook: Some(wait_without_gil),
        };
        let cache = if config.disk.is_some() {
            // Creating the disk tier is pure OS I/O — release the GIL.
            without_gil(py, || StreamCache::try_new(upstream, schema, config))
        } else {
            StreamCache::try_new(upstream, schema, config)
        }
        .map_err(arrow_to_boundary)?;
        let provider = TeeTableProvider::new(cache).with_runtime(runtime_handle()?);
        Ok(Self {
            provider: Arc::new(provider),
        })
    }

    /// Export this provider to datafusion-python over `datafusion-ffi`.
    fn __datafusion_table_provider__<'py>(
        &self,
        py: Python<'py>,
        session: Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyCapsule>> {
        let codec = ffi_logical_codec(&session)?;
        let provider: Arc<dyn TableProvider> = self.provider.clone();
        // Filter pushdown off: the provider does not use filters yet, and
        // turning it off avoids serialising them over the codec.
        let ffi =
            FFI_TableProvider::new_with_ffi_codec(provider, false, Some(runtime_handle()?), codec);
        PyCapsule::new_with_value(py, ffi, c"datafusion_table_provider")
    }

    #[getter]
    fn schema(&self) -> PyArrowType<Schema> {
        PyArrowType((*self.cache().schema()).clone())
    }

    /// Number of physical scan nodes planned over this table.
    #[getter]
    fn scans_planned(&self) -> u64 {
        self.provider.stats().scans_planned()
    }

    /// Number of scan partitions that started executing.
    #[getter]
    fn scans_executed(&self) -> u64 {
        self.provider.stats().scans_executed()
    }

    #[getter]
    fn ingested_count(&self, py: Python<'_>) -> PyResult<u64> {
        without_gil(py, || self.cache().ingested_count()).map_err(arrow_to_boundary)
    }

    #[getter]
    fn evicted_count(&self, py: Python<'_>) -> PyResult<u64> {
        without_gil(py, || self.cache().evicted_count()).map_err(arrow_to_boundary)
    }

    #[getter]
    fn upstream_exhausted(&self, py: Python<'_>) -> PyResult<bool> {
        without_gil(py, || self.cache().upstream_exhausted()).map_err(arrow_to_boundary)
    }

    #[getter]
    fn is_sealed(&self, py: Python<'_>) -> PyResult<bool> {
        without_gil(py, || self.cache().is_sealed()).map_err(arrow_to_boundary)
    }

    /// Close reader registration so eviction can start (sealed only).
    fn seal(&self, py: Python<'_>) -> PyResult<()> {
        without_gil(py, || self.cache().seal()).map_err(arrow_to_boundary)
    }

    /// Release cached data and delete the disk tier.  Idempotent.
    fn close(&self, py: Python<'_>) -> PyResult<()> {
        without_gil(py, || self.cache().close()).map_err(arrow_to_boundary)
    }
}

#[pymodule]
fn _batchcorder_datafusion(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTeeTableProvider>()?;
    Ok(())
}

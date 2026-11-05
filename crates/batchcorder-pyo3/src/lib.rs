//! PyO3 glue shared by batchcorder's Python extensions (`batchcorder` and
//! `batchcorder-datafusion`): GIL handling around blocking cache calls, a
//! Python-backed [`Upstream`], error mapping, and constructor-argument
//! parsing.
//!
//! # GIL management
//!
//! Raw CPython FFI is used instead of PyO3's safe wrappers because:
//!  - `py.detach(f)` requires `F: Ungil`, which forbids reacquiring the GIL
//!    inside `f`.  [`PyUpstream::next_batch`] needs to reacquire it.
//!  - `Python::with_gil()` was removed in PyO3 0.28.
//!
//! Safety contract: [`without_gil`] saves the thread state and restores it on
//! drop; [`with_gil_acquired`] ensures/releases via
//! `PyGILState_Ensure/Release`.  All helpers are no-ops under the
//! free-threaded build (`Py_GIL_DISABLED`).

use std::path::PathBuf;

use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::ArrowArrayStreamReader;
use arrow_schema::ArrowError;
use batchcorder_core::{
    Bounding, CacheConfig, DiskConfig, StreamCache, Upstream, WaitHook, WritePolicy,
};
use pyo3::exceptions::{PyIOError, PyMemoryError, PyValueError};
use pyo3::prelude::*;

// ── error mapping ────────────────────────────────────────────────────────────

/// Map an [`ArrowError`] from a cache-level operation (construction, reader
/// creation, ingestion) to the matching Python exception.
pub fn arrow_to_boundary(e: ArrowError) -> PyErr {
    match e {
        ArrowError::MemoryError(msg) => PyMemoryError::new_err(msg),
        ArrowError::InvalidArgumentError(msg) => PyValueError::new_err(msg),
        ArrowError::IoError(msg, _) => PyIOError::new_err(msg),
        _ => arrow_pyarrow::PyArrowException::new_err(e.to_string()),
    }
}

/// Map an [`ArrowError`] raised while iterating a reader.  Unlike
/// [`arrow_to_boundary`], I/O errors surface as `ArrowException`.
pub fn arrow_to_py_err(e: ArrowError) -> PyErr {
    match e {
        ArrowError::MemoryError(msg) => PyMemoryError::new_err(msg),
        ArrowError::InvalidArgumentError(msg) => PyValueError::new_err(msg),
        _ => arrow_pyarrow::PyArrowException::new_err(e.to_string()),
    }
}

// ── argument parsing ─────────────────────────────────────────────────────────

/// Parse the `write_policy` constructor argument.
pub fn parse_write_policy(write_policy: &str) -> PyResult<WritePolicy> {
    match write_policy {
        "on_insertion" => Ok(WritePolicy::OnInsertion),
        "on_eviction" => Ok(WritePolicy::OnEviction),
        other => Err(PyValueError::new_err(format!(
            "write_policy must be 'on_insertion' or 'on_eviction', got {other:?}"
        ))),
    }
}

/// Combine the `disk_path` / `disk_capacity` constructor arguments, which
/// must be given together or not at all.
pub fn parse_disk_config(
    disk_path: Option<String>,
    disk_capacity: Option<u64>,
) -> PyResult<Option<DiskConfig>> {
    match (disk_path, disk_capacity) {
        (Some(path), Some(capacity)) => Ok(Some(DiskConfig {
            path: PathBuf::from(path),
            capacity,
        })),
        (None, None) => Ok(None),
        _ => Err(PyValueError::new_err(
            "disk_path and disk_capacity must both be provided, or both omitted",
        )),
    }
}

// ── GIL management ───────────────────────────────────────────────────────────

/// Run `f` with the GIL released.  The caller must hold the GIL (the `py`
/// token proves it).
pub fn without_gil<T, F: FnOnce() -> T>(_py: Python<'_>, f: F) -> T {
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

/// Run `f` holding the GIL, acquiring it if this thread does not hold it.
pub fn with_gil_acquired<T, F: FnOnce() -> T>(f: F) -> T {
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

/// [`batchcorder_core::WaitHook`] that releases the GIL while a reader waits
/// for another reader's ingest.
///
/// The waiting thread may hold the GIL (a host that calls the Arrow C stream
/// without releasing it), and the ingesting thread may need the GIL to pull
/// from a Python upstream; without this the two would deadlock.
///
/// A thread with no Python thread state (a query engine's worker thread)
/// cannot hold the GIL, so it just waits: taking the GIL only to release it
/// would make every such wait queue behind Python threads.  Otherwise the
/// abi3 API cannot ask "do I hold the GIL?", so take it with
/// `PyGILState_Ensure` (a no-op re-entry if already held) and then release
/// it fully with `PyEval_SaveThread`.  The core calls this with no cache lock
/// held, so blocking in `PyGILState_Ensure` here cannot deadlock.
pub fn wait_without_gil(wait: &mut dyn FnMut()) {
    #[cfg(not(Py_GIL_DISABLED))]
    {
        // SAFETY: only reads this thread's GIL-state slot; valid on any
        // thread once the interpreter is initialised, which it is whenever
        // this extension's code runs.
        if unsafe { pyo3::ffi::PyGILState_GetThisThreadState() }.is_null() {
            wait();
            return;
        }
        struct ReleaseGuard(pyo3::ffi::PyGILState_STATE);
        impl Drop for ReleaseGuard {
            fn drop(&mut self) {
                unsafe { pyo3::ffi::PyGILState_Release(self.0) };
            }
        }
        struct RestoreGuard(*mut pyo3::ffi::PyThreadState);
        impl Drop for RestoreGuard {
            fn drop(&mut self) {
                unsafe { pyo3::ffi::PyEval_RestoreThread(self.0) };
            }
        }
        // Declared in this order so the restore runs before the release.
        let _ensured = ReleaseGuard(unsafe { pyo3::ffi::PyGILState_Ensure() });
        let _saved = RestoreGuard(unsafe { pyo3::ffi::PyEval_SaveThread() });
        wait();
    }
    #[cfg(Py_GIL_DISABLED)]
    wait();
}

// ── PyUpstream ───────────────────────────────────────────────────────────────

/// An Arrow C stream imported from Python.  The producer may be implemented
/// in Python (e.g. a pyarrow reader over a generator), so every pull holds
/// the GIL.  Readers call `next_batch` with the GIL released, so this
/// reacquires it rather than assuming it is held.
pub struct PyUpstream(ArrowArrayStreamReader);

impl PyUpstream {
    /// Wrap a stream imported from Python, e.g. the `reader` argument of a
    /// constructor taking `PyArrowType<ArrowArrayStreamReader>`.
    pub fn new(reader: ArrowArrayStreamReader) -> Self {
        Self(reader)
    }
}

impl Upstream for PyUpstream {
    fn next_batch(&mut self) -> Option<Result<RecordBatch, ArrowError>> {
        with_gil_acquired(|| self.0.next())
    }
}

// ── cache construction ───────────────────────────────────────────────────────

/// Build a [`StreamCache`] over a stream imported from Python, the way both
/// extensions' constructors do: a GIL-reacquiring [`PyUpstream`], the
/// GIL-releasing wait hook, and the GIL released while the disk tier is
/// created.  Callers parse and validate their arguments first, so argument
/// errors keep each constructor's own precedence.
pub fn build_cache(
    py: Python<'_>,
    reader: ArrowArrayStreamReader,
    memory_capacity: Option<usize>,
    disk: Option<DiskConfig>,
    write_policy: WritePolicy,
    bounding: Bounding,
) -> PyResult<StreamCache> {
    let schema = reader.schema();
    let upstream: Box<dyn Upstream> = Box::new(PyUpstream::new(reader));
    let wait_hook: WaitHook = wait_without_gil;
    let config = CacheConfig {
        memory_capacity,
        disk,
        write_policy,
        bounding,
        wait_hook: Some(wait_hook),
    };
    if config.disk.is_some() {
        // Creating the disk tier is pure OS I/O: release the GIL.
        without_gil(py, || StreamCache::try_new(upstream, schema, config))
    } else {
        StreamCache::try_new(upstream, schema, config)
    }
    .map_err(arrow_to_boundary)
}

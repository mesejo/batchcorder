use pyo3::prelude::*;
use pyo3_stub_gen::StubInfo;

mod cached_dataset;

// LCOV_EXCL_START — PyO3 module init and stub gatherer are macro-generated
// entry points exercised only through the C FFI boundary; the instrumentation
// cannot reliably attribute coverage to them.
#[pymodule]
fn _batchcorder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<cached_dataset::PyStreamCache>()?;
    m.add_class::<cached_dataset::PyStreamCacheReader>()?;
    m.add_class::<cached_dataset::PyCastingStreamCache>()?;
    Ok(())
}

/// Gather the information needed to generate the stub files.
///
/// Hand-written instead of `define_stub_info_gatherer!`, which reads
/// `pyproject.toml` from `CARGO_MANIFEST_DIR`: this crate is a member of a
/// virtual workspace, and the pyproject lives at the repository root.
pub fn stub_info() -> pyo3_stub_gen::Result<StubInfo> {
    let manifest_dir: &std::path::Path = env!("CARGO_MANIFEST_DIR").as_ref();
    StubInfo::from_pyproject_toml(manifest_dir.join("../../pyproject.toml"))
}
// LCOV_EXCL_STOP

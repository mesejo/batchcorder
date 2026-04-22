use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

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

define_stub_info_gatherer!(stub_info);
// LCOV_EXCL_STOP

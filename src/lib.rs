use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

mod cached_dataset;

/// Register the `batchcorder._batchcorder` Python extension module.
///
/// Exposes:
/// * [`cached_dataset::PyStreamCache`] — cached Arrow dataset (memory-only or hybrid)
/// * [`cached_dataset::PyStreamCacheReader`] — independent reader handle
/// * [`cached_dataset::PyCastingStreamCache`] — replayable cast view of a dataset
#[pymodule]
fn _batchcorder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<cached_dataset::PyStreamCache>()?;
    m.add_class::<cached_dataset::PyStreamCacheReader>()?;
    m.add_class::<cached_dataset::PyCastingStreamCache>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);

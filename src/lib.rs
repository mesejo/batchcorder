use pyo3::prelude::*;
use pyo3_stub_gen::define_stub_info_gatherer;

mod cached_dataset;

/// Register the `batchcorder._batchcorder` Python extension module.
///
/// Exposes:
/// * [`cached_dataset::PyCachedDataset`] — hybrid-cached Arrow dataset
/// * [`cached_dataset::PyCachedDatasetReader`] — independent reader handle
#[pymodule]
fn _batchcorder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<cached_dataset::PyCachedDataset>()?;
    m.add_class::<cached_dataset::PyCachedDatasetReader>()?;
    Ok(())
}

define_stub_info_gatherer!(stub_info);

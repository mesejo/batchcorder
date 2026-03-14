use pyo3::prelude::*;

mod cached_dataset;

/// Register the `multirecord` Python extension module.
///
/// Exposes:
/// * [`cached_dataset::PyCachedDataset`] — hybrid-cached Arrow dataset
/// * [`cached_dataset::PyCachedDatasetReader`] — independent reader handle
#[pymodule]
fn batchcorder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<cached_dataset::PyCachedDataset>()?;
    m.add_class::<cached_dataset::PyCachedDatasetReader>()?;
    Ok(())
}

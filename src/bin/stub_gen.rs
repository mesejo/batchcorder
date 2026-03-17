use pyo3_stub_gen::Result;

fn main() -> Result<()> {
    let stub = _batchcorder::stub_info()?;
    stub.generate()?;
    Ok(())
}

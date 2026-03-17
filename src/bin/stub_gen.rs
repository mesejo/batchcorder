use pyo3_stub_gen::Result;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    let stub = _batchcorder::stub_info()?;
    let module = stub
        .modules
        .get("batchcorder")
        .expect("Module 'batchcorder' not found in stub info");
    let content = module.format_with_config(stub.config.use_type_statement);
    let dest = Path::new("python/batchcorder/_batchcorder.pyi");
    fs::write(dest, content)?;
    Ok(())
}

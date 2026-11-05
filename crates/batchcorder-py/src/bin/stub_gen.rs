use pyo3_stub_gen::{Result, StubInfo};
use std::fs;

/// Generate stub files in mixed layout, but write each module as
/// `<python_root>/<module_path>/<package_name>.pyi` instead of
/// `<python_root>/<module_path>/__init__.pyi`.
///
/// This matches how a compiled sub-extension is laid out on disk:
/// `batchcorder/_batchcorder.pyi` rather than
/// `batchcorder/__init__.pyi`.
fn generate_mixed_as_module(stub: &StubInfo, package_name: &str) -> Result<()> {
    for (name, module) in &stub.modules {
        let dir = stub
            .python_root
            .join(name.replace('-', "_").replace('.', "/"));

        if !dir.exists() {
            fs::create_dir_all(&dir)?;
        }

        let content = module.format_with_config(stub.config.use_type_statement);
        fs::write(dir.join(format!("{package_name}.pyi")), content)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let stub = _batchcorder::stub_info()?;
    generate_mixed_as_module(&stub, "_batchcorder")?;
    Ok(())
}

#[path = "parity/comparator.rs"]
mod comparator;
#[path = "parity/gguf.rs"]
mod gguf;
#[path = "parity/mutations.rs"]
mod mutations;
#[path = "parity/observation.rs"]
mod observation;
#[path = "parity/prototype.rs"]
mod prototype;
#[path = "parity/reader.rs"]
mod reader;
#[path = "parity/surfaces.rs"]
mod surfaces;

use anyhow::{ensure, Context, Result};

#[test]
fn committed_fixture_layout_and_policies() -> Result<()> {
    let root = reader::fixture_root();
    if !reader::present(&root)? {
        return Ok(());
    }
    let mut count = 0;
    for entry in std::fs::read_dir(&root)? {
        let path = entry?.path();
        if !path.is_dir() {
            continue;
        }
        if path
            .file_name()
            .unwrap()
            .to_string_lossy()
            .starts_with("gguf-")
        {
            gguf::validate_fixture(&path)?;
            count += 1;
            continue;
        }
        let fixture = reader::read(&path)?;
        ensure!(fixture.inputs.is_object(), "invalid inputs");
        ensure!(
            comparator::compare(&fixture.expected, &fixture.expected.observation).is_empty(),
            "fixture fails its own policies: {}",
            path.display()
        );
        count += 1;
    }
    ensure!(
        count > 0,
        "fixture directory exists but contains no fixtures"
    );
    Ok(())
}

#[test]
fn gguf_models_cpu() -> Result<()> {
    gguf::run_all(mlx_rs::Device::cpu())
}

#[test]
fn gguf_models_metal() -> Result<()> {
    gguf::run_all(mlx_rs::Device::gpu())
}

#[test]
fn gguf_mutation_qualification() -> Result<()> {
    gguf::qualify_mutations()
}

#[test]
fn public_generation_cpu_parity() -> Result<()> {
    let root = reader::fixture_root();
    if !reader::present(&root)? {
        return Ok(());
    }
    for name in [
        "llama-base",
        "llama-sharded",
        "llama-sliding",
        "llama-quant4",
        "qwen3-base",
        "qwen3-quant4",
    ] {
        let path = root.join(name);
        let fixture = reader::read(&path).with_context(|| name)?;
        mlx_rs::with_device(mlx_rs::Device::cpu(), || prototype::run(&path, &fixture))
            .with_context(|| name)?;
    }
    Ok(())
}

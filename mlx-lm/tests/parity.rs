#[path = "parity/comparator.rs"]
mod comparator;
#[path = "parity/mutations.rs"]
mod mutations;
#[path = "parity/observation.rs"]
mod observation;
#[path = "parity/prototype.rs"]
mod prototype;
#[path = "parity/reader.rs"]
mod reader;

use anyhow::{ensure, Result};

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
fn prototype_llama_base_prefill_cache_and_greedy_parity() -> Result<()> {
    let path = reader::fixture_root().join("llama-base");
    if !reader::present(&path)? {
        return Ok(());
    }
    let fixture = reader::read(&path)?;
    let expected = fixture.expected.prototype_prefill_and_greedy();
    let observed = mlx_rs::with_device(mlx_rs::Device::cpu(), || prototype::run(&path, &fixture))?;
    let failures = comparator::compare(&expected, &observed);
    ensure!(
        failures.is_empty(),
        "{}",
        failures
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("\n")
    );
    Ok(())
}

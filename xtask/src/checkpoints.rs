use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub(crate) fn hash(path: &Path) -> Result<String, String> {
    let out = Command::new("shasum")
        .args(["-a", "256"])
        .arg(path)
        .output()
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!("cannot hash {}", path.display()));
    }
    String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .next()
        .map(str::to_owned)
        .ok_or("missing hash".into())
}

fn read(path: &Path) -> Result<Value, String> {
    serde_json::from_slice(&fs::read(path).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

fn cases(oracle: &Value) -> Vec<String> {
    oracle["contract"]["cases"]
        .as_object()
        .unwrap()
        .keys()
        .flat_map(|name| {
            oracle["contract"]["devices"]
                .as_array()
                .unwrap()
                .iter()
                .map(move |device| format!("{name}/{}", device.as_str().unwrap()))
        })
        .collect()
}

pub(crate) fn checkpoint_evidence(root: &Path, log: &str) -> Result<Value, String> {
    let Some(path) = std::env::var_os("MLX_LM_REAL_CHECKPOINTS") else {
        return Err("required checkpoint manifest is missing".into());
    };
    collect_checkpoint_evidence(root, &PathBuf::from(path), log)
}

fn collect_checkpoint_evidence(root: &Path, path: &Path, log: &str) -> Result<Value, String> {
    let manifest = read(path)?;
    let report_path = manifest["expected_report"]["path"]
        .as_str()
        .ok_or("missing oracle path")?;
    let mut report = read(Path::new(report_path))?;
    if manifest["expected_report"]["sha256"] != hash(Path::new(report_path))? {
        return Err("checkpoint oracle hash mismatch".into());
    }
    for key in [
        "schema_version",
        "entries",
        "cases",
        "devices",
        "python_environment",
    ] {
        if manifest[key] != report["contract"][key] {
            return Err("checkpoint contract mismatch".into());
        }
    }
    for (name, entry) in report["contract"]["entries"]
        .as_object_mut()
        .ok_or("missing entries")?
    {
        entry["local_directory"] = name.clone().into();
    }
    let oracle_path = root.join("conformance/mlx-lm/bump-oracle.json");
    let oracle = read(&oracle_path)?;
    validate_live_oracle(&report, &oracle)?;
    let receipt_path = Path::new(report_path).with_file_name("live-oracle-generation.json");
    let receipt = read(&receipt_path)?;
    for (key, file) in [
        ("generator_sha256", "live_oracle.py"),
        ("oracle_generator_sha256", "real_checkpoints.py"),
        ("requirements_sha256", "requirements.lock"),
        ("committed_sha256", "bump-oracle.json"),
    ] {
        if receipt[key] != hash(&root.join("conformance/mlx-lm").join(file))? {
            return Err(format!("live oracle provenance mismatch: {key}"));
        }
    }
    if receipt["python_version"] != "3.12.14"
        || receipt["live_sha256"] != manifest["expected_report"]["sha256"]
    {
        return Err("live oracle generation receipt mismatch".into());
    }
    let cases = cases(&oracle);
    for case in &cases {
        if log
            .lines()
            .filter(|line| line.contains(&format!("PASS: {case}:")))
            .count()
            != 1
        {
            return Err(format!("missing or duplicate checkpoint execution: {case}"));
        }
    }
    Ok(
        json!({"oracle_sha256": hash(&oracle_path)?, "manifest_path": path, "manifest_sha256": hash(path)?, "live_sha256": manifest["expected_report"]["sha256"], "generation": receipt, "cases": cases}),
    )
}

fn validate_live_oracle(report: &Value, oracle: &Value) -> Result<(), String> {
    if report["schema_version"] != oracle["schema_version"]
        || report["contract"] != oracle["contract"]
        || report["environment"] != oracle["environment"]
        || report["results"]
            .as_object()
            .map(|r| r.keys().collect::<Vec<_>>())
            != oracle["contract"]["cases"]
                .as_object()
                .map(|r| r.keys().collect::<Vec<_>>())
    {
        return Err("live oracle contract or environment mismatch".into());
    }
    for devices in report["results"]
        .as_object()
        .ok_or("missing live results")?
        .values()
    {
        if devices
            .as_object()
            .map(|d| d.keys().map(String::as_str).collect::<Vec<_>>())
            != Some(vec!["cpu", "metal"])
        {
            return Err("live oracle device inventory mismatch".into());
        }
        for device in ["cpu", "metal"] {
            let result = &devices[device];
            if result["completed"] != true
                || result["greedy_ids"].as_array().map(Vec::len) != Some(32)
                || result["greedy_steps"].as_array().map(Vec::len) != Some(32)
            {
                return Err("incomplete live oracle case".into());
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_oracle_requires_complete_results_for_the_input_contract() {
        let oracle: Value =
            serde_json::from_str(include_str!("../../conformance/mlx-lm/bump-oracle.json"))
                .unwrap();
        let mut live = oracle.clone();
        for name in oracle["contract"]["cases"].as_object().unwrap().keys() {
            for device in ["cpu", "metal"] {
                live["results"][name][device] = json!({
                    "completed": true,
                    "greedy_ids": vec![123; 32],
                    "greedy_steps": vec![json!({"top1_id": 123}); 32],
                });
            }
        }
        assert!(validate_live_oracle(&live, &oracle).is_ok());
        for pointer in [
            "/contract/cases",
            "/contract/entries",
            "/contract/devices",
            "/schema_version",
            "/environment",
            "/results",
            "/results/qwen3-06b-4bit/cpu/completed",
            "/results/qwen3-06b-4bit/cpu/greedy_ids",
            "/results/qwen3-06b-4bit/cpu/greedy_steps",
            "/results/qwen3-06b-4bit/metal",
        ] {
            let mut bad = live.clone();
            *bad.pointer_mut(pointer).unwrap() = Value::Null;
            assert!(validate_live_oracle(&bad, &oracle).is_err(), "{pointer}");
        }
        let mut missing = live.clone();
        missing["results"]
            .as_object_mut()
            .unwrap()
            .remove("qwen3-06b-4bit");
        assert!(validate_live_oracle(&missing, &oracle).is_err());
        let mut extra = live.clone();
        extra["results"]["extra"] = live["results"]["qwen3-06b-4bit"].clone();
        assert!(validate_live_oracle(&extra, &oracle).is_err());
        let mut extra_device = live.clone();
        extra_device["results"]["qwen3-06b-4bit"]["extra"] =
            live["results"]["qwen3-06b-4bit"]["cpu"].clone();
        assert!(validate_live_oracle(&extra_device, &oracle).is_err());
    }
}

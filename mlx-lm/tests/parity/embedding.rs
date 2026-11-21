//! EmbeddingGemma 2 text embedder against the NumPy f32 reference fixture.

use anyhow::{ensure, Context, Result};
use mlx_lm::{Embedder, EmbeddingError};
use mlx_rs::{with_device, Array, Device};
use serde_json::Value;
use std::{fs, path::PathBuf};

pub fn fixture() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../conformance/mlx-lm/embedding-fixtures/embedding-gemma2-text")
}

fn json(name: &str) -> Result<Value> {
    Ok(serde_json::from_slice(&fs::read(fixture().join(name))?)?)
}

fn rows(array: &Array) -> Result<Vec<f32>> {
    ensure!(
        array.dtype() == mlx_rs::Dtype::Float32,
        "embeddings must be f32"
    );
    Ok(array.try_as_slice::<f32>()?.to_vec())
}

fn within(label: &str, expected: &[f32], actual: &[f32], atol: f32, rtol: f32) -> Result<()> {
    ensure!(expected.len() == actual.len(), "{label}: length differs");
    for (index, (&e, &a)) in expected.iter().zip(actual).enumerate() {
        ensure!(
            a.is_finite() && (e - a).abs() <= atol + rtol * e.abs(),
            "{label}: element {index} expected {e}, got {a}"
        );
    }
    Ok(())
}

pub fn run(device: Device) -> Result<()> {
    let inputs = json("inputs.json")?;
    let expectations = json("expectations.json")?;
    let tolerance = &expectations["tolerances"]["embeddings"];
    let atol = tolerance["atol"].as_f64().context("atol")? as f32;
    let rtol = tolerance["rtol"].as_f64().context("rtol")? as f32;
    let texts: Vec<&str> = inputs["texts"]
        .as_array()
        .context("texts")?
        .iter()
        .map(|t| t.as_str().context("text"))
        .collect::<Result<_>>()?;
    let ids: Vec<Vec<u32>> = serde_json::from_value(inputs["token_ids"].clone())?;
    let golden = Array::load_safetensors(fixture().join("expectations.safetensors"))?;
    let golden = golden.get("embeddings").context("embeddings golden")?;
    let golden_shape = golden.shape().to_vec();
    let golden = rows(golden)?;

    with_device(device, || {
        let embedder = Embedder::from_dir(fixture())?;
        for (text, expected) in texts.iter().zip(&ids) {
            let encoded = embedder
                .tokenizer()
                .encode_with_special_tokens(text, true)?;
            let encoded: Vec<u32> = encoded.into_iter().map(u32::from).collect();
            ensure!(&encoded == expected, "token ids differ for {text:?}");
        }
        let batch = embedder.embed(&texts)?;
        ensure!(
            batch.shape() == golden_shape.as_slice(),
            "batch shape differs"
        );
        ensure!(
            embedder.dimension() as i32 == golden_shape[1],
            "dimension differs"
        );
        within("batch", &golden, &rows(&batch)?, atol, rtol)?;

        // Each text alone must reproduce its padded-batch row.
        let width = embedder.dimension();
        for (index, text) in texts.iter().enumerate() {
            let single = embedder.embed(&[text])?;
            let expected = &golden[index * width..(index + 1) * width];
            within(
                &format!("single {index}"),
                expected,
                &rows(&single)?,
                atol,
                rtol,
            )?;
        }

        ensure!(matches!(
            embedder.embed(&[]),
            Err(EmbeddingError::EmptyBatch)
        ));
        ensure!(matches!(
            embedder.embed(&["one", "a <|image|> fox"]),
            Err(EmbeddingError::MediaToken { input: 1, .. })
        ));
        let long = vec!["one"; 64].join(" ");
        ensure!(matches!(
            embedder.embed(&[long.as_str()]),
            Err(EmbeddingError::SequenceTooLong {
                input: 0,
                length: 66,
                limit: 64
            })
        ));
        Ok(())
    })
}

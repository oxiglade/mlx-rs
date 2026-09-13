use super::*;
use crate::arch::ArchitectureFactory;
use anyhow::{ensure, Context, Result};
use mlx_rs::ops::indexing::TryIndexOp;
use serde_json::Value;
use std::{
    num::NonZeroUsize,
    path::{Path, PathBuf},
};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm")
}
fn fixture(name: &str) -> PathBuf {
    root().join("fixtures").join(name)
}
fn document(name: &str) -> Result<Value> {
    Ok(serde_json::from_slice(&std::fs::read(
        fixture(name).join("expectations.json"),
    )?)?)
}
fn tokenizer(name: &str) -> Result<Tokenizer> {
    let doc = document(name)?;
    let path = fixture(name).join(
        doc["gguf"]["tokenizer_files"]["tokenizer.json"]["path"]
            .as_str()
            .context("tokenizer path")?,
    );
    Ok(Tokenizer::from_dir(
        path.parent().context("tokenizer directory")?,
    )?)
}
fn load_fixture(name: &str) -> Result<Model> {
    let file = mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || {
        GgufFile::load(fixture(name).join("model.gguf"))
    })?;
    Ok(Model::from_gguf(file, tokenizer(name)?)?)
}
fn compare(
    actual: &Array,
    expected: &safetensors::SafeTensors<'_>,
    doc: &Value,
    key: &str,
) -> Result<()> {
    let tensor = expected.tensor(key)?;
    ensure!(
        actual
            .shape()
            .iter()
            .map(|&n| n as usize)
            .collect::<Vec<_>>()
            == tensor.shape(),
        "shape {key}"
    );
    let dtype = match actual.dtype() {
        Dtype::Float32 => "float32",
        Dtype::Float16 => "float16",
        other => anyhow::bail!("unexpected native dtype {other:?}"),
    };
    ensure!(doc["native_dtypes"][key] == dtype, "native dtype {key}");
    let (atol, rtol) = match doc["tolerance_policy"].as_str().context("policy")? {
        "f32-v1" => (2e-4_f64, 2e-4_f64),
        "gguf-f16-v1" => (5e-3, 5e-3),
        "gguf-affine-v1" => (2e-2, 2e-3),
        other => anyhow::bail!("unknown tolerance {other}"),
    };
    let actual = actual.to_vec_cast::<f32>()?;
    let wanted: Vec<_> = tensor
        .data()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    ensure!(actual.len() == wanted.len(), "length {key}");
    for (i, (a, b)) in actual.into_iter().zip(wanted).enumerate() {
        ensure!(
            a.is_finite()
                && b.is_finite()
                && f64::from((a - b).abs()) <= atol + rtol * f64::from(b.abs()),
            "{key}[{i}]: {a} != {b}"
        );
    }
    Ok(())
}
fn compare_cache(
    cache: &crate::Cache,
    expected: &safetensors::SafeTensors<'_>,
    doc: &Value,
    label: &str,
) -> Result<()> {
    let states = doc["cache"][label].as_array().context("cache states")?;
    let views = cache.logical_layers()?;
    ensure!(views.len() == states.len(), "layer count");
    for (info, view) in cache.info().zip(views) {
        let state = &states[info.layer];
        ensure!(json!(info.processed_tokens) == state["offset"], "offset");
        ensure!(
            json!([view.positions.start, view.positions.end]) == state["retained_range"],
            "retained range"
        );
        for (kind, array) in [("keys", view.keys), ("values", view.values)] {
            compare(
                &array,
                expected,
                doc,
                &format!("cache.{label}.layer{}.{kind}", info.layer),
            )?;
        }
    }
    Ok(())
}
fn forward_fixture(name: &str) -> Result<()> {
    let mut model = load_fixture(name)?;
    let doc = document(name)?;
    let bytes = std::fs::read(fixture(name).join("expectations.safetensors"))?;
    let expected = safetensors::SafeTensors::deserialize(&bytes)?;
    let inputs: Value = serde_json::from_slice(&std::fs::read(fixture(name).join("inputs.json"))?)?;
    ensure!(
        inputs["token_ids"]["canonical"] == doc["prefill"]["token_ids"],
        "canonical inputs"
    );
    let tokens: Vec<u32> = serde_json::from_value(doc["prefill"]["token_ids"].clone())?;
    let greedy: Vec<u32> = serde_json::from_value(doc["decode"]["greedy_ids"].clone())?;
    ensure!(greedy.len() == 8, "eight decode steps");
    for (label, chunk) in [
        ("full", tokens.len()),
        ("chunk1", 1),
        ("chunk3", 3),
        ("chunk8", 8),
    ] {
        let decoder = &mut model.decoder;
        let mut cache = crate::Cache::new(
            decoder.config().model_type.clone(),
            decoder.cache_layout(),
            &crate::CacheOptions::default(),
            NonZeroUsize::new(tokens.len()),
        )?;
        let mut outputs = Vec::new();
        for ids in tokens.chunks(chunk) {
            let mut step = cache.step()?;
            let logits =
                decoder.forward(&Array::from_slice(ids, &[1, ids.len() as i32]), &mut step)?;
            step.evaluate_and_commit(&[&logits])?;
            outputs.push(logits);
        }
        let mut logits = mlx_rs::ops::concatenate(&outputs, 1)?;
        compare(&logits, &expected, &doc, &format!("prefill.{label}.logits"))?;
        compare(&logits, &expected, &doc, "prefill.full.logits")?;
        compare_cache(&cache, &expected, &doc, "after_prefill")?;
        for (index, &id) in greedy.iter().enumerate() {
            let last = logits.try_index((.., -1, ..))?;
            ensure!(
                mlx_rs::ops::indexing::argmax_axis(&last, -1, false)?.try_item_exact::<u32>()?
                    == id,
                "{name}/{label}/greedy{index}"
            );
            let mut step = cache.step()?;
            logits = decoder.forward(&Array::from_slice(&[id], &[1, 1]), &mut step)?;
            step.evaluate_and_commit(&[&logits])?;
            compare(
                &logits.squeeze_axes(&[1])?,
                &expected,
                &doc,
                &format!("decode.step{index}.logits"),
            )?;
        }
        compare_cache(&cache, &expected, &doc, "after_decode")?;
    }
    let prompt: Vec<_> = tokens.into_iter().map(TokenId::from).collect();
    let options = crate::GenerationOptions {
        max_tokens: NonZeroUsize::new(8).unwrap(),
        stop: crate::StopPolicy {
            tokens: crate::StopTokenPolicy::Exact(vec![]),
            ..Default::default()
        },
        ..Default::default()
    };
    let mut observed = Vec::new();
    for event in model.generate(crate::Prompt::Tokens(&prompt), options)? {
        if let crate::GenerationEvent::Token { token_id, .. } = event? {
            observed.push(u32::from(token_id));
        }
    }
    ensure!(observed == greedy, "public greedy prefix");
    Ok(())
}
macro_rules! forwards {
    ($($test:ident => $name:literal),* $(,)?) => { $(#[test] fn $test() -> Result<()> { forward_fixture($name) })* };
}
forwards! {
    gguf_llama_f32 => "gguf-llama-f32", gguf_llama_f16 => "gguf-llama-f16",
    gguf_llama_q4_0 => "gguf-llama-q4_0", gguf_llama_q4_1 => "gguf-llama-q4_1", gguf_llama_q8_0 => "gguf-llama-q8_0",
    gguf_qwen3_f32 => "gguf-qwen3-f32", gguf_qwen3_f16 => "gguf-qwen3-f16",
    gguf_qwen3_q4_0 => "gguf-qwen3-q4_0", gguf_qwen3_q4_1 => "gguf-qwen3-q4_1", gguf_qwen3_q8_0 => "gguf-qwen3-q8_0",
}

#[test]
fn loading_inside_caller_gpu_stream() -> Result<()> {
    for name in ["gguf-llama-q4_1", "gguf-qwen3-q4_1"] {
        let file = mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || {
            GgufFile::load(fixture(name).join("model.gguf"))
        })?;
        mlx_rs::with_stream(&mlx_rs::Stream::gpu(), || {
            Model::from_gguf(file, tokenizer(name).unwrap())
        })?;
    }
    Ok(())
}

fn error_json(error: LoadError) -> Value {
    match error {
        LoadError::Config(ConfigError::MissingField { field }) => {
            json!({"variant":"LoadError::Config(ConfigError::MissingField)","field":field})
        }
        LoadError::Config(ConfigError::InvalidGgufMetadata { key, .. }) => {
            json!({"variant":"LoadError::Config(ConfigError::InvalidGgufMetadata)","key":key})
        }
        LoadError::Config(ConfigError::InvalidNumericField { field, .. }) => {
            json!({"variant":"LoadError::Config(ConfigError::InvalidNumericField)","field":field})
        }
        LoadError::Config(ConfigError::UnsupportedGgufMetadata { key, .. }) => {
            json!({"variant":"LoadError::Config(ConfigError::UnsupportedGgufMetadata)","key":key})
        }
        LoadError::Config(ConfigError::UnsupportedArchitecture(value)) => {
            json!({"variant":"LoadError::Config(ConfigError::UnsupportedArchitecture)","value":value})
        }
        LoadError::Config(ConfigError::UnsupportedRope(_)) => {
            json!({"variant":"LoadError::Config(ConfigError::UnsupportedRope)"})
        }
        LoadError::Weights(crate::WeightError::MissingKey(key)) => {
            json!({"variant":"LoadError::Weights(WeightError::MissingKey)","key":key})
        }
        LoadError::Weights(crate::WeightError::UnexpectedKey(key)) => {
            json!({"variant":"LoadError::Weights(WeightError::UnexpectedKey)","key":key})
        }
        LoadError::Weights(crate::WeightError::ShapeMismatch {
            key,
            expected,
            actual,
        }) => {
            json!({"variant":"LoadError::Weights(WeightError::ShapeMismatch)","key":key,"expected":expected,"actual":actual})
        }
        LoadError::TokenizerVocabularyOutOfRange {
            token_id,
            vocabulary_size,
        } => {
            json!({"variant":"LoadError::TokenizerVocabularyOutOfRange","token_id":u32::from(token_id),"vocabulary_size":vocabulary_size})
        }
        LoadError::TokenizerMetadataMismatch { key, .. } => {
            json!({"variant":"LoadError::TokenizerMetadataMismatch","key":key})
        }
        other => panic!("unexpected error: {other:?}"),
    }
}

mod recipes {
    use anyhow::{bail, ensure, Context, Result};
    use serde_json::Value;
    use std::{
        collections::BTreeMap,
        fmt::Write,
        io::{Cursor, Read},
    };

    fn take<const N: usize>(input: &mut Cursor<&[u8]>) -> Result<[u8; N]> {
        let mut bytes = [0; N];
        input.read_exact(&mut bytes)?;
        Ok(bytes)
    }

    fn u32_value(input: &mut Cursor<&[u8]>) -> Result<u32> {
        Ok(u32::from_le_bytes(take(input)?))
    }

    fn usize_value(input: &mut Cursor<&[u8]>) -> Result<usize> {
        Ok(u64::from_le_bytes(take(input)?).try_into()?)
    }

    fn string(input: &mut Cursor<&[u8]>) -> Result<String> {
        let size = usize_value(input)?;
        let start = input.position() as usize;
        let bytes = input
            .get_ref()
            .get(start..start + size)
            .context("GGUF string")?;
        let value = String::from_utf8(bytes.to_vec())?;
        input.set_position((start + size) as u64);
        Ok(value)
    }

    fn skip_value(input: &mut Cursor<&[u8]>, kind: u32) -> Result<()> {
        let size = match kind {
            4..=6 => 4,
            7 => 1,
            8 => {
                string(input)?;
                return Ok(());
            }
            9 => {
                let element = u32_value(input)?;
                let count = usize_value(input)?;
                ensure!(
                    element != 9 && count <= input.get_ref().len(),
                    "metadata array"
                );
                for _ in 0..count {
                    skip_value(input, element)?;
                }
                return Ok(());
            }
            other => bail!("unsupported fixture metadata type {other}"),
        };
        let end = input.position() + size;
        ensure!(end <= input.get_ref().len() as u64, "truncated metadata");
        input.set_position(end);
        Ok(())
    }

    fn write_string(output: &mut Vec<u8>, value: &str) {
        output.extend_from_slice(&(value.len() as u64).to_le_bytes());
        output.extend_from_slice(value.as_bytes());
    }

    fn write_value(output: &mut Vec<u8>, kind: u32, value: &Value) -> Result<()> {
        match kind {
            4 => output.extend_from_slice(
                &u32::try_from(value.as_u64().context("u32 value")?)?.to_le_bytes(),
            ),
            5 => output.extend_from_slice(
                &i32::try_from(value.as_i64().context("i32 value")?)?.to_le_bytes(),
            ),
            6 => output
                .extend_from_slice(&(value.as_f64().context("f32 value")? as f32).to_le_bytes()),
            7 => output.push(u8::from(value.as_bool().context("bool value")?)),
            8 => write_string(output, value.as_str().context("string value")?),
            9 => {
                let element = u32::try_from(value[0].as_u64().context("array type")?)?;
                let items = value[1].as_array().context("array values")?;
                ensure!(element != 9, "nested metadata array");
                output.extend_from_slice(&element.to_le_bytes());
                output.extend_from_slice(&(items.len() as u64).to_le_bytes());
                for item in items {
                    write_value(output, element, item)?;
                }
            }
            other => bail!("unsupported recipe metadata type {other}"),
        }
        Ok(())
    }

    #[derive(Clone)]
    struct Tensor {
        shape: Vec<usize>,
        kind: u32,
        bytes: Vec<u8>,
    }

    fn align(bytes: &mut Vec<u8>) {
        bytes.resize(bytes.len().next_multiple_of(32), 0);
    }

    pub(super) fn materialize(bytes: &[u8], recipe: &Value) -> Result<Vec<u8>> {
        let mut input = Cursor::new(bytes);
        ensure!(
            &take::<4>(&mut input)? == b"GGUF" && u32_value(&mut input)? == 3,
            "GGUF v3"
        );
        let count = usize_value(&mut input)?;
        let metadata_count = usize_value(&mut input)?;
        let mut metadata = BTreeMap::new();
        for _ in 0..metadata_count {
            let key = string(&mut input)?;
            let start = input.position() as usize;
            let kind = u32_value(&mut input)?;
            skip_value(&mut input, kind)?;
            metadata.insert(key, bytes[start..input.position() as usize].to_vec());
        }
        if let Some(alignment) = metadata.get("general.alignment") {
            ensure!(
                alignment == &[4_u32.to_le_bytes(), 32_u32.to_le_bytes()].concat(),
                "fixture alignment"
            );
        }
        let mut descriptors = Vec::new();
        for _ in 0..count {
            let key = string(&mut input)?;
            let rank = u32_value(&mut input)?;
            ensure!((1..=4).contains(&rank), "tensor rank");
            let mut shape = (0..rank)
                .map(|_| usize_value(&mut input))
                .collect::<Result<Vec<_>>>()?;
            shape.reverse();
            let kind = u32_value(&mut input)?;
            let offset = usize_value(&mut input)?;
            descriptors.push((key, shape, kind, offset));
        }
        let base = (input.position() as usize).next_multiple_of(32);
        let mut tensors = BTreeMap::new();
        for (key, shape, kind, offset) in descriptors {
            let elements: usize = shape.iter().product();
            let size = match kind {
                0 => elements * 4,
                1 => elements * 2,
                2 | 3 | 8 => {
                    ensure!(shape.len() == 2 && shape[1] % 32 == 0, "block tensor shape");
                    elements / 32
                        * match kind {
                            2 => 18,
                            3 => 20,
                            _ => 34,
                        }
                }
                other => bail!("unsupported fixture tensor type {other}"),
            };
            let data = bytes
                .get(base + offset..base + offset + size)
                .context("tensor bytes")?;
            tensors.insert(
                key,
                Tensor {
                    shape,
                    kind,
                    bytes: data.to_vec(),
                },
            );
        }
        let key = recipe["key"].as_str().context("recipe key")?;
        match recipe["operation"].as_str().context("recipe operation")? {
            "set_metadata" => {
                let kind = u32::try_from(recipe["type"].as_u64().context("metadata type")?)?;
                let mut value = kind.to_le_bytes().to_vec();
                write_value(&mut value, kind, &recipe["value"])?;
                metadata.insert(key.to_owned(), value);
            }
            "remove_metadata" => {
                metadata.remove(key).context("metadata key")?;
            }
            "remove_tensor" => {
                tensors.remove(key).context("tensor key")?;
            }
            operation @ ("copy_tensor" | "rename_tensor") => {
                let tensor = tensors.get(key).context("tensor key")?.clone();
                let target = recipe["target"].as_str().context("target key")?;
                tensors.insert(target.to_owned(), tensor);
                if operation == "rename_tensor" {
                    tensors.remove(key);
                }
            }
            operation @ ("truncate_rows" | "truncate_columns") => {
                let tensor = tensors.get_mut(key).context("tensor key")?;
                ensure!(
                    tensor.shape.len() == 2 && tensor.shape[0] > 0,
                    "matrix shape"
                );
                let stride = tensor.bytes.len() / tensor.shape[0];
                if operation == "truncate_rows" {
                    let rows = usize::try_from(recipe["rows"].as_u64().context("rows")?)?;
                    ensure!(rows <= tensor.shape[0], "truncate rows");
                    tensor.bytes.truncate(stride * rows);
                    tensor.shape[0] = rows;
                } else {
                    let columns = usize::try_from(recipe["columns"].as_u64().context("columns")?)?;
                    ensure!(columns <= tensor.shape[1], "truncate columns");
                    let width = stride * columns / tensor.shape[1];
                    tensor.bytes = tensor
                        .bytes
                        .chunks_exact(stride)
                        .flat_map(|row| row[..width].iter().copied())
                        .collect();
                    tensor.shape[1] = columns;
                }
            }
            other => bail!("unknown GGUF recipe operation {other}"),
        }
        let mut header = b"GGUF".to_vec();
        header.extend_from_slice(&3_u32.to_le_bytes());
        header.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        header.extend_from_slice(&(metadata.len() as u64).to_le_bytes());
        for (key, value) in metadata {
            write_string(&mut header, &key);
            header.extend_from_slice(&value);
        }
        let mut payload = Vec::new();
        for (key, tensor) in tensors {
            align(&mut payload);
            write_string(&mut header, &key);
            header.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for dimension in tensor.shape.iter().rev() {
                header.extend_from_slice(&(*dimension as u64).to_le_bytes());
            }
            header.extend_from_slice(&tensor.kind.to_le_bytes());
            header.extend_from_slice(&(payload.len() as u64).to_le_bytes());
            payload.extend_from_slice(&tensor.bytes);
        }
        align(&mut header);
        header.extend_from_slice(&payload);
        Ok(header)
    }

    pub(super) fn verify_hash(bytes: &[u8], recipe: &Value) -> Result<()> {
        let actual = hex_sha256(bytes);
        let expected = recipe["materialized_sha256"]
            .as_str()
            .context("frozen recipe hash")?;
        ensure!(
            actual == expected,
            "{}: materialized SHA-256 expected {expected}, got {actual}",
            recipe["id"]
        );
        Ok(())
    }

    // Same dependency-free SHA-256 used by xtask::verify_oracle_boundary.
    fn hex_sha256(bytes: &[u8]) -> String {
        let mut encoded = String::with_capacity(64);
        for byte in sha256(bytes) {
            write!(&mut encoded, "{byte:02x}").expect("write SHA-256 hex");
        }
        encoded
    }

    fn sha256(bytes: &[u8]) -> [u8; 32] {
        const INITIAL: [u32; 8] = [
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
            0x5be0cd19,
        ];
        const ROUND: [u32; 64] = [
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
            0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
            0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
            0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
            0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
            0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
            0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
            0xc67178f2,
        ];

        let bit_len = (bytes.len() as u64).wrapping_mul(8);
        let mut padded = bytes.to_vec();
        padded.push(0x80);
        while padded.len() % 64 != 56 {
            padded.push(0);
        }
        padded.extend_from_slice(&bit_len.to_be_bytes());

        let mut hash = INITIAL;
        for chunk in padded.chunks(64) {
            let mut words = [0u32; 64];
            for (index, word) in words.iter_mut().take(16).enumerate() {
                *word = u32::from_be_bytes(chunk[index * 4..index * 4 + 4].try_into().unwrap());
            }
            for index in 16..64 {
                let s0 = words[index - 15].rotate_right(7)
                    ^ words[index - 15].rotate_right(18)
                    ^ (words[index - 15] >> 3);
                let s1 = words[index - 2].rotate_right(17)
                    ^ words[index - 2].rotate_right(19)
                    ^ (words[index - 2] >> 10);
                words[index] = words[index - 16]
                    .wrapping_add(s0)
                    .wrapping_add(words[index - 7])
                    .wrapping_add(s1);
            }

            let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = hash;
            for index in 0..64 {
                let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
                let choose = (e & f) ^ (!e & g);
                let temp1 = h
                    .wrapping_add(sum1)
                    .wrapping_add(choose)
                    .wrapping_add(ROUND[index])
                    .wrapping_add(words[index]);
                let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
                let majority = (a & b) ^ (a & c) ^ (b & c);
                let temp2 = sum0.wrapping_add(majority);
                h = g;
                g = f;
                f = e;
                e = d.wrapping_add(temp1);
                d = c;
                c = b;
                b = a;
                a = temp1.wrapping_add(temp2);
            }
            for (value, compressed) in hash.iter_mut().zip([a, b, c, d, e, f, g, h]) {
                *value = value.wrapping_add(compressed);
            }
        }

        let mut digest = [0u8; 32];
        for (bytes, value) in digest.chunks_mut(4).zip(hash) {
            bytes.copy_from_slice(&value.to_be_bytes());
        }
        digest
    }

    #[test]
    fn sha256_known_vector() {
        assert_eq!(
            hex_sha256(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}

fn recipe_bytes(case: &Value) -> Result<Vec<u8>> {
    let base = fixture(case["base"].as_str().context("recipe base")?);
    let bytes = if case["operation"] == "load_core" {
        std::fs::read(base)?
    } else {
        recipes::materialize(&std::fs::read(base.join("model.gguf"))?, case)?
    };
    recipes::verify_hash(&bytes, case)?;
    Ok(bytes)
}

#[test]
fn malformed_recipe_bytes_match_python() -> Result<()> {
    let cases: Value = serde_json::from_slice(&std::fs::read(root().join("gguf_cases.json"))?)?;
    for case in cases["cases"].as_array().context("recipes")? {
        let mut bytes = recipe_bytes(case)?;
        bytes[0] ^= 1;
        ensure!(
            recipes::verify_hash(&bytes, case).is_err(),
            "corrupt recipe must fail hash check"
        );
    }
    Ok(())
}

#[test]
fn malformed_recipes_exact_typed_errors() -> Result<()> {
    let cases: Value = serde_json::from_slice(&std::fs::read(root().join("gguf_cases.json"))?)?;
    let temp = tempfile::tempdir()?;
    for case in cases["cases"].as_array().context("recipes")? {
        let id = case["id"].as_str().context("id")?;
        let bytes = recipe_bytes(case)?;
        let actual = if case["operation"] == "load_core" {
            let error = GgufFile::load(fixture(case["base"].as_str().unwrap()))
                .err()
                .context("expected core rejection")?;
            ensure!(
                matches!(error, mlx_rs::io::GgufError::Exception(_)),
                "opaque core exception"
            );
            json!({"variant":"GgufError::Exception"})
        } else {
            std::fs::write(temp.path().join(format!("{id}.gguf")), bytes)?;
            let file = mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || {
                GgufFile::load(temp.path().join(format!("{id}.gguf")))
            })?;
            let error = Model::from_gguf(file, tokenizer(case["base"].as_str().unwrap())?)
                .err()
                .context("expected model rejection")?;
            error_json(error)
        };
        ensure!(
            actual == case["expected"],
            "{id}: expected {}, got {actual}",
            case["expected"]
        );
    }
    Ok(())
}

#[test]
fn metadata_required_order_and_exact_scalar_types() -> Result<()> {
    for field in [
        "block_count",
        "embedding_length",
        "feed_forward_length",
        "attention.head_count",
        "attention.head_count_kv",
        "attention.layer_norm_rms_epsilon",
        "rope.freq_base",
    ] {
        let name = format!("llama.{field}");
        let base = GgufFile::load(fixture("gguf-llama-f32").join("model.gguf"))?;
        let doc = document("gguf-llama-f32")?;
        let weights = WeightManifest::from_gguf(&base)?
            .normalize_gguf(|key| crate::arch::llama::Factory.map_gguf_key(key))?;
        for replacement in [
            None,
            Some(GgufMetadataValue::String("wrong".into())),
            Some(Array::from_slice(&[2_u32], &[1]).into()),
            Some(Array::from_bool(true).into()),
        ] {
            let mut file = GgufFile::new()?;
            for key in doc["gguf"]["metadata"]
                .as_object()
                .context("metadata")?
                .keys()
            {
                if key != &name {
                    file.insert_metadata(key, base.get_metadata(key)?.context("metadata value")?)?;
                }
            }
            if let Some(value) = replacement.clone() {
                file.insert_metadata(&name, value)?;
            }
            let error = raw_config(&file, &weights)
                .err()
                .context("expected metadata rejection")?;
            match (replacement, error) {
                (None, LoadError::Config(ConfigError::MissingField { field })) => {
                    ensure!(field == name)
                }
                (Some(_), LoadError::Config(ConfigError::InvalidGgufMetadata { key, .. })) => {
                    ensure!(key == name)
                }
                (_, other) => anyhow::bail!("{name}: {other:?}"),
            }
        }
    }
    Ok(())
}

#[test]
fn metadata_optional_profile_rejections() -> Result<()> {
    let base = GgufFile::load(fixture("gguf-llama-f32").join("model.gguf"))?;
    let doc = document("gguf-llama-f32")?;
    let weights = WeightManifest::from_gguf(&base)?
        .normalize_gguf(|key| crate::arch::llama::Factory.map_gguf_key(key))?;
    for (key, value, variant) in [
        (
            "llama.rope.scaling.factor",
            Array::from_f32(2.0).into(),
            "unsupported",
        ),
        (
            "llama.use_parallel_residual",
            Array::from_bool(true).into(),
            "unsupported",
        ),
        (
            "llama.attention.sliding_window",
            Array::from_int(8).into(),
            "unsupported",
        ),
        (
            "llama.tensor_data_layout",
            GgufMetadataValue::String("custom".into()),
            "unsupported",
        ),
        (
            "llama.attention.scale",
            Array::from_f32(1.0).into(),
            "unsupported",
        ),
        (
            "llama.rope.dimension_count",
            Array::from_int(8).into(),
            "rope",
        ),
        (
            "llama.rope.scaling.finetuned",
            Array::from_int(1).into(),
            "mistyped",
        ),
        ("general.file_type", Array::from_f32(1.0).into(), "mistyped"),
        (
            "llama.context_length",
            Array::from_int(-1).into(),
            "numeric",
        ),
    ] {
        let mut file = GgufFile::new()?;
        for source in doc["gguf"]["metadata"]
            .as_object()
            .context("metadata")?
            .keys()
        {
            if source != key {
                file.insert_metadata(
                    source,
                    base.get_metadata(source)?.context("metadata value")?,
                )?;
            }
        }
        file.insert_metadata(key, value)?;
        let error = raw_config(&file, &weights)
            .err()
            .context("expected profile rejection")?;
        match (variant, error) {
            (
                "unsupported",
                LoadError::Config(ConfigError::UnsupportedGgufMetadata { key: observed, .. }),
            ) => ensure!(observed == key),
            (
                "mistyped",
                LoadError::Config(ConfigError::InvalidGgufMetadata { key: observed, .. }),
            ) => ensure!(observed == key),
            ("numeric", LoadError::Config(ConfigError::InvalidNumericField { field, .. })) => {
                ensure!(field == key)
            }
            ("rope", LoadError::Config(ConfigError::UnsupportedRope(_))) => {}
            (_, other) => anyhow::bail!("{key}: {other:?}"),
        }
    }
    Ok(())
}

#[test]
fn evaluation_failure_returns_no_model() -> Result<()> {
    mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || -> Result<()> {
        let name = "gguf-llama-f32";
        let base = GgufFile::load(fixture(name).join("model.gguf"))?;
        let doc = document(name)?;
        let mut file = GgufFile::new()?;
        for key in doc["gguf"]["metadata"]
            .as_object()
            .context("metadata")?
            .keys()
        {
            file.insert_metadata(key, base.get_metadata(key)?.context("metadata value")?)?;
        }
        let singular = mlx_rs::linalg::inv(Array::from_slice(&[0_f32], &[1, 1]))?;
        for (key, array) in base.arrays()? {
            if key == "output.weight" {
                file.insert_array(key, &mlx_rs::ops::broadcast_to(&singular, array.shape())?)?;
            } else {
                file.insert_array(key, &array)?;
            }
        }
        ensure!(matches!(
            Model::from_gguf(file, tokenizer(name)?),
            Err(LoadError::Weights(crate::WeightError::Exception(_)))
        ));
        Ok(())
    })
}

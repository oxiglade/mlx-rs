use std::{
    env,
    ffi::{OsStr, OsString},
    fs,
    path::{Path, PathBuf},
    sync::{Mutex, MutexGuard},
};

use anyhow::{bail, Context, Result};
use mlx_lm::{
    cache::KeyValueCache,
    models::llama::{load_llama_model, load_llama_tokenizer, sample, ModelInput},
};
use mlx_rs::{
    module::Module,
    ops::{concatenate, indexing::IndexOp},
    Array, Dtype,
};
use safetensors::{Dtype as SafeDtype, SafeTensors};
use serde::Deserialize;
use tempfile::{tempdir, TempDir};

static ENVIRONMENT: Mutex<()> = Mutex::new(());

#[derive(Debug, Deserialize)]
struct Expectations {
    schema_version: u32,
    fixture: String,
    prompt: Prompt,
    prefill_logits: TensorExpectation,
    decode: DecodeExpectation,
    prefill_cache: CacheExpectation,
    qualification: Qualification,
}

#[derive(Debug, Deserialize)]
struct Prompt {
    text: String,
    token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct TensorExpectation {
    tensor: String,
    shape: Vec<usize>,
    dtype: String,
    policy: FloatPolicy,
}

#[derive(Debug, Deserialize)]
struct FloatPolicy {
    atol: f32,
    rtol: f32,
}

#[derive(Debug, Deserialize)]
struct DecodeExpectation {
    steps: usize,
    token_ids: Vec<u32>,
}

#[derive(Debug, Deserialize)]
struct CacheExpectation {
    layers: Vec<LayerCacheExpectation>,
}

#[derive(Debug, Deserialize)]
struct LayerCacheExpectation {
    keys: ArrayExpectation,
    values: ArrayExpectation,
}

#[derive(Debug, Deserialize)]
struct ArrayExpectation {
    shape: Vec<i32>,
    dtype: String,
}

#[derive(Debug, Deserialize)]
struct Qualification {
    weights: String,
    comparator_rejects: bool,
}

#[derive(Debug)]
struct Observed {
    prefill_logits: Vec<f32>,
    prefill_shape: Vec<i32>,
    prefill_dtype: Dtype,
    token_ids: Vec<u32>,
}

#[derive(Debug, Clone, Default)]
struct InspectableCache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl KeyValueCache for InspectableCache {
    fn offset(&self) -> i32 {
        self.offset
    }

    fn max_size(&self) -> Option<i32> {
        None
    }

    fn update_and_fetch(
        &mut self,
        keys: Array,
        values: Array,
    ) -> std::result::Result<(Array, Array), mlx_rs::error::Exception> {
        match (self.keys.take(), self.values.take()) {
            (Some(existing_keys), Some(existing_values)) => {
                self.keys = Some(concatenate(&[existing_keys, keys], -2)?);
                self.values = Some(concatenate(&[existing_values, values], -2)?);
            }
            _ => {
                self.keys = Some(keys);
                self.values = Some(values);
            }
        }
        let keys = self.keys.as_ref().expect("cache keys must be initialized");
        self.offset = keys.dim(-2);
        Ok((
            keys.clone(),
            self.values
                .as_ref()
                .expect("cache values must be initialized")
                .clone(),
        ))
    }
}

struct OfflineEnvironment {
    _lock: MutexGuard<'static, ()>,
    cache: TempDir,
    previous: Vec<(&'static str, Option<OsString>)>,
}

impl OfflineEnvironment {
    fn enter() -> Result<Self> {
        let lock = ENVIRONMENT
            .lock()
            .expect("offline environment lock poisoned");
        let cache = tempdir()?;
        let variables = [
            ("HF_HOME", cache.path().join("hf-home").into_os_string()),
            (
                "HF_HUB_CACHE",
                cache.path().join("hf-hub-cache").into_os_string(),
            ),
            (
                "HUGGINGFACE_HUB_CACHE",
                cache.path().join("huggingface-hub-cache").into_os_string(),
            ),
            (
                "TRANSFORMERS_CACHE",
                cache.path().join("transformers-cache").into_os_string(),
            ),
            ("HF_HUB_OFFLINE", OsString::from("1")),
            ("TRANSFORMERS_OFFLINE", OsString::from("1")),
            ("HF_ENDPOINT", OsString::from("http://127.0.0.1:9")),
        ];
        let mut previous = Vec::with_capacity(variables.len());
        for (name, value) in variables {
            previous.push((name, env::var_os(name)));
            env::set_var(name, value);
        }
        Ok(Self {
            _lock: lock,
            cache,
            previous,
        })
    }

    fn assert_unused(&self) -> Result<()> {
        if fs::read_dir(self.cache.path())?.next().is_some() {
            bail!(
                "fixture loading accessed a Hugging Face cache under {}",
                self.cache.path().display()
            );
        }
        Ok(())
    }
}

impl Drop for OfflineEnvironment {
    fn drop(&mut self) {
        for (name, value) in self.previous.drain(..).rev() {
            match value {
                Some(value) => env::set_var(name, value),
                None => env::remove_var(name),
            }
        }
    }
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/sentinel/fixtures/tiny-llama")
}

fn read_expectations(fixture: &Path) -> Result<Expectations> {
    let bytes = fs::read(fixture.join("expectations.json"))?;
    serde_json::from_slice(&bytes).context("parse sentinel expectations")
}

fn read_expected_logits(fixture: &Path, expected: &TensorExpectation) -> Result<Vec<f32>> {
    let bytes = fs::read(fixture.join("expectations.safetensors"))?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    let tensor = tensors.tensor(&expected.tensor)?;
    if tensor.dtype() != SafeDtype::F32 {
        bail!("{} is not F32", expected.tensor);
    }
    if tensor.shape() != expected.shape {
        bail!(
            "{} shape is {:?}, expected {:?}",
            expected.tensor,
            tensor.shape(),
            expected.shape
        );
    }
    let width = std::mem::size_of::<f32>();
    tensor
        .data()
        .chunks_exact(width)
        .map(|bytes| {
            let word: [u8; 4] = bytes.try_into().expect("four-byte f32 chunk");
            Ok(f32::from_le_bytes(word))
        })
        .collect()
}

fn materialize_last_logits(logits: &Array) -> Result<Array> {
    logits
        .index((0, -1, ..))
        .multiply(Array::from_f32(1.0))
        .context("materialize final-position logits")
}

fn greedy_token(logits: &Array) -> Result<u32> {
    let token = sample(logits, 0.0)?;
    token.eval()?;
    Ok(token.item_exact::<u32>())
}

fn check_cache(caches: &[Option<InspectableCache>], expected: &CacheExpectation) -> Result<()> {
    if caches.len() != expected.layers.len() {
        bail!(
            "cache has {} layers, expected {}",
            caches.len(),
            expected.layers.len()
        );
    }
    for (index, (cache, expected)) in caches.iter().zip(&expected.layers).enumerate() {
        let cache = cache
            .as_ref()
            .with_context(|| format!("cache layer {index} is absent"))?;
        check_cached_array(index, "keys", cache.keys.as_ref(), &expected.keys)?;
        check_cached_array(index, "values", cache.values.as_ref(), &expected.values)?;
    }
    Ok(())
}

fn check_cached_array(
    layer: usize,
    name: &str,
    actual: Option<&Array>,
    expected: &ArrayExpectation,
) -> Result<()> {
    let actual = actual.with_context(|| format!("cache layer {layer} {name} are absent"))?;
    if actual.shape() != expected.shape {
        bail!(
            "cache layer {layer} {name} shape is {:?}, expected {:?}",
            actual.shape(),
            expected.shape
        );
    }
    if expected.dtype != "F32" || actual.dtype() != Dtype::Float32 {
        bail!(
            "cache layer {layer} {name} dtype is {:?}, expected {}",
            actual.dtype(),
            expected.dtype
        );
    }
    Ok(())
}

fn run_fixture(fixture: &Path, expected: &Expectations) -> Result<Observed> {
    let offline = OfflineEnvironment::enter()?;
    let tokenizer = load_llama_tokenizer(fixture)?;
    let encoding = tokenizer
        .encode(expected.prompt.text.as_str(), false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if encoding.get_ids() != expected.prompt.token_ids {
        bail!(
            "tokenizer encoded {:?}, expected {:?}",
            encoding.get_ids(),
            expected.prompt.token_ids
        );
    }
    let decoded = tokenizer
        .decode(&expected.prompt.token_ids, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    let round_trip = tokenizer
        .encode(decoded, false)
        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
    if round_trip.get_ids() != expected.prompt.token_ids {
        bail!("tokenizer does not round-trip the sentinel prompt");
    }

    let mut model = load_llama_model(fixture)?;
    let prompt = Array::from_slice(
        &expected.prompt.token_ids,
        &[1, expected.prompt.token_ids.len() as i32],
    );
    let mut cache = Vec::<Option<InspectableCache>>::new();
    let logits = model.forward(ModelInput {
        inputs: &prompt,
        mask: None,
        cache: &mut cache,
    })?;
    let prefill = materialize_last_logits(&logits)?;
    prefill.eval()?;
    check_cache(&cache, &expected.prefill_cache)?;

    let mut token_ids = Vec::with_capacity(expected.decode.steps);
    let mut next_token = greedy_token(&prefill)?;
    for step in 0..expected.decode.steps {
        token_ids.push(next_token);
        if step + 1 < expected.decode.steps {
            let input = Array::from_slice(&[next_token], &[1, 1]);
            let logits = model.forward(ModelInput {
                inputs: &input,
                mask: None,
                cache: &mut cache,
            })?;
            next_token = greedy_token(&materialize_last_logits(&logits)?)?;
        }
    }
    offline.assert_unused()?;
    Ok(Observed {
        prefill_logits: prefill.to_vec_exact::<f32>().unwrap(),
        prefill_shape: prefill.shape().to_vec(),
        prefill_dtype: prefill.dtype(),
        token_ids,
    })
}

fn compare(observed: &Observed, expected: &Expectations, expected_logits: &[f32]) -> Result<()> {
    let expected_shape = expected
        .prefill_logits
        .shape
        .iter()
        .map(|&dimension| i32::try_from(dimension))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    if observed.prefill_shape != expected_shape {
        bail!(
            "prefill logits shape is {:?}, expected {:?}",
            observed.prefill_shape,
            expected.prefill_logits.shape
        );
    }
    if expected.prefill_logits.dtype != "F32" || observed.prefill_dtype != Dtype::Float32 {
        bail!(
            "prefill logits dtype is {:?}, expected {}",
            observed.prefill_dtype,
            expected.prefill_logits.dtype
        );
    }
    if observed.prefill_logits.len() != expected_logits.len() {
        bail!(
            "prefill logits length is {}, expected {}",
            observed.prefill_logits.len(),
            expected_logits.len()
        );
    }
    let policy = &expected.prefill_logits.policy;
    for (index, (&actual, &expected)) in observed
        .prefill_logits
        .iter()
        .zip(expected_logits)
        .enumerate()
    {
        if !actual.is_finite() || !expected.is_finite() {
            bail!("prefill logit {index} is non-finite: actual={actual}, expected={expected}");
        }
        let difference = (actual - expected).abs();
        let limit = policy.atol + policy.rtol * expected.abs();
        if difference > limit {
            bail!(
                "prefill logit {index} differs: actual={actual}, expected={expected}, difference={difference}, limit={limit}"
            );
        }
    }
    if observed.token_ids != expected.decode.token_ids {
        bail!(
            "decoded token ids are {:?}, expected {:?}",
            observed.token_ids,
            expected.decode.token_ids
        );
    }
    Ok(())
}

fn copy_fixture_with_weights(source: &Path, weights: &OsStr) -> Result<TempDir> {
    let scratch = tempdir()?;
    fs::copy(
        source.join("config.json"),
        scratch.path().join("config.json"),
    )?;
    fs::copy(
        source.join("tokenizer.json"),
        scratch.path().join("tokenizer.json"),
    )?;
    fs::copy(
        source.join(weights),
        scratch.path().join("model.safetensors"),
    )?;
    Ok(scratch)
}

#[test]
fn offline_llama_prefill_decode_and_cache_match_numpy() -> Result<()> {
    let fixture = fixture_dir();
    let expected = read_expectations(&fixture)?;
    if expected.schema_version != 1 || expected.fixture != "tiny-llama" {
        bail!("unsupported sentinel expectation bundle");
    }
    let expected_logits = read_expected_logits(&fixture, &expected.prefill_logits)?;
    let observed = run_fixture(&fixture, &expected)?;
    compare(&observed, &expected, &expected_logits)
}

#[test]
fn perturbed_weights_are_rejected_by_the_sentinel_comparator() -> Result<()> {
    let fixture = fixture_dir();
    let expected = read_expectations(&fixture)?;
    if !expected.qualification.comparator_rejects {
        bail!("generator did not qualify the perturbation");
    }
    let expected_logits = read_expected_logits(&fixture, &expected.prefill_logits)?;
    let scratch = copy_fixture_with_weights(&fixture, OsStr::new(&expected.qualification.weights))?;
    let observed = run_fixture(scratch.path(), &expected)?;
    match compare(&observed, &expected, &expected_logits) {
        Err(error) if error.to_string().starts_with("prefill logit ") => {}
        Err(error) => bail!("perturbation was rejected for the wrong reason: {error}"),
        Ok(()) => bail!("sentinel comparator accepted the perturbed weights"),
    }
    Ok(())
}

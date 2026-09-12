use std::path::Path;

use anyhow::{ensure, Context, Result};
use mlx_lm::{
    cache::KeyValueCache,
    models::llama::{sample, Model, ModelArgs, ModelInput},
};
use mlx_rs::{
    module::{Module, ModuleParametersExt},
    ops::{concatenate, indexing::IndexOp},
    Array, Dtype,
};
use safetensors::Dtype as SafeDtype;

use super::{
    observation::{CacheState, Observation, Tensor},
    reader::Fixture,
};

#[derive(Clone, Default)]
struct Cache {
    keys: Option<Array>,
    values: Option<Array>,
    offset: i32,
}

impl KeyValueCache for Cache {
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
    ) -> Result<(Array, Array), mlx_rs::error::Exception> {
        let added = keys.dim(-2);
        let keys = match &self.keys {
            Some(old) => concatenate(&[old, &keys], -2)?,
            None => keys,
        };
        let values = match &self.values {
            Some(old) => concatenate(&[old, &values], -2)?,
            None => values,
        };
        self.keys = Some(keys.clone());
        self.values = Some(values.clone());
        self.offset += added;
        Ok((keys, values))
    }
}

pub fn observe(array: &Array) -> Result<Tensor> {
    // Cache transposes and sliced logits need row-major materialization before exact host reads.
    let array = array.contiguous()?;
    array.eval()?;
    let shape = array
        .shape()
        .iter()
        .map(|&d| usize::try_from(d))
        .collect::<Result<Vec<_>, _>>()?;
    let (dtype, bytes) = match array.dtype() {
        Dtype::Float32 => (
            SafeDtype::F32,
            array
                .to_vec_exact::<f32>()?
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect(),
        ),
        dtype => anyhow::bail!("prototype emitted unsupported dtype {dtype:?}"),
    };
    Ok(Tensor {
        shape,
        dtype,
        bytes,
    })
}

pub fn run(path: &Path, fixture: &Fixture) -> Result<Observation> {
    let mut model = load(path)?;
    let prompt = Array::from_slice(
        &fixture.prefill.token_ids,
        &[1, i32::try_from(fixture.prefill.length)?],
    );
    let mut caches = Vec::<Option<Cache>>::new();
    let mut logits = model.forward(ModelInput {
        inputs: &prompt,
        mask: None,
        cache: &mut caches,
    })?;
    let mut observed = Observation::default();
    observed
        .tensors
        .insert("prefill.full.logits".into(), observe(&logits)?);
    for (layer, cache) in caches.iter().enumerate() {
        let cache = cache.as_ref().context("prototype omitted a cache layer")?;
        let key = format!("cache.after_prefill.layer{layer}");
        let keys = cache.keys.as_ref().context("prototype omitted keys")?;
        let values = cache.values.as_ref().context("prototype omitted values")?;
        observed
            .tensors
            .insert(format!("{key}.keys"), observe(keys)?);
        observed
            .tensors
            .insert(format!("{key}.values"), observe(values)?);
        let offset = usize::try_from(cache.offset)?;
        let retained = usize::try_from(keys.dim(-2))?;
        observed.caches.insert(
            key,
            CacheState {
                offset,
                retained: offset
                    .checked_sub(retained)
                    .context("cache range exceeds offset")?..offset,
            },
        );
    }
    let mut ids = Vec::new();
    for step in 0..8 {
        let token = sample(&logits.index((0, -1, ..)), 0.0)?;
        token.eval()?;
        let token = token.item_exact::<u32>();
        ids.push(token);
        if step < 7 {
            let input = Array::from_slice(&[token], &[1, 1]);
            logits = model.forward(ModelInput {
                inputs: &input,
                mask: None,
                cache: &mut caches,
            })?;
        }
    }
    observed.greedy_ids = Some(ids);
    Ok(observed)
}

fn load(path: &Path) -> Result<Model> {
    let mut config: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("config.json"))?)?;
    let heads = config["num_attention_heads"]
        .as_i64()
        .context("missing attention heads")?;
    let hidden = config["hidden_size"]
        .as_i64()
        .context("missing hidden size")?;
    ensure!(heads > 0 && hidden > 0, "invalid prototype dimensions");
    if config["head_dim"].is_null() {
        ensure!(hidden % heads == 0, "head dimension is not integral");
        config["head_dim"] = (hidden / heads).into();
    }
    if config["num_key_value_heads"].is_null() {
        config["num_key_value_heads"] = heads.into();
    }
    if config["rope_theta"].is_null() {
        config["rope_theta"] = 10000.into();
    }
    if config["max_position_embeddings"].is_null() {
        ensure!(
            config["rope_scaling"].is_null(),
            "scaled prototype RoPE needs max_position_embeddings"
        );
        // Python permits None; the prototype's default RoPE ignores this required integer.
        config["max_position_embeddings"] = 0.into();
    }
    let args: ModelArgs = serde_json::from_value(config)?;
    let mut model = Model::new(args)?;
    model.load_safetensors(path.join("model.safetensors"))?;
    Ok(model)
}

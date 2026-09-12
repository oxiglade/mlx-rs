use std::path::Path;

use anyhow::Result;
use mlx_lm::{oracle_hooks, Model, TokenId};
use mlx_rs::{
    ops::indexing::{argmax_axis, IndexOp},
    Array, Dtype,
};
use safetensors::Dtype as SafeDtype;

use super::{
    observation::{CacheState, Observation, Tensor},
    reader::Fixture,
};

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
    let mut model = Model::from_dir(path)?;
    let tokens: Vec<TokenId> = fixture
        .prefill
        .token_ids
        .iter()
        .copied()
        .map(TokenId::from)
        .collect();
    let (mut logits, mut session) = oracle_hooks::prefill_logits(&mut model, &tokens, None)?;
    let mut observed = Observation::default();
    observed
        .tensors
        .insert("prefill.full.logits".into(), observe(&logits)?);
    for layer in session.cache_view()?.layers {
        let key = format!("cache.after_prefill.layer{}", layer.layer);
        observed
            .tensors
            .insert(format!("{key}.keys"), observe(&layer.keys)?);
        observed
            .tensors
            .insert(format!("{key}.values"), observe(&layer.values)?);
        observed.caches.insert(
            key,
            CacheState {
                offset: layer.positions.end,
                retained: layer.positions,
            },
        );
    }
    let mut ids = Vec::new();
    for step in 0..8 {
        let token = argmax_axis(logits.index((0, -1, ..)), -1, None)?.contiguous()?;
        token.eval()?;
        let token = token.item_exact::<u32>();
        ids.push(token);
        if step < 7 {
            logits = session.decode_step(TokenId::from(token))?;
        }
    }
    observed.greedy_ids = Some(ids);
    Ok(observed)
}

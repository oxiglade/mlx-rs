//! Gemma 4 assistant (MTP drafter) loader: strip the `model.` wrapper onto the
//! drafter param walk; hydrate the `token_ordering` buffer (not a `#[param]`).

use std::collections::HashMap;
use std::path::Path;

use mlx_rs::module::ModuleParameters;
use mlx_rs::quantization::Quantizable;
use mlx_rs::transforms::eval_params;
use mlx_rs::Array;

use crate::error::Error;
use crate::gemma4::mtp::config::DrafterConfig;
use crate::gemma4::mtp::drafter::Drafter;
use crate::loader::{apply_post_load_memory_policy, list_shards, rewrite_quantised_keys};

/// The `masked_embedding.token_ordering` buffer key — a registered buffer in
/// the checkpoint, not a learned `#[param]`, so it's hydrated separately.
const TOKEN_ORDERING_KEY: &str = "masked_embedding.token_ordering";

/// Drop quantiser-stat / rope-freq keys the param walk does not expect.
fn should_drop(key: &str) -> bool {
    key.contains("self_attn.rotary_emb")
}

/// Strip the `model.` transformer wrapper; leave projection/centroid keys.
fn rewrite_key(key: &str) -> String {
    key.strip_prefix("model.")
        .map(str::to_string)
        .unwrap_or_else(|| key.to_string())
}

pub fn load_drafter(cfg: &DrafterConfig, dir: &Path) -> Result<Drafter, Error> {
    let mut drafter = Drafter::new(cfg)?;
    // Quantised assistant checkpoints (e.g. `*-assistant-8bit`) carry their
    // own quantisation; the drafter param walk then expects `…inner.weight`
    // keys, matching `rewrite_quantised_keys` output below.
    if let Some(q) = cfg.quantization.as_ref() {
        drafter = drafter.try_into_quantized(q.group_size, q.bits)?;
    }

    let shards = list_shards(dir)?;
    let mut raw: HashMap<String, Array> = HashMap::new();
    let mut token_ordering: Option<Array> = None;
    for path in shards {
        let loaded = Array::load_safetensors(&path).map_err(Error::LoadWeights)?;
        for (k, v) in loaded {
            if should_drop(&k) {
                continue;
            }
            let key = rewrite_key(&k);
            if key == TOKEN_ORDERING_KEY {
                token_ordering = Some(v);
                continue;
            }
            raw.insert(key, v);
        }
    }

    let weights = rewrite_quantised_keys(raw);
    let mut leftover: Vec<String> = Vec::new();
    bind(&mut drafter, weights, "drafter", &mut leftover);

    // Hydrate the token_ordering buffer onto the centroid head.
    if let Some(order) = token_ordering {
        match drafter.masked_embedding.as_mut() {
            Some(me) => me.token_ordering = order,
            None => leftover.push(format!(
                "{TOKEN_ORDERING_KEY} present but drafter has no centroid head"
            )),
        }
    } else if drafter.masked_embedding.is_some() {
        return Err(Error::config(
            "gemma4 drafter: centroid head enabled but token_ordering missing",
        ));
    }

    if !leftover.is_empty() {
        leftover.sort();
        return Err(Error::Other(
            format!(
                "gemma4 drafter loader: {} unbound key(s); first 8: {:?}",
                leftover.len(),
                &leftover.iter().take(8).collect::<Vec<_>>()
            )
            .into(),
        ));
    }

    eval_params(drafter.parameters()).map_err(Error::Exception)?;
    apply_post_load_memory_policy();
    Ok(drafter)
}

fn bind(
    drafter: &mut Drafter,
    weights: HashMap<String, Array>,
    prefix: &str,
    leftover: &mut Vec<String>,
) {
    let mut params = drafter.parameters_mut().flatten();
    for (k, v) in weights {
        if let Some(slot) = params.get_mut(&*k) {
            **slot = v;
        } else {
            leftover.push(format!("{prefix}.{k}"));
        }
    }
}

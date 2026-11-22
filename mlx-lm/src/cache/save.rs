use super::{
    concatenate, dimension, full::FullCache, invalid, recurrent::Recurrent, slice, validate, Cache,
    CacheCheckpoint, CacheError, CacheKind, LayerCache, LayerCacheSpec, TokenLedger,
};
use crate::{AttentionKind, ModelType, TokenId};
use mlx_rs::Array;
use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
    ops::Range,
    path::Path,
    rc::Rc,
};

const MODEL_TYPE: &str = "model_type";
const TOKENS: &str = "tokens";

fn rotating() -> CacheError {
    CacheError::UnsupportedPolicy("rotating layers cannot be saved".into())
}
fn save(
    architecture: &ModelType,
    tokens: &[TokenId],
    mut arrays: Vec<(String, Array)>,
    path: &Path,
) -> Result<(), CacheError> {
    let ids: Vec<u32> = tokens.iter().copied().map(u32::from).collect();
    arrays.push((
        TOKENS.to_owned(),
        Array::from_slice(&ids, &[dimension(ids.len())?]),
    ));
    let metadata = HashMap::from([(MODEL_TYPE.to_owned(), architecture.as_str().to_owned())]);
    Array::save_safetensors(arrays, &metadata, path)?;
    Ok(())
}
fn load(
    architecture: &ModelType,
    path: &Path,
) -> Result<(HashMap<String, Array>, Vec<TokenId>), CacheError> {
    // MLX has no GPU implementation of the safetensors Load primitive.
    let (mut arrays, metadata) = mlx_rs::with_stream(&mlx_rs::Stream::cpu(), || {
        Array::load_safetensors_with_metadata(path)
    })?;
    let tokens = arrays
        .remove(TOKENS)
        .ok_or(CacheError::FingerprintMismatch)?;
    if metadata.get(MODEL_TYPE).map(String::as_str) != Some(architecture.as_str()) {
        return Err(CacheError::FingerprintMismatch);
    }
    if tokens.ndim() != 1 {
        return Err(CacheError::FingerprintMismatch);
    }
    tokens.eval()?;
    let tokens = tokens
        .to_vec_exact::<u32>()
        .map_err(|_| CacheError::FingerprintMismatch)?;
    Ok((arrays, tokens.into_iter().map(TokenId::from).collect()))
}
fn evaluate<'a>(layers: impl Iterator<Item = &'a Box<dyn LayerCache>>) -> Result<(), CacheError> {
    let arrays = layers
        .map(|layer| layer.arrays().map(|(keys, values)| [keys, values]))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(mlx_rs::transforms::eval(arrays.iter().flatten())?)
}

impl CacheCheckpoint {
    /// Writes the arrays and tokens to a safetensors file that
    /// [`Model::load_checkpoint`](crate::Model::load_checkpoint) reads. The path must end
    /// in `.safetensors`. Rotating layers and a checkpoint without tokens cannot be saved.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), CacheError> {
        if self.tokens().is_empty() {
            return Err(invalid("a checkpoint without tokens cannot be saved"));
        }
        let mut arrays = Vec::new();
        for (i, layer) in &self.layers {
            if layer.info(*i).kind != CacheKind::Recurrent {
                return Err(rotating());
            }
            let (first, second) = layer.arrays()?;
            arrays.extend([(format!("{i}.0"), first), (format!("{i}.1"), second)]);
        }
        save(&self.architecture, self.tokens(), arrays, path.as_ref())
    }
    pub(crate) fn load(
        path: &Path,
        architecture: &ModelType,
        layout: &[LayerCacheSpec],
        model_identity: Rc<()>,
    ) -> Result<Self, CacheError> {
        let (mut arrays, tokens) = load(architecture, path)?;
        let mut layers = BTreeMap::new();
        for (i, spec) in layout.iter().enumerate() {
            if spec.attention != AttentionKind::Recurrent {
                continue;
            }
            let mut state = |part: u8| {
                arrays
                    .remove(&format!("{i}.{part}"))
                    .ok_or(CacheError::FingerprintMismatch)
            };
            let state = (state(0)?, state(1)?);
            let layer = Recurrent::from_state(spec, tokens.len(), state)?;
            layers.insert(i, Box::new(layer) as Box<dyn LayerCache>);
        }
        if !arrays.is_empty() {
            return Err(CacheError::FingerprintMismatch);
        }
        evaluate(layers.values())?;
        Ok(Self {
            architecture: architecture.clone(),
            model_identity,
            ledger: TokenLedger {
                ids: Rc::new(tokens),
            },
            layers,
        })
    }
}

impl Cache {
    /// Writes the full layers' K/V for `positions`, with the tokens up to their end, to a
    /// safetensors file that [`Model::load_cache`](crate::Model::load_cache) reads. The path
    /// must end in `.safetensors`. Rotating layers cannot be saved.
    pub fn save_positions(
        &self,
        positions: Range<usize>,
        path: impl AsRef<Path>,
    ) -> Result<(), CacheError> {
        self.validate_ledger()?;
        let tokens = self.ledger.tokens();
        if positions.is_empty() || positions.end > tokens.len() {
            return Err(invalid(
                "positions must be a nonempty range of cached tokens",
            ));
        }
        let mut arrays = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            match layer.info(i).kind {
                CacheKind::Recurrent => {}
                CacheKind::Rotating => return Err(rotating()),
                CacheKind::Full => {
                    let (keys, values) = layer.arrays()?;
                    arrays.extend([
                        (format!("{i}.keys"), slice(&keys, positions.clone())?),
                        (format!("{i}.values"), slice(&values, positions.clone())?),
                    ]);
                }
            }
        }
        save(
            &self.architecture,
            &tokens[..positions.end],
            arrays,
            path.as_ref(),
        )
    }
    pub(crate) fn load(
        checkpoint: &CacheCheckpoint,
        blocks: &[impl AsRef<Path>],
        layout: &[LayerCacheSpec],
        model_identity: &Rc<()>,
    ) -> Result<Self, CacheError> {
        if !Rc::ptr_eq(model_identity, &checkpoint.model_identity) {
            return Err(CacheError::FingerprintMismatch);
        }
        let tokens = checkpoint.tokens();
        let mut parts: BTreeMap<usize, [Vec<Array>; 2]> = BTreeMap::new();
        let mut end = 0;
        for path in blocks {
            let (mut arrays, prefix) = load(&checkpoint.architecture, path.as_ref())?;
            let follows = || {
                invalid("cache blocks do not follow one another through the checkpoint's tokens")
            };
            if prefix.len() <= end || !tokens.starts_with(&prefix) {
                return Err(follows());
            }
            for (i, spec) in layout.iter().enumerate() {
                if spec.attention == AttentionKind::Recurrent {
                    continue;
                }
                let mut part = |name| {
                    arrays
                        .remove(&format!("{i}.{name}"))
                        .ok_or(CacheError::FingerprintMismatch)
                };
                let (keys, values) = (part("keys")?, part("values")?);
                let length =
                    validate(spec, &keys, &values).map_err(|_| CacheError::FingerprintMismatch)?;
                if length != prefix.len() - end {
                    return Err(follows());
                }
                let [all_keys, all_values] = parts.entry(i).or_default();
                all_keys.push(keys);
                all_values.push(values);
            }
            if !arrays.is_empty() {
                return Err(CacheError::FingerprintMismatch);
            }
            end = prefix.len();
        }
        if end != tokens.len() {
            return Err(invalid("cache blocks do not reach the checkpoint's tokens"));
        }
        let layers = layout
            .iter()
            .enumerate()
            .map(|(i, spec)| {
                Ok(match (checkpoint.layers.get(&i), parts.remove(&i)) {
                    (Some(saved), None) => saved.clone_box(),
                    (None, Some([keys, values])) => Box::new(FullCache::from_arrays(
                        spec,
                        concatenate(&keys, 2)?,
                        concatenate(&values, 2)?,
                    )?),
                    _ => return Err(CacheError::FingerprintMismatch),
                })
            })
            .collect::<Result<Vec<_>, CacheError>>()?;
        evaluate(layers.iter())?;
        Ok(Self {
            architecture: checkpoint.architecture.clone(),
            model_identity: Rc::clone(model_identity),
            ledger: checkpoint.ledger.clone(),
            tracks_tokens: true,
            invalidated: false,
            last_submission: None,
            layers,
            _thread_bound: PhantomData,
        })
    }
}

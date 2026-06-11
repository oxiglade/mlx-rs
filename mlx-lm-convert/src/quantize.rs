//! Per-tensor quantisation. Wraps `mlx_rs::ops::quantize` and packages
//! the (packed_w, scales, biases) triple under their conventional key
//! suffixes (`.weight`, `.scales`, `.biases`).

use mlx_rs::{ops::quantize, Array};

use crate::{anyhow, plan::QuantClass, Result};

/// One entry to write into the output sharded safetensors.
pub struct OutTensor {
    pub key: String,
    pub array: Array,
}

/// Apply the [`QuantClass`] decision to a single (dst_key, tensor).
/// `Body` uses the user-chosen `body_bits`/`body_group_size`; `Pinned`
/// uses the rule-provided values; `Skip` returns the tensor as-is.
///
/// The destination key is expected to already end in `.weight` (the
/// post-rewrite convention). The split into `.weight` / `.scales` /
/// `.biases` is applied here.
pub fn classify_and_quantize(
    dst_key: String,
    tensor: Array,
    class: QuantClass,
    body_bits: i32,
    body_group_size: i32,
) -> Result<Vec<OutTensor>> {
    let (bits, group_size) = match class {
        QuantClass::Skip => {
            return Ok(vec![OutTensor {
                key: dst_key,
                array: tensor,
            }]);
        }
        QuantClass::Body => (body_bits, body_group_size),
        QuantClass::Pinned { group_size, bits } => (bits, group_size),
    };

    let stem = dst_key
        .strip_suffix(".weight")
        .ok_or_else(|| anyhow!("quantize: expected dst_key to end in .weight, got {dst_key:?}"))?;

    let (packed, scales, biases) = quantize(&tensor, group_size, bits)?;
    Ok(vec![
        OutTensor {
            key: format!("{stem}.weight"),
            array: packed,
        },
        OutTensor {
            key: format!("{stem}.scales"),
            array: scales,
        },
        OutTensor {
            key: format!("{stem}.biases"),
            array: biases,
        },
    ])
}

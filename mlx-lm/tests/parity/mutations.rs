use std::collections::{BTreeMap, BTreeSet};

use safetensors::Dtype;
use serde_json::json;

use super::{
    comparator::{compare, Class},
    observation::{CacheState, Expectations, Observation, Policy, Tensor, Tolerance},
};

fn tensor(shape: &[usize], values: &[f32]) -> Tensor {
    Tensor {
        shape: shape.to_vec(),
        dtype: Dtype::F32,
        bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
    }
}

fn synthetic() -> Expectations {
    let mut tensors = BTreeMap::from([
        (
            "prefill.full.logits".into(),
            tensor(&[1, 2, 3], &[0.25, 1.0, -2.0, 3.0, 0.5, -1.0]),
        ),
        ("quant.proj.scales".into(), tensor(&[2], &[0.25, 0.5])),
        ("quant.proj.biases".into(), tensor(&[2], &[-1.0, -2.0])),
    ]);
    let mut caches = BTreeMap::new();
    for layer in 0..2 {
        let key = format!("cache.after_prefill.layer{layer}");
        tensors.insert(
            format!("{key}.keys"),
            tensor(&[1, 1, 2, 1], &[layer as f32 + 1.0, 2.0]),
        );
        tensors.insert(
            format!("{key}.values"),
            tensor(&[1, 1, 2, 1], &[layer as f32 + 3.0, 4.0]),
        );
        caches.insert(
            key,
            CacheState {
                offset: 2,
                retained: 0..2,
            },
        );
    }
    for case in ["top_p", "top_k", "min_p"] {
        tensors.insert(
            format!("sampling.{case}.filtered_logprobs"),
            tensor(&[3], &[-0.1, -2.0, f32::NEG_INFINITY]),
        );
    }
    let policies = tensors
        .keys()
        .map(|key| {
            (
                key.clone(),
                if key.starts_with("quant.") {
                    Policy::ExactBits
                } else {
                    Policy::Float(Tolerance {
                        atol: 2e-4,
                        rtol: 2e-4,
                    })
                },
            )
        })
        .collect();
    Expectations {
        policies,
        observation: Observation {
            config: Some(json!({"hidden_size": 2, "layer_count": 2})),
            tokenizer: Some(
                json!({"encodings": {"prompt": [1, 2]}, "decodings": {"prompt": "a b"}, "eos_tokens": [0, 3], "eos_source": "generation_config"}),
            ),
            chat: Some(json!({"closed": {"rendered_utf8_hex": "612062", "token_ids": [1, 2]}})),
            tensors,
            caches,
            greedy_ids: Some(vec![1, 2]),
            sampled_ids: BTreeMap::from([
                ("top_p".into(), vec![1, 1]),
                ("top_k".into(), vec![1, 2]),
                ("min_p".into(), vec![2, 1]),
            ]),
            text_deltas: Some(vec!["a".into(), " b".into(), "!".into()]),
            finish: Some(("length".into(), None)),
            errors: BTreeMap::from([("missing_weight".into(), "WeightError::MissingKey".into())]),
        },
    }
}

fn perturb(observed: &mut Observation, key: &str) {
    observed.tensors.get_mut(key).unwrap().bytes[..4].copy_from_slice(&9.0f32.to_le_bytes());
}

fn swap(observed: &mut Observation, left: &str, right: &str) {
    let a = observed.tensors[left].clone();
    let b = observed.tensors[right].clone();
    observed.tensors.insert(left.into(), b);
    observed.tensors.insert(right.into(), a);
}

type Mutation = (&'static str, Class, fn(&mut Observation));

#[test]
fn every_mutation_fails_only_in_its_assigned_class() {
    let expected = synthetic();
    assert!(compare(&expected, &expected.observation).is_empty());
    let mutations: Vec<Mutation> = vec![
        ("resolved config", Class::Config, |o| {
            o.config.as_mut().unwrap()["hidden_size"] = json!(4)
        }),
        ("EOS-set removal", Class::Tokenizer, |o| {
            o.tokenizer.as_mut().unwrap()["eos_tokens"] = json!([0])
        }),
        ("chat whitespace", Class::Chat, |o| {
            o.chat.as_mut().unwrap()["closed"]["rendered_utf8_hex"] = json!("61202062")
        }),
        ("shape", Class::Shape, |o| {
            o.tensors.get_mut("prefill.full.logits").unwrap().shape = vec![1, 3, 2]
        }),
        ("equal-value dtype", Class::Dtype, |o| {
            let t = o.tensors.get_mut("prefill.full.logits").unwrap();
            t.bytes = t
                .floats()
                .unwrap()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            t.dtype = Dtype::F64;
        }),
        ("weight perturbation", Class::Value, |o| {
            perturb(o, "prefill.full.logits")
        }),
        ("one-position logit permutation", Class::Value, |o| {
            let t = o.tensors.get_mut("prefill.full.logits").unwrap();
            let (a, b) = t.bytes.split_at_mut(4);
            a.swap_with_slice(&mut b[..4]);
        }),
        ("K/V swap", Class::CacheValue, |o| {
            swap(
                o,
                "cache.after_prefill.layer0.keys",
                "cache.after_prefill.layer0.values",
            )
        }),
        ("cache layer reorder", Class::CacheValue, |o| {
            swap(
                o,
                "cache.after_prefill.layer0.keys",
                "cache.after_prefill.layer1.keys",
            );
            swap(
                o,
                "cache.after_prefill.layer0.values",
                "cache.after_prefill.layer1.values",
            );
        }),
        ("cache offset", Class::CacheOffset, |o| {
            o.caches
                .get_mut("cache.after_prefill.layer0")
                .unwrap()
                .offset += 1
        }),
        ("cache retained range", Class::CacheRange, |o| {
            o.caches
                .get_mut("cache.after_prefill.layer0")
                .unwrap()
                .retained = 1..3
        }),
        ("cache value beyond tolerance", Class::CacheValue, |o| {
            perturb(o, "cache.after_prefill.layer0.keys")
        }),
        ("greedy sampled ID replacement", Class::SampledId, |o| {
            o.greedy_ids.as_mut().unwrap()[0] = 0
        }),
        ("CPU sampled ID replacement", Class::SampledId, |o| {
            o.sampled_ids.get_mut("top_p").unwrap()[0] = 0
        }),
        ("top-p bypass", Class::SamplingSupport, |o| {
            o.tensors
                .get_mut("sampling.top_p.filtered_logprobs")
                .unwrap()
                .bytes[8..12]
                .copy_from_slice(&(-3.0f32).to_le_bytes())
        }),
        ("top-k bypass", Class::SamplingSupport, |o| {
            o.tensors
                .get_mut("sampling.top_k.filtered_logprobs")
                .unwrap()
                .bytes[8..12]
                .copy_from_slice(&(-3.0f32).to_le_bytes())
        }),
        ("min-p bypass", Class::SamplingSupport, |o| {
            o.tensors
                .get_mut("sampling.min_p.filtered_logprobs")
                .unwrap()
                .bytes[8..12]
                .copy_from_slice(&(-3.0f32).to_le_bytes())
        }),
        ("text-delta boundary shift", Class::TextDelta, |o| {
            o.text_deltas = Some(vec!["a ".into(), "b".into(), "!".into()])
        }),
        ("missing final flush text", Class::TextDelta, |o| {
            *o.text_deltas.as_mut().unwrap().last_mut().unwrap() = String::new()
        }),
        ("missing final flush event", Class::OutputCount, |o| {
            o.text_deltas.as_mut().unwrap().pop();
        }),
        ("finish-reason inversion", Class::FinishReason, |o| {
            o.finish.as_mut().unwrap().0 = "stop".into()
        }),
        ("stop-token replacement", Class::FinishReason, |o| {
            o.finish.as_mut().unwrap().1 = Some(3)
        }),
        ("error class", Class::Error, |o| {
            o.errors
                .insert("missing_weight".into(), "LoadError::Io".into());
        }),
        ("missing tensor", Class::OutputCount, |o| {
            o.tensors.remove("prefill.full.logits");
        }),
        ("affine scale/bias swap", Class::Value, |o| {
            swap(o, "quant.proj.scales", "quant.proj.biases")
        }),
    ];
    let mut covered = BTreeSet::new();
    for (name, class, mutate) in mutations {
        let mut actual = expected.observation.clone();
        mutate(&mut actual);
        let failures = compare(&expected, &actual);
        assert!(!failures.is_empty(), "{name} escaped detection");
        assert_eq!(
            failures.iter().map(|f| f.class).collect::<BTreeSet<_>>(),
            BTreeSet::from([class]),
            "{name}: {failures:?}"
        );
        covered.insert(class.as_str());
    }
    assert_eq!(
        covered,
        BTreeSet::from([
            "config",
            "tokenizer",
            "chat",
            "shape",
            "dtype",
            "value",
            "cache_offset",
            "cache_range",
            "cache_value",
            "sampling_support",
            "sampled_id",
            "text_delta",
            "finish_reason",
            "error_class",
            "output_count"
        ])
    );
}

#[test]
fn tolerance_and_exact_bit_policies_are_distinct() {
    let mut expected = synthetic();
    let mut actual = expected.observation.clone();
    actual.tensors.get_mut("prefill.full.logits").unwrap().bytes[..4]
        .copy_from_slice(&0.2501f32.to_le_bytes());
    assert!(compare(&expected, &actual).is_empty());
    expected
        .policies
        .insert("prefill.full.logits".into(), Policy::ExactBits);
    assert_eq!(compare(&expected, &actual)[0].class, Class::Value);
    let mut actual = expected.observation.clone();
    actual.tensors.get_mut("quant.proj.scales").unwrap().bytes[..4]
        .copy_from_slice(&0.25000003f32.to_le_bytes());
    assert_eq!(compare(&expected, &actual)[0].class, Class::Value);
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut actual = expected.observation.clone();
        actual
            .tensors
            .get_mut("cache.after_prefill.layer0.keys")
            .unwrap()
            .bytes[..4]
            .copy_from_slice(&value.to_le_bytes());
        assert_eq!(compare(&expected, &actual)[0].class, Class::CacheValue);
    }
}

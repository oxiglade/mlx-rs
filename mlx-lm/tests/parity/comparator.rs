use std::collections::BTreeMap;

use super::observation::{Expectations, Observation, Policy, Tensor};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Class {
    Config,
    Tokenizer,
    Chat,
    Shape,
    Dtype,
    Value,
    CacheOffset,
    CacheRange,
    CacheValue,
    SamplingSupport,
    SampledId,
    TextDelta,
    FinishReason,
    Error,
    OutputCount,
    Processor,
    TextStop,
    Progress,
    CacheTrim,
}

impl Class {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Config => "config",
            Self::Tokenizer => "tokenizer",
            Self::Chat => "chat",
            Self::Shape => "shape",
            Self::Dtype => "dtype",
            Self::Value => "value",
            Self::CacheOffset => "cache_offset",
            Self::CacheRange => "cache_range",
            Self::CacheValue => "cache_value",
            Self::SamplingSupport => "sampling_support",
            Self::SampledId => "sampled_id",
            Self::TextDelta => "text_delta",
            Self::FinishReason => "finish_reason",
            Self::Error => "error_class",
            Self::OutputCount => "output_count",
            Self::Processor => "processor",
            Self::TextStop => "text_stop",
            Self::Progress => "progress",
            Self::CacheTrim => "cache_trim",
        }
    }
}

#[derive(Debug)]
pub struct Failure {
    pub class: Class,
    pub surface: String,
    pub detail: String,
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: {}: {}",
            self.class.as_str(),
            self.surface,
            self.detail
        )
    }
}

fn fail(failures: &mut Vec<Failure>, class: Class, surface: &str, detail: &str) {
    failures.push(Failure {
        class,
        surface: surface.into(),
        detail: detail.into(),
    });
}

fn equal<T: PartialEq>(
    failures: &mut Vec<Failure>,
    class: Class,
    surface: &str,
    expected: &T,
    actual: &T,
) {
    if expected != actual {
        fail(failures, class, surface, "does not match expectation");
    }
}

fn keys<E, A>(
    failures: &mut Vec<Failure>,
    surface: &str,
    expected: &BTreeMap<String, E>,
    actual: &BTreeMap<String, A>,
) {
    if !expected.keys().eq(actual.keys()) {
        fail(failures, Class::OutputCount, surface, "output keys differ");
    }
}

fn sequence<T: PartialEq>(
    failures: &mut Vec<Failure>,
    class: Class,
    surface: &str,
    expected: &Option<Vec<T>>,
    actual: &Option<Vec<T>>,
) {
    match (expected, actual) {
        (Some(e), Some(a)) if e.len() == a.len() => equal(failures, class, surface, e, a),
        (None, None) => {}
        _ => fail(
            failures,
            Class::OutputCount,
            surface,
            "event count or presence differs",
        ),
    }
}

pub fn compare(expected: &Expectations, actual: &Observation) -> Vec<Failure> {
    let e = &expected.observation;
    let mut failures = Vec::new();
    equal(
        &mut failures,
        Class::Config,
        "config",
        &e.config,
        &actual.config,
    );
    equal(
        &mut failures,
        Class::Tokenizer,
        "tokenizer",
        &e.tokenizer,
        &actual.tokenizer,
    );
    equal(&mut failures, Class::Chat, "chat", &e.chat, &actual.chat);
    equal(
        &mut failures,
        Class::Processor,
        "processing",
        &e.processing,
        &actual.processing,
    );
    equal(
        &mut failures,
        Class::Processor,
        "processor histories",
        &e.processor_histories,
        &actual.processor_histories,
    );
    equal(
        &mut failures,
        Class::TextStop,
        "text cases",
        &e.text_cases,
        &actual.text_cases,
    );
    equal(
        &mut failures,
        Class::Progress,
        "prefill.progress",
        &e.progress,
        &actual.progress,
    );
    equal(
        &mut failures,
        Class::CacheTrim,
        "cache.trim_after_wrap",
        &e.trim_after_wrap,
        &actual.trim_after_wrap,
    );
    sequence(
        &mut failures,
        Class::SampledId,
        "decode.greedy_ids",
        &e.greedy_ids,
        &actual.greedy_ids,
    );
    sequence(
        &mut failures,
        Class::TextDelta,
        "decode.text_deltas",
        &e.text_deltas,
        &actual.text_deltas,
    );
    equal(
        &mut failures,
        Class::FinishReason,
        "decode.finish",
        &e.finish,
        &actual.finish,
    );
    keys(
        &mut failures,
        "sampling",
        &e.sampled_ids,
        &actual.sampled_ids,
    );
    for (key, ids) in &e.sampled_ids {
        if let Some(actual) = actual.sampled_ids.get(key) {
            if ids.len() != actual.len() {
                fail(
                    &mut failures,
                    Class::OutputCount,
                    key,
                    "sample count differs",
                );
            } else {
                equal(&mut failures, Class::SampledId, key, ids, actual);
            }
        }
    }
    keys(&mut failures, "errors", &e.errors, &actual.errors);
    for (key, error) in &e.errors {
        if let Some(actual) = actual.errors.get(key) {
            equal(&mut failures, Class::Error, key, error, actual);
        }
    }
    keys(&mut failures, "cache states", &e.caches, &actual.caches);
    for (key, state) in &e.caches {
        if let Some(actual) = actual.caches.get(key) {
            equal(
                &mut failures,
                Class::CacheOffset,
                key,
                &state.offset,
                &actual.offset,
            );
            equal(
                &mut failures,
                Class::CacheRange,
                key,
                &state.retained,
                &actual.retained,
            );
        }
    }
    keys(&mut failures, "tensors", &e.tensors, &actual.tensors);
    for (key, tensor) in &e.tensors {
        if let Some(actual) = actual.tensors.get(key) {
            compare_tensor(
                &mut failures,
                key,
                tensor,
                actual,
                expected.policies.get(key),
            );
        }
    }
    failures
}

fn compare_tensor(
    failures: &mut Vec<Failure>,
    key: &str,
    e: &Tensor,
    a: &Tensor,
    policy: Option<&Policy>,
) {
    if e.shape != a.shape || !e.valid_layout() || !a.valid_layout() {
        fail(
            failures,
            Class::Shape,
            key,
            "shape or payload length differs",
        );
        return;
    }
    if e.dtype != a.dtype {
        fail(failures, Class::Dtype, key, "dtype differs");
        return;
    }
    let class = if key.starts_with("processing.") {
        Class::Processor
    } else if key.starts_with("cache.trim_after_wrap.") {
        Class::CacheTrim
    } else if key.starts_with("cache.") {
        Class::CacheValue
    } else {
        Class::Value
    };
    let Some(policy) = policy else {
        fail(failures, Class::Config, key, "missing tolerance policy");
        return;
    };
    match policy {
        Policy::ExactBits => equal(failures, class, key, &e.bytes, &a.bytes),
        Policy::Float(tolerance) => {
            let (Ok(e), Ok(a)) = (e.floats(), a.floats()) else {
                fail(
                    failures,
                    Class::Dtype,
                    key,
                    "float policy requires floating dtype",
                );
                return;
            };
            if key.starts_with("sampling.")
                && e.iter()
                    .zip(&a)
                    .any(|(e, a)| (*e == f64::NEG_INFINITY) != (*a == f64::NEG_INFINITY))
            {
                fail(
                    failures,
                    Class::SamplingSupport,
                    key,
                    "filtered support differs",
                );
                return;
            }
            for (index, (e, a)) in e.iter().zip(&a).enumerate() {
                let matches = if e.is_nan() || a.is_nan() {
                    false
                } else if e.is_infinite() || a.is_infinite() {
                    e == a
                } else if tolerance.atol == 0.0 && tolerance.rtol == 0.0 {
                    e.to_bits() == a.to_bits()
                } else {
                    (e - a).abs() <= tolerance.atol + tolerance.rtol * e.abs()
                };
                if !matches {
                    fail(
                        failures,
                        class,
                        key,
                        &format!("element {index}: expected {e}, got {a}"),
                    );
                    break;
                }
            }
        }
    }
}

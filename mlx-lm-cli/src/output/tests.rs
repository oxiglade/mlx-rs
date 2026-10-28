use super::*;
use std::{collections::VecDeque, io};

#[derive(Default)]
struct Writer {
    bytes: Vec<u8>,
    writes: VecDeque<Result<usize, io::ErrorKind>>,
    flushes: VecDeque<Result<(), io::ErrorKind>>,
    flush_count: usize,
}
impl io::Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let count = self
            .writes
            .pop_front()
            .unwrap_or(Ok(bytes.len()))
            .map_err(io::Error::from)?;
        let count = count.min(bytes.len());
        self.bytes.extend_from_slice(&bytes[..count]);
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.flush_count += 1;
        self.flushes
            .pop_front()
            .unwrap_or(Ok(()))
            .map_err(io::Error::from)
    }
}
fn token(text: &str, finish_reason: Option<OutputFinish>) -> OutputEvent {
    OutputEvent::Token {
        token_id: 13,
        text: text.into(),
        finish_reason,
    }
}

#[test]
fn jsonl_exact_schema_order_and_empty_final_delta() {
    let mut output = Writer::default();
    for event in [
        OutputEvent::Prefill {
            processed: 0,
            total: 9,
        },
        token("雪\n\"", None),
        token("", Some(OutputFinish::Length)),
        token("", Some(OutputFinish::Stop)),
    ] {
        write_event(&mut output, &event, GenerationFormat::Jsonl).unwrap();
    }
    assert_eq!(String::from_utf8(output.bytes).unwrap(), concat!(
        "{\"version\":1,\"event\":\"prefill\",\"processed\":0,\"total\":9}\n",
        "{\"version\":1,\"event\":\"token\",\"token_id\":13,\"text\":\"雪\\n\\\"\",\"finish_reason\":null}\n",
        "{\"version\":1,\"event\":\"token\",\"token_id\":13,\"text\":\"\",\"finish_reason\":\"length\"}\n",
        "{\"version\":1,\"event\":\"token\",\"token_id\":13,\"text\":\"\",\"finish_reason\":\"stop\"}\n",
    ));
    assert_eq!(output.flush_count, 4);
}

#[test]
fn text_is_exact_deltas_with_one_flush_each() {
    let mut output = Writer::default();
    write_event(
        &mut output,
        &OutputEvent::Prefill {
            processed: 0,
            total: 9,
        },
        GenerationFormat::Text,
    )
    .unwrap();
    for text in [" hello", "雪\n", ""] {
        write_event(&mut output, &token(text, None), GenerationFormat::Text).unwrap();
    }
    assert_eq!(output.bytes, " hello雪\n".as_bytes());
    assert_eq!(output.flush_count, 3);
}

#[test]
fn writers_retry_interrupted_and_short_writes() {
    for format in [GenerationFormat::Text, GenerationFormat::Jsonl] {
        let event = token(" hello雪", Some(OutputFinish::Length));
        let mut expected = Vec::new();
        write_event(&mut expected, &event, format).unwrap();
        let mut output = Writer {
            writes: [
                Err(io::ErrorKind::Interrupted),
                Ok(1),
                Ok(2),
                Err(io::ErrorKind::Interrupted),
                Ok(1),
            ]
            .into(),
            flushes: [Err(io::ErrorKind::Interrupted), Ok(())].into(),
            ..Writer::default()
        };
        write_event(&mut output, &event, format).unwrap();
        assert_eq!(output.bytes, expected);
        assert_eq!(output.flush_count, 2);
    }
}

#[test]
fn writers_preserve_write_and_flush_error_kinds() {
    for format in [GenerationFormat::Text, GenerationFormat::Jsonl] {
        for kind in [io::ErrorKind::Other, io::ErrorKind::BrokenPipe] {
            for during_flush in [false, true] {
                let mut output = Writer::default();
                if during_flush {
                    output.flushes.push_back(Err(kind));
                } else {
                    output.writes.extend([Ok(1), Err(kind)]);
                }
                let error = write_event(&mut output, &token("hello", None), format).unwrap_err();
                assert_eq!(error.is_broken_pipe(), kind == io::ErrorKind::BrokenPipe);
                assert!(matches!(error, CliError::Output(error) if error.kind() == kind));
            }
        }
    }
}

#[test]
fn zero_write_is_output_failure() {
    let mut output = Writer {
        writes: [Ok(0)].into(),
        ..Writer::default()
    };
    assert!(
        matches!(write_event(&mut output, &token("x", None), GenerationFormat::Text), Err(CliError::Output(error)) if error.kind() == io::ErrorKind::WriteZero)
    );
}

#[test]
fn config_encoding_covers_public_variants_and_sorts_paths() {
    use mlx_lm::*;
    use std::{collections::BTreeMap, num::NonZeroUsize};
    let affine = AffineQuantization {
        group_size: NonZeroUsize::new(32).unwrap(),
        bits: 4,
    };
    let config = Config {
        model_type: ModelType::new("qwen3"),
        dimensions: TransformerDimensions {
            hidden_size: 32,
            layer_count: 2,
            intermediate_size: 64,
            attention_heads: 4,
            kv_heads: 2,
            head_dim: 8,
            vocabulary_size: 64,
            max_positions: Some(128),
            rms_norm_epsilon: 0.00001,
        },
        rope: RopeConfig {
            dimensions: 8,
            theta: 10000.0,
            traditional: false,
            scaling: RopeScaling::Llama3 {
                factor: 8.0,
                low_frequency_factor: 1.0,
                high_frequency_factor: 4.0,
                original_max_positions: 128,
            },
        },
        attention: vec![
            AttentionKind::Full,
            AttentionKind::Sliding {
                window: NonZeroUsize::new(8).unwrap(),
            },
        ],
        tie_word_embeddings: true,
        attention_bias: false,
        mlp_bias: false,
        quantization: Some(QuantizationConfig {
            default: affine.clone(),
            layers: BTreeMap::from([
                (
                    ParameterPath::new("z.weight"),
                    LayerQuantization::Affine(affine),
                ),
                (
                    ParameterPath::new("a.weight"),
                    LayerQuantization::Unquantized,
                ),
            ]),
        }),
    };
    let value = config_value(&config);
    assert_eq!(value["model_type"], "qwen3");
    assert_eq!(value["dimensions"].as_object().unwrap().len(), 9);
    assert_eq!(
        value["attention"],
        serde_json::json!([{"kind":"full"},{"kind":"sliding","window":8}])
    );
    assert_eq!(
        value["rope"]["scaling"],
        serde_json::json!({"kind":"llama3", "factor":8.0,"low_frequency_factor":1.0,"high_frequency_factor":4.0,"original_max_positions":128})
    );
    assert_eq!(
        value["quantization"]["default"],
        serde_json::json!({"group_size":32,"bits":4})
    );
    assert_eq!(
        value["quantization"]["layers"]["a.weight"],
        serde_json::json!({"kind":"unquantized"})
    );
    assert_eq!(
        value["quantization"]["layers"]["z.weight"],
        serde_json::json!({"kind":"affine","group_size":32,"bits":4})
    );
    let encoded = serde_json::to_string(&value).unwrap();
    assert!(encoded.find("a.weight").unwrap() < encoded.find("z.weight").unwrap());
    let mut config = config;
    config.quantization = None;
    config.rope.scaling = RopeScaling::None;
    assert_eq!(
        config_value(&config)["rope"]["scaling"],
        serde_json::json!({"kind":"none"})
    );
    assert!(config_value(&config)["quantization"].is_null());
    config.rope.scaling = RopeScaling::Linear { factor: 2.0 };
    assert_eq!(
        config_value(&config)["rope"]["scaling"],
        serde_json::json!({"kind":"linear","factor":2.0})
    );
}

#[test]
fn info_root_schema_and_writer_errors() {
    let info = InfoRecord {
        version: 1,
        config: serde_json::json!({}),
        tokenizer: TokenizerRecord {
            bos_token: Some("<s>"),
            eos_tokens: vec![2, 3],
        },
        hub_provenance: None,
    };
    let mut output = Vec::new();
    write_info(&mut output, &info, InfoFormat::Json).unwrap();
    assert_eq!(String::from_utf8(output).unwrap(), "{\"version\":1,\"config\":{},\"tokenizer\":{\"bos_token\":\"<s>\",\"eos_tokens\":[2,3]},\"hub_provenance\":null}\n");
    for format in [InfoFormat::Text, InfoFormat::Json] {
        let mut output = Writer {
            writes: [Err(io::ErrorKind::BrokenPipe)].into(),
            ..Writer::default()
        };
        assert!(write_info(&mut output, &info, format)
            .unwrap_err()
            .is_broken_pipe());
    }
}

#[test]
fn info_writers_retry_and_preserve_io_failures() {
    let info = InfoRecord {
        version: 1,
        config: serde_json::json!({"model_type":"llama"}),
        tokenizer: TokenizerRecord {
            bos_token: None,
            eos_tokens: vec![2],
        },
        hub_provenance: None,
    };
    for format in [InfoFormat::Text, InfoFormat::Json] {
        let mut expected = Vec::new();
        write_info(&mut expected, &info, format).unwrap();
        let mut output = Writer {
            writes: [Ok(1), Err(io::ErrorKind::Interrupted), Ok(2)].into(),
            flushes: [Err(io::ErrorKind::Interrupted), Ok(())].into(),
            ..Writer::default()
        };
        write_info(&mut output, &info, format).unwrap();
        assert_eq!(output.bytes, expected);
        for kind in [io::ErrorKind::Other, io::ErrorKind::BrokenPipe] {
            for during_flush in [false, true] {
                let mut output = Writer::default();
                if during_flush {
                    output.flushes.push_back(Err(kind));
                } else {
                    output.writes.extend([Ok(1), Err(kind)]);
                }
                let error = write_info(&mut output, &info, format).unwrap_err();
                assert_eq!(error.is_broken_pipe(), kind == io::ErrorKind::BrokenPipe);
                assert!(matches!(error, CliError::Output(error) if error.kind() == kind));
            }
        }
    }
}

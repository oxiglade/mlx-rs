use super::*;
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct TextFixture {
    schema_version: u32,
    decoders: BTreeMap<String, DecoderCase>,
}

#[derive(Deserialize)]
struct DecoderCase {
    tokenizer: Value,
    token_ids: Vec<u32>,
    decoded_utf8_hex: String,
    emitted_prefix: String,
    preserves_prefix: bool,
    finish_flush: Option<String>,
    expected_error: Option<String>,
}

fn text_fixture() -> Result<TextFixture, TokenizerError> {
    Ok(serde_json::from_str(include_str!(
        "../../../conformance/mlx-lm/fixtures/llama-base/text_cases.json"
    ))?)
}

#[test]
fn oracle_decoder_cases() -> Result<(), TokenizerError> {
    let fixture = text_fixture()?;
    assert_eq!(fixture.schema_version, 1);
    assert_eq!(fixture.decoders.len(), 5);
    for (name, case) in fixture.decoders {
        let tokenizer = Tokenizer::from_bytes(&serde_json::to_vec(&case.tokenizer)?)?;
        let ids: Vec<_> = case.token_ids.iter().copied().map(TokenId::from).collect();
        let decoded = tokenizer.decode(&ids)?;
        let hex: String = decoded
            .as_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(hex, case.decoded_utf8_hex, "{name}: final decode");
        let expected_deltas: &[Option<&str>] = match name.as_str() {
            "byte_fallback_complete" => &[None, Some("é"), Some(" ")],
            "byte_fallback_empty" => &[],
            "byte_fallback_incomplete" => &[None],
            "byte_fallback_rewrite" | "byte_level_incomplete_tail" => &[None, Some("é"), None],
            _ => panic!("unqualified decoder trace: {name}"),
        };
        assert_eq!(ids.len(), expected_deltas.len(), "{name}");
        let mut stream = tokenizer.decode_stream();
        let mut emitted = String::new();
        for (index, id) in ids.into_iter().enumerate() {
            let delta = stream.step(id)?;
            assert_eq!(delta.as_deref(), expected_deltas[index], "{name}/{index}");
            if let Some(delta) = delta {
                emitted.push_str(&delta);
            }
        }
        assert_eq!(emitted, case.emitted_prefix, "{name}: emitted prefix");
        assert_eq!(
            decoded.starts_with(&emitted),
            case.preserves_prefix,
            "{name}"
        );
        if let Some(expected_error) = case.expected_error {
            assert_eq!(expected_error, "DecodeStreamError::InvalidPrefix", "{name}");
            assert!(case.finish_flush.is_none(), "{name}");
            let error = stream.finish().expect_err("rewritten prefix must fail");
            let TokenizerError::Tokenizer(source) = error else {
                panic!("{name}: wrong error boundary: {error}");
            };
            let Some(tokenizers::tokenizer::DecodeStreamError::InvalidPrefix {
                token_id,
                expected_prefix,
                actual_string,
            }) = source.downcast_ref::<tokenizers::tokenizer::DecodeStreamError>()
            else {
                panic!("{name}: lost InvalidPrefix: {source}");
            };
            assert_eq!(Some(token_id), case.token_ids.last(), "{name}");
            assert_eq!(expected_prefix, &emitted, "{name}");
            assert_eq!(actual_string, &decoded, "{name}");
        } else {
            let flush = stream.finish()?;
            assert_eq!(Some(&flush), case.finish_flush.as_ref(), "{name}");
            emitted.push_str(&flush);
            assert_eq!(emitted, decoded, "{name}: complete stream");
        }
    }
    Ok(())
}

#[test]
fn decoder_flush_passes_through_stop_filter() -> Result<(), Box<dyn std::error::Error>> {
    let fixture = text_fixture()?;
    for (name, stop, expected_text, expected_stop) in [
        ("byte_level_incomplete_tail", "é�", "", true),
        ("byte_level_incomplete_tail", "é!", "é�", false),
        ("byte_fallback_incomplete", "�", "", true),
    ] {
        let case = &fixture.decoders[name];
        let tokenizer = Tokenizer::from_bytes(&serde_json::to_vec(&case.tokenizer)?)?;
        let mut stream = tokenizer.decode_stream();
        let mut filter = crate::stop::StopStringFilter::new(vec![stop.into()])?;
        for &id in &case.token_ids {
            if let Some(delta) = stream.step(TokenId::from(id))? {
                assert_eq!(filter.process(&delta), (String::new(), false), "{name}");
            }
        }
        assert_eq!(
            filter.finish(&stream.finish()?),
            (expected_text.into(), expected_stop),
            "{name}/{stop}"
        );
    }
    Ok(())
}

use super::*;
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Deserialize)]
struct Expectations {
    tokenizer: TokenizerExpectations,
    decode: DecodeExpectations,
    chat: BTreeMap<String, ChatCase>,
}

#[derive(Deserialize)]
struct DecodeExpectations {
    token_ids: Vec<u32>,
    text_deltas: Vec<String>,
}

#[derive(Deserialize)]
struct TokenizerExpectations {
    encodings: BTreeMap<String, Vec<u32>>,
    decodings: BTreeMap<String, String>,
    eos_tokens: Vec<u32>,
    eos_source: String,
}

#[derive(Deserialize)]
struct ChatCase {
    messages: Vec<WireMessage>,
    continuation: String,
    rendered_utf8_hex: String,
    token_ids: Vec<u32>,
    template_options: ThinkingOptions,
}

#[derive(Deserialize)]
struct ThinkingOptions {
    enable_thinking: Option<bool>,
}

#[derive(Deserialize)]
struct WireMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct Inputs {
    prompts: BTreeMap<String, String>,
    token_ids: BTreeMap<String, Vec<u32>>,
}

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures")
}

fn ids(values: &[u32]) -> Vec<TokenId> {
    values.iter().copied().map(TokenId::from).collect()
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, TokenizerError> {
    Ok(serde_json::from_slice(&std::fs::read(path)?)?)
}

#[test]
fn every_oracle_tokenizer_and_chat_case() -> Result<(), TokenizerError> {
    let mut fixtures = std::fs::read_dir(fixture_root())?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<Result<Vec<_>, _>>()?;
    fixtures.sort();
    assert!(!fixtures.is_empty());
    for directory in fixtures {
        if !directory.is_dir() {
            continue;
        }
        let expected: Expectations = read_json(&directory.join("expectations.json"))?;
        let inputs: Inputs = read_json(&directory.join("inputs.json"))?;
        let tokenizer = Tokenizer::from_dir(&directory)?;
        let file = Tokenizer::from_file(directory.join("tokenizer.json"))?;
        let bytes = Tokenizer::from_bytes(&std::fs::read(directory.join("tokenizer.json"))?)?;
        for (name, encoding) in &expected.tokenizer.encodings {
            let prompt_name = if name == "special_with_defaults" {
                "special"
            } else {
                name
            };
            let prompt = inputs.prompts.get(prompt_name).ok_or_else(|| {
                TokenizerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("missing fixture prompt {name}"),
                ))
            })?;
            for tokenizer in [&tokenizer, &file, &bytes] {
                let actual = if name == "special_with_defaults" {
                    tokenizer.encode_with_special_tokens(prompt, true)?
                } else {
                    tokenizer.encode(prompt)?
                };
                assert_eq!(actual, ids(encoding), "{directory:?}/{name}");
            }
        }
        for (name, decoding) in &expected.tokenizer.decodings {
            let input_name = if name == "special_skipped" {
                "special"
            } else {
                name
            };
            let tokens = inputs.token_ids.get(input_name).ok_or_else(|| {
                TokenizerError::Io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("missing fixture IDs {name}"),
                ))
            })?;
            let actual = if name == "special_skipped" {
                tokenizer.decode_with_special_tokens(&ids(tokens), true)?
            } else {
                tokenizer.decode(&ids(tokens))?
            };
            assert_eq!(&actual, decoding, "{directory:?}/{name}");
        }
        assert_eq!(
            tokenizer.eos_tokens(),
            ids(&expected.tokenizer.eos_tokens),
            "{directory:?}"
        );
        let (resolved, source) = tokenizer.resolve_eos(&directory)?;
        assert_eq!(resolved, tokenizer.eos_tokens());
        assert_eq!(source, expected.tokenizer.eos_source, "{directory:?}");
        for (name, case) in expected.chat {
            let messages: Vec<_> = case
                .messages
                .into_iter()
                .map(|message| Message {
                    role: match message.role.as_str() {
                        "system" => Role::System,
                        "user" => Role::User,
                        "assistant" => Role::Assistant,
                        "tool" => Role::Tool,
                        _ => Role::Other(message.role),
                    },
                    content: message.content,
                })
                .collect();
            let continuation = match case.continuation.as_str() {
                "closed" => ChatContinuation::Closed,
                "start_assistant" => ChatContinuation::StartAssistant,
                "continue_last" => ChatContinuation::ContinueLast,
                _ => return Err(ChatTemplateError::IncompatibleContinuation.into()),
            };
            let rendered = tokenizer.render_chat(
                &messages,
                ChatTemplateOptions {
                    continuation,
                    enable_thinking: case.template_options.enable_thinking,
                },
            )?;
            let hex: String = rendered
                .as_bytes()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect();
            assert_eq!(hex, case.rendered_utf8_hex, "{directory:?}/{name}");
            assert_eq!(
                tokenizer.encode(&rendered)?,
                ids(&case.token_ids),
                "{directory:?}/{name}"
            );
        }
        assert_eq!(
            expected.decode.token_ids.len(),
            expected.decode.text_deltas.len(),
            "{directory:?}/decode"
        );
        let mut stream = tokenizer.decode_stream();
        for (i, id) in expected.decode.token_ids.iter().enumerate() {
            assert_eq!(
                stream.step(TokenId::from(*id))?,
                Some(expected.decode.text_deltas[i].clone()),
                "{directory:?}/decode step {i}"
            );
        }
        assert_eq!(stream.finish()?, "", "{directory:?}/decode");
        for tokens in inputs.token_ids.values() {
            let mut stream = tokenizer.decode_stream();
            let mut text = String::new();
            for id in ids(tokens) {
                if let Some(delta) = stream.step(id)? {
                    text.push_str(&delta);
                }
            }
            text.push_str(&stream.finish()?);
            assert_eq!(text, tokenizer.decode(&ids(tokens))?);
        }
    }
    Ok(())
}

fn assets(config: Value) -> Result<tempfile::TempDir, TokenizerError> {
    let directory = tempfile::tempdir()?;
    std::fs::copy(
        fixture_root().join("llama-base/tokenizer.json"),
        directory.path().join("tokenizer.json"),
    )?;
    write_json(&directory.path().join("tokenizer_config.json"), config)?;
    Ok(directory)
}

fn write_json(path: &Path, value: Value) -> Result<(), TokenizerError> {
    Ok(std::fs::write(path, serde_json::to_vec(&value)?)?)
}

#[test]
fn eos_precedence_and_loader_override() -> Result<(), TokenizerError> {
    let directory = assets(json!({"eos_token": {"content": "<|eot_id|>"}}))?;
    let path = directory.path();
    assert_eq!(Tokenizer::from_dir(path)?.eos_tokens(), ids(&[3]));
    write_json(&path.join("config.json"), json!({"eos_token_id": 2}))?;
    assert_eq!(
        Tokenizer::from_dir(path)?.resolve_eos(path)?,
        (ids(&[2]), "config")
    );
    for value in [json!(null), json!([]), json!(0)] {
        write_json(
            &path.join("generation_config.json"),
            json!({"eos_token_id": value}),
        )?;
        assert_eq!(Tokenizer::from_dir(path)?.eos_tokens(), ids(&[2]));
    }
    for value in [json!(8), json!([9, 8, 9])] {
        write_json(
            &path.join("generation_config.json"),
            json!({"eos_token_id": value}),
        )?;
        assert_eq!(Tokenizer::from_dir(path)?.eos_tokens(), parse_eos(&value)?);
    }
    assert_eq!(
        Tokenizer::from_dir_with_eos(path, ids(&[3, 2, 3]))?.eos_tokens(),
        ids(&[2, 3])
    );
    assert!(Tokenizer::from_dir_with_eos(path, Vec::new())?
        .eos_tokens()
        .is_empty());
    std::fs::write(path.join("generation_config.json"), b"invalid")?;
    std::fs::write(path.join("config.json"), b"invalid")?;
    assert_eq!(
        Tokenizer::from_dir_with_eos(path, ids(&[8]))?.eos_tokens(),
        ids(&[8])
    );
    std::fs::remove_file(path.join("generation_config.json"))?;
    write_json(&path.join("config.json"), json!({"eos_token_id": []}))?;
    assert!(Tokenizer::from_dir(path)?.eos_tokens().is_empty());
    write_json(&path.join("config.json"), json!({"eos_token_id": null}))?;
    assert_eq!(
        Tokenizer::from_dir(path)?.resolve_eos(path)?,
        (ids(&[3]), "tokenizer")
    );
    Ok(())
}

#[test]
fn invalid_eos_and_asset_errors_are_typed() -> Result<(), TokenizerError> {
    let directory = assets(json!({"eos_token": "<|eot_id|>"}))?;
    let path = directory.path();
    for value in [
        json!(-1),
        json!(4294967296_u64),
        json!(1.5),
        json!("2"),
        json!(true),
        json!({}),
        json!([2, null]),
    ] {
        for file in ["config.json", "generation_config.json"] {
            write_json(&path.join(file), json!({"eos_token_id": value}))?;
            assert!(matches!(
                Tokenizer::from_dir(path),
                Err(TokenizerError::InvalidEos(_))
            ));
            std::fs::remove_file(path.join(file))?;
        }
    }
    write_json(
        &path.join("tokenizer_config.json"),
        json!({"eos_token": "not in vocabulary"}),
    )?;
    assert!(matches!(
        Tokenizer::from_dir(path),
        Err(TokenizerError::InvalidEos(_))
    ));
    assert!(matches!(
        Tokenizer::from_file(path.join("absent")),
        Err(TokenizerError::MissingFile(_))
    ));
    assert!(matches!(
        Tokenizer::from_bytes(b"bad JSON"),
        Err(TokenizerError::Tokenizer(_))
    ));
    std::fs::write(path.join("tokenizer_config.json"), b"bad JSON")?;
    assert!(matches!(
        Tokenizer::from_dir(path),
        Err(TokenizerError::Json(_))
    ));
    assert!(Tokenizer::from_dir_with_eos(path, ids(&[2])).is_err());
    Ok(())
}

#[test]
fn selected_template_and_special_token_context() -> Result<(), TokenizerError> {
    let template = "{{ bos_token }}{{ additional_special_tokens[0] }}{% for m in messages %}{{ m.role }}:{{ m.content.strip().upper() }}{% endfor %}{% if add_generation_prompt %}A{% endif %}";
    for selection in [
        json!(template),
        json!({"default": template, "tool_use": "{% bad %}"}),
        json!([{"name": "tool_use", "template": "{% bad %}"}, {"name": "default", "template": template}]),
    ] {
        let directory = assets(json!({"chat_template": selection,
            "bos_token": {"content": "B"}, "additional_special_tokens": [{"content": "X"}]}))?;
        let tokenizer = Tokenizer::from_dir(directory.path())?;
        let messages = [
            Message {
                role: Role::Tool,
                content: " café ".into(),
            },
            Message {
                role: Role::Other("critic".into()),
                content: " yes ".into(),
            },
        ];
        assert_eq!(
            tokenizer.render_chat(
                &messages,
                ChatTemplateOptions {
                    continuation: ChatContinuation::StartAssistant,
                    ..Default::default()
                }
            )?,
            "BXtool:CAFÉcritic:YESA"
        );
    }
    Ok(())
}

#[test]
fn missing_invalid_and_raising_templates_are_typed() -> Result<(), TokenizerError> {
    for template in [json!(null), json!({"tool_use": "text"}), json!([])] {
        let directory = assets(json!({"chat_template": template}))?;
        let tokenizer = Tokenizer::from_dir(directory.path())?;
        assert!(matches!(
            tokenizer.render_chat(&[], ChatTemplateOptions::default()),
            Err(ChatTemplateError::MissingTemplate)
        ));
    }
    let directory = assets(json!({"chat_template": "{% invalid %}"}))?;
    assert!(matches!(
        Tokenizer::from_dir(directory.path()),
        Err(TokenizerError::ChatTemplate(ChatTemplateError::Minijinja(
            _
        )))
    ));
    write_json(
        &directory.path().join("tokenizer_config.json"),
        json!({"chat_template": "{{ raise_exception('rejected') }}"}),
    )?;
    assert!(matches!(
        Tokenizer::from_dir(directory.path())?.render_chat(&[], ChatTemplateOptions::default()),
        Err(ChatTemplateError::Minijinja(_))
    ));
    Ok(())
}

#[test]
fn continuation_spacing_repetition_empty_and_missing_content() -> Result<(), ChatTemplateError> {
    let options = ChatTemplateOptions {
        continuation: ChatContinuation::ContinueLast,
        ..Default::default()
    };
    let mut env = environment();
    for (template, message, expected) in [
        ("{{ messages[-1].content }}!", " café  ", " café  "),
        ("{{ messages[-1].content.strip() }}!", " café  ", "café"),
        ("{{ messages[-1].content }}!", "", ""),
        ("{{ messages[-1].content.strip() }}!", " \n ", ""),
        ("same!{{ messages[-1].content }}!same", "same", "same!same"),
    ] {
        env.add_template("chat", template)?;
        let messages = [Message {
            role: Role::Assistant,
            content: message.into(),
        }];
        assert_eq!(
            render_chat(&env, &Default::default(), &messages, options.clone())?,
            expected
        );
    }
    assert!(matches!(
        render_chat(&env, &Default::default(), &[], options.clone()),
        Err(ChatTemplateError::IncompatibleContinuation)
    ));
    let messages = [Message {
        role: Role::Assistant,
        content: "same".into(),
    }];
    for template in [
        "same!",
        "{{ messages[-1].content.replace('same', '') }}",
        "{{ messages[-1].content.replace('CONTINUE_FINAL_MESSAGE_TAG', '') }}",
    ] {
        env.add_template("chat", template)?;
        assert!(matches!(
            render_chat(&env, &Default::default(), &messages, options.clone()),
            Err(ChatTemplateError::FinalMessageNotFound)
        ));
    }
    Ok(())
}

#[test]
fn whitespace_undefined_and_thinking_options() -> Result<(), TokenizerError> {
    let directory = assets(
        json!({"chat_template": "x\n  {% if true %}\ny\n  {% endif %}\n{{ missing }}{% if enable_thinking is not defined %}default{% elif enable_thinking %}think{% else %}quiet{% endif %}"}),
    )?;
    let tokenizer = Tokenizer::from_dir(directory.path())?;
    for (enable_thinking, expected) in [
        (None, "default"),
        (Some(true), "think"),
        (Some(false), "quiet"),
    ] {
        assert_eq!(
            tokenizer.render_chat(
                &[],
                ChatTemplateOptions {
                    enable_thinking,
                    ..Default::default()
                }
            )?,
            format!("x\ny\n{expected}")
        );
    }
    Ok(())
}

#[test]
fn streaming_utf8_and_final_flush() -> Result<(), TokenizerError> {
    let mut raw: Value = read_json(&fixture_root().join("llama-base/tokenizer.json"))?;
    raw["added_tokens"] = json!([]);
    raw["model"]["vocab"] = json!({"<unk>": 0, "<0xC3>": 1, "<0xA9>": 2, "<0x20>": 3});
    raw["decoder"] = json!({"type": "ByteFallback"});
    let tokenizer = Tokenizer::from_bytes(&serde_json::to_vec(&raw)?)?;
    let mut stream = tokenizer.decode_stream();
    assert_eq!(stream.step(TokenId(1))?, None);
    assert_eq!(stream.step(TokenId(2))?, Some("é".into()));
    assert_eq!(stream.step(TokenId(3))?, Some(" ".into()));
    assert_eq!(stream.finish()?, "");
    let mut stream = tokenizer.decode_stream();
    assert_eq!(stream.step(TokenId(1))?, None);
    assert_eq!(stream.finish()?, "�");
    assert_eq!(tokenizer.decode_stream().finish()?, "");
    let mut stream = tokenizer.decode_stream();
    assert_eq!(stream.step(TokenId(1))?, None);
    assert_eq!(stream.step(TokenId(2))?, Some("é".into()));
    assert_eq!(stream.step(TokenId(1))?, None);
    assert!(matches!(stream.finish(), Err(TokenizerError::Tokenizer(_))));

    raw["model"]["vocab"] = json!({"<unk>": 0, "Ã": 1, "©": 2, "Ġ": 3});
    raw["decoder"] = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true});
    let tokenizer = Tokenizer::from_bytes(&serde_json::to_vec(&raw)?)?;
    let mut stream = tokenizer.decode_stream();
    assert_eq!(stream.step(TokenId(1))?, None);
    assert_eq!(stream.step(TokenId(2))?, Some("é".into()));
    assert_eq!(stream.step(TokenId(1))?, None);
    assert_eq!(stream.finish()?, "�");
    Ok(())
}

#[test]
fn encoding_failure_keeps_the_tokenizers_error() -> Result<(), TokenizerError> {
    let mut raw: Value = read_json(&fixture_root().join("llama-base/tokenizer.json"))?;
    raw["model"]["unk_token"] = json!("missing unknown token");
    let tokenizer = Tokenizer::from_bytes(&serde_json::to_vec(&raw)?)?;
    assert!(matches!(
        tokenizer.encode("not-in-vocabulary"),
        Err(TokenizerError::Tokenizer(_))
    ));
    Ok(())
}

#[test]
fn qwen3_template_fixture() -> Result<(), ChatTemplateError> {
    let config: Value = serde_json::from_str(include_str!(
        "../../tests/fixtures/qwen3/tokenizer_config.json"
    ))?;
    let template = config
        .get("chat_template")
        .and_then(Value::as_str)
        .ok_or(ChatTemplateError::MissingTemplate)?;
    let mut env = environment();
    env.add_template_owned("chat", template.to_owned())?;
    let messages = [Message {
        role: Role::User,
        content: "hello".into(),
    }];
    let rendered = render_chat(
        &env,
        &Default::default(),
        &messages,
        ChatTemplateOptions::default(),
    )?;
    assert_eq!(rendered, "<|im_start|>user\nhello<|im_end|>\n");
    Ok(())
}
#[test]
fn continuation_handles_trimmed_unicode_and_typed_failures() -> Result<(), ChatTemplateError> {
    let mut env = environment();
    env.add_template("chat", "{{ messages[-1].content | trim }}!")?;
    let options = ChatTemplateOptions {
        continuation: ChatContinuation::ContinueLast,
        ..Default::default()
    };
    let message = [Message {
        role: Role::Assistant,
        content: " café  ".into(),
    }];
    assert_eq!(
        render_chat(&env, &Default::default(), &message, options.clone())?,
        "café"
    );
    assert!(matches!(
        render_chat(&env, &Default::default(), &[], options.clone()),
        Err(ChatTemplateError::IncompatibleContinuation)
    ));
    env.add_template("chat", "removed")?;
    assert!(matches!(
        render_chat(&env, &Default::default(), &message, options),
        Err(ChatTemplateError::FinalMessageNotFound)
    ));
    Ok(())
}

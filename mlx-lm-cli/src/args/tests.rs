use super::*;

fn parse(extra: &[&str]) -> Result<Cli, clap::Error> {
    Cli::try_parse_from(
        [
            "mlx-lm",
            "generate",
            "--model",
            "missing",
            "--prompt",
            " literal prompt ",
        ]
        .into_iter()
        .chain(extra.iter().copied()),
    )
}

#[test]
fn generate_accepts_explicit_local_source() {
    assert!(parse(&[]).is_ok());
}

#[test]
fn defaults_match_library() {
    let Command::Generate(args) = parse(&[]).unwrap().command else {
        panic!()
    };
    let options = args.options();
    let defaults = mlx_lm::GenerationOptions::default();
    assert_eq!(args.prompt, " literal prompt ");
    assert_eq!(options.max_tokens, defaults.max_tokens);
    assert_eq!(options.prefill_chunk_size, defaults.prefill_chunk_size);
    assert_eq!(options.sampling.temperature, defaults.sampling.temperature);
    assert!(options.sampling.top_p.is_none());
    assert!(options.sampling.top_k.is_none());
    assert!(options.sampling.min_p.is_none());
    assert!(options.sampling.seed.is_none());
    assert!(options.stop.stop_strings.is_empty());
    assert!(matches!(
        options.stop.tokens,
        mlx_lm::StopTokenPolicy::Tokenizer
    ));
}

#[test]
fn translates_every_sampling_flag_and_literal_stops() {
    let Command::Generate(args) = parse(&[
        "--max-tokens",
        "8",
        "--temperature",
        "0.7",
        "--top-p",
        "0.9",
        "--top-k",
        "5",
        "--min-p",
        "0.1",
        "--min-tokens-to-keep",
        "2",
        "--seed",
        "18446744073709551615",
        "--stop",
        " 雪\n",
        "--stop",
        "END",
        "--format",
        "jsonl",
    ])
    .unwrap()
    .command
    else {
        panic!()
    };
    let options = args.options();
    assert_eq!(options.max_tokens.get(), 8);
    assert_eq!(options.sampling.temperature, 0.7);
    assert_eq!(options.sampling.top_p, Some(0.9));
    assert_eq!(options.sampling.top_k.unwrap().get(), 5);
    let min_p = options.sampling.min_p.unwrap();
    assert_eq!(min_p.probability, 0.1);
    assert_eq!(min_p.min_tokens_to_keep.get(), 2);
    assert_eq!(options.sampling.seed, Some(u64::MAX));
    assert_eq!(options.stop.stop_strings, [" 雪\n", "END"]);
    assert_eq!(args.format, GenerationFormat::Jsonl);
}

#[test]
fn min_p_defaults_to_one_and_accepts_endpoints() {
    for probability in ["0", "1"] {
        let Command::Generate(args) = parse(&["--min-p", probability]).unwrap().command else {
            panic!()
        };
        assert_eq!(
            args.options()
                .sampling
                .min_p
                .unwrap()
                .min_tokens_to_keep
                .get(),
            1
        );
    }
    assert!(parse(&["--top-p", "1", "--temperature", "0"]).is_ok());
}

#[test]
fn rejects_invalid_options_before_loading() {
    for flags in [
        vec!["--max-tokens", "0"],
        vec!["--top-k", "0"],
        vec!["--min-p", "0.5", "--min-tokens-to-keep", "0"],
        vec!["--min-tokens-to-keep", "1"],
        vec!["--temperature=-1"],
        vec!["--temperature", "NaN"],
        vec!["--temperature", "inf"],
        vec!["--top-p", "0"],
        vec!["--top-p", "1.1"],
        vec!["--top-p", "NaN"],
        vec!["--min-p=-0.1"],
        vec!["--min-p", "1.1"],
        vec!["--min-p", "inf"],
        vec!["--stop", ""],
        vec!["--seed", "-1"],
        vec!["--max-tokens", "18446744073709551616"],
        vec!["--format", "json"],
    ] {
        assert_eq!(parse(&flags).unwrap_err().exit_code(), 2, "{flags:?}");
    }
}

#[test]
fn vocabulary_limits_are_left_to_public_validation() {
    let Command::Generate(args) = parse(&["--top-k", "65"]).unwrap().command else {
        panic!()
    };
    assert!(matches!(
        args.options().sampling.validate(64),
        Err(mlx_lm::SamplingError::TopKExceedsVocabulary { .. })
    ));
}

#[test]
fn exactly_two_commands_and_explicit_source() {
    for command in ["chat", "convert", "serve", "help"] {
        assert!(Cli::try_parse_from(["mlx-lm", command]).is_err());
    }
    assert!(Cli::try_parse_from(["mlx-lm", "info"]).is_err());
    assert!(Cli::try_parse_from(["mlx-lm", "info", "some/path"]).is_err());
    assert!(Cli::try_parse_from(["mlx-lm", "info", "--model", "x", "--format", "json"]).is_ok());
    assert!(Cli::try_parse_from(["mlx-lm", "info", "--model", "x", "--format", "jsonl"]).is_err());
}

#[cfg(not(feature = "hf-hub"))]
#[test]
fn disabled_hub_grammar() {
    for flags in [
        vec!["--repo", "org/model"],
        vec!["--revision", "main"],
        vec!["--offline"],
        vec!["--cache-dir", "cache"],
    ] {
        assert!(parse(&flags).is_err(), "{flags:?}");
    }
}

#[cfg(feature = "hf-hub")]
#[test]
fn hub_flags_require_repo_and_translate() {
    for flags in [
        vec!["--repo", "org/model"],
        vec!["--revision", "main"],
        vec!["--offline"],
        vec!["--cache-dir", "cache"],
    ] {
        assert!(parse(&flags).is_err(), "{flags:?}");
    }
    let Command::Info(args) = Cli::try_parse_from([
        "mlx-lm",
        "info",
        "--repo",
        "org/model",
        "--revision",
        "refs/pr/1",
        "--offline",
        "--cache-dir",
        "cache",
    ])
    .unwrap()
    .command
    else {
        panic!()
    };
    let options = args.source.hub_options();
    assert_eq!(options.revision.as_deref(), Some("refs/pr/1"));
    assert!(options.offline);
    assert_eq!(
        options.cache_dir.unwrap(),
        std::path::PathBuf::from("cache")
    );
}

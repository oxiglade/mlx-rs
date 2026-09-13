use super::*;
use crate::{
    model::gguf::{tokenizer_metadata, GgufTokenizerMetadata},
    LoadError,
};

impl Tokenizer {
    pub(crate) fn validate_gguf(
        &self,
        file: &mlx_rs::io::GgufFile,
        vocabulary_size: usize,
    ) -> Result<(), LoadError> {
        self.validate_gguf_pairing(&tokenizer_metadata(file)?, vocabulary_size)
    }

    fn validate_gguf_pairing(
        &self,
        metadata: &GgufTokenizerMetadata,
        vocabulary_size: usize,
    ) -> Result<(), LoadError> {
        let bos = self
            .bos_token()
            .and_then(|token| self.inner.token_to_id(token))
            .map(TokenId);
        let mut ids: Vec<_> = self
            .inner
            .get_vocab(true)
            .into_values()
            .map(TokenId)
            .chain(self.eos.iter().copied())
            .chain(bos)
            .chain(metadata.bos_token_id)
            .chain(metadata.eos_token_id)
            .collect();
        // Configured special-token content may differ from the resolved EOS override.
        for name in ["bos_token", "eos_token"] {
            if let Some(id) = self
                .special_tokens
                .get(name)
                .and_then(Value::as_str)
                .and_then(|s| self.inner.token_to_id(s))
            {
                ids.push(TokenId(id));
            }
        }
        ids.sort_unstable_by_key(|id| id.0);
        for token_id in ids {
            if u64::from(token_id.0) >= vocabulary_size as u64 {
                return Err(LoadError::TokenizerVocabularyOutOfRange {
                    token_id,
                    vocabulary_size,
                });
            }
        }
        if metadata
            .tokens
            .as_ref()
            .is_some_and(|tokens| tokens.len() > vocabulary_size)
        {
            return Err(crate::ConfigError::InvalidNumericField {
                field: "tokenizer.ggml.tokens".into(),
                reason: "token count exceeds embedding rows".into(),
            }
            .into());
        }
        for (key, embedded, agrees) in [
            (
                "tokenizer.ggml.bos_token_id",
                metadata.bos_token_id,
                metadata.bos_token_id == bos,
            ),
            (
                "tokenizer.ggml.eos_token_id",
                metadata.eos_token_id,
                metadata
                    .eos_token_id
                    .is_some_and(|id| self.eos.contains(&id)),
            ),
        ] {
            let Some(id) = embedded else { continue };
            if !agrees {
                return Err(LoadError::TokenizerMetadataMismatch {
                    key: key.into(),
                    expected: id.0.to_string(),
                    actual: if key.ends_with("bos_token_id") {
                        format!("{:?}", bos.map(u32::from))
                    } else {
                        format!(
                            "{:?}",
                            self.eos.iter().copied().map(u32::from).collect::<Vec<_>>()
                        )
                    },
                });
            }
            if let Some(tokens) = &metadata.tokens {
                let embedded = tokens.get(id.0 as usize);
                let caller = if key.ends_with("bos_token_id") {
                    self.bos_token().map(str::to_owned)
                } else {
                    self.inner.id_to_token(id.0)
                };
                if embedded != caller.as_ref() || embedded.is_none() {
                    return Err(LoadError::TokenizerMetadataMismatch {
                        key: "tokenizer.ggml.tokens".into(),
                        expected: format!("{}: {embedded:?}", id.0),
                        actual: format!("{}: {caller:?}", id.0),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Tokenizer {
        Tokenizer::from_dir(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures/llama-base"),
        )
        .unwrap()
    }
    fn embedded(t: &Tokenizer) -> GgufTokenizerMetadata {
        GgufTokenizerMetadata {
            tokens: None,
            bos_token_id: t
                .bos_token()
                .and_then(|s| t.inner.token_to_id(s))
                .map(TokenId),
            eos_token_id: t.eos.first().copied(),
        }
    }
    #[test]
    fn gguf_pairing_bounds_and_padded_embeddings() {
        let mut t = fixture();
        let metadata = embedded(&t);
        t.validate_gguf_pairing(&metadata, 128).unwrap();
        t.eos.push(TokenId(128));
        assert!(matches!(
            t.validate_gguf_pairing(&metadata, 128),
            Err(LoadError::TokenizerVocabularyOutOfRange {
                token_id: TokenId(128),
                vocabulary_size: 128
            })
        ));
        t.eos.pop();
        t.inner
            .add_tokens(&[tokenizers::AddedToken::from("outside_vocab", false)]);
        assert!(matches!(
            t.validate_gguf_pairing(&metadata, 64),
            Err(LoadError::TokenizerVocabularyOutOfRange {
                token_id: TokenId(64),
                vocabulary_size: 64
            })
        ));
    }
    #[test]
    fn gguf_pairing_same_size_special_id_mismatch() {
        let mut t = fixture();
        let metadata = embedded(&t);
        t.special_tokens
            .insert("bos_token".into(), "<|start_header_id|>".into());
        assert!(
            matches!(t.validate_gguf_pairing(&metadata, 64), Err(LoadError::TokenizerMetadataMismatch { key, .. }) if key == "tokenizer.ggml.bos_token_id")
        );
        let mut t = fixture();
        t.eos = vec![TokenId(4)];
        assert!(
            matches!(t.validate_gguf_pairing(&metadata, 64), Err(LoadError::TokenizerMetadataMismatch { key, .. }) if key == "tokenizer.ggml.eos_token_id")
        );
    }
    #[test]
    fn gguf_pairing_missing_config_and_embedded_spelling() {
        let t = fixture();
        let mut metadata = embedded(&t);
        let bare = Tokenizer::from_bytes(
            &std::fs::read(
                Path::new(env!("CARGO_MANIFEST_DIR"))
                    .join("../conformance/mlx-lm/fixtures/llama-base/tokenizer.json"),
            )
            .unwrap(),
        )
        .unwrap();
        assert!(matches!(
            bare.validate_gguf_pairing(&metadata, 64),
            Err(LoadError::TokenizerMetadataMismatch { .. })
        ));
        let mut tokens: Vec<_> = (0..64).map(|id| t.inner.id_to_token(id).unwrap()).collect();
        metadata.tokens = Some(tokens.clone());
        t.validate_gguf_pairing(&metadata, 64).unwrap();
        tokens[metadata.bos_token_id.unwrap().0 as usize] = "wrong spelling".into();
        metadata.tokens = Some(tokens);
        assert!(
            matches!(t.validate_gguf_pairing(&metadata, 64), Err(LoadError::TokenizerMetadataMismatch { key, .. }) if key == "tokenizer.ggml.tokens")
        );
    }
    #[test]
    fn gguf_pairing_does_not_claim_interior_token_identity() {
        let t = fixture();
        let metadata = embedded(&t);
        let path =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../conformance/mlx-lm/fixtures/llama-base");
        let mut json: Value =
            serde_json::from_slice(&std::fs::read(path.join("tokenizer.json")).unwrap()).unwrap();
        let hello = json["model"]["vocab"]["hello"].clone();
        let world = json["model"]["vocab"]["world"].clone();
        json["model"]["vocab"]["hello"] = world;
        json["model"]["vocab"]["world"] = hello;
        let mut changed = Tokenizer::from_bytes(&serde_json::to_vec(&json).unwrap()).unwrap();
        changed.eos = t.eos.clone();
        changed.special_tokens = t.special_tokens.clone();
        changed.validate_gguf_pairing(&metadata, 64).unwrap();
        assert_ne!(
            changed.encode("hello world").unwrap(),
            t.encode("hello world").unwrap()
        );
    }
}

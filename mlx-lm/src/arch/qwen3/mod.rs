use super::{ArchitectureFactory, DecoderModel, ParsedArchitecture};
use crate::{
    config::RawConfig,
    weights::{WeightDisposition, WeightManifest},
    Config, ConfigError, LoadError,
};

#[derive(serde::Deserialize)]
struct WireConfig {
    #[serde(flatten)]
    fields: std::collections::BTreeMap<String, serde_json::Value>,
}
pub(crate) struct ParsedConfig {
    pub(crate) config: Config,
    wire: WireConfig,
}
pub(crate) struct Factory;
impl ArchitectureFactory for Factory {
    fn parse_config(&self, _raw: &RawConfig) -> Result<ParsedArchitecture, ConfigError> {
        Err(ConfigError::UnsupportedArchitecture(
            crate::NotYetImplemented("qwen3 config parsing").to_string(),
        ))
    }
    fn build(
        &self,
        _parsed: ParsedArchitecture,
        _weights: &WeightManifest,
    ) -> Result<Box<dyn DecoderModel>, LoadError> {
        Err(ConfigError::UnsupportedArchitecture(
            crate::NotYetImplemented("qwen3 model construction").to_string(),
        )
        .into())
    }
    fn map_safetensors_key(&self, _external: &str) -> WeightDisposition {
        WeightDisposition::Reject
    }
    fn map_gguf_key(&self, _external: &str) -> WeightDisposition {
        WeightDisposition::Reject
    }
}

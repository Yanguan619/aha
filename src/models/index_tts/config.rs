//! IndexTTS 2.5 的配置（来自 GGUF 内嵌的 `config.yaml`）。

use serde::Deserialize;

/// `config.yaml` 的顶层结构（目前只把 gpt 段结构化，其余先留着原样）
#[derive(Debug, Clone, Deserialize)]
pub struct IndexTtsConfig {
    pub gpt: GptConfig,
    /// 训练用的 mel 参数，推理时用来确认 n_mels / hop_length
    #[serde(default)]
    pub dataset: Option<DatasetConfig>,
    /// semantic codec 段，字段和 `mask_gct::config::SemanticCodec` 对得上
    #[serde(default)]
    pub semantic_codec: Option<crate::models::mask_gct::config::SemanticCodec>,
    /// s2mel 段：嵌套很深且带字符串化的 `"None"`，先保留成 Value，用到时再结构化
    #[serde(default)]
    pub s2mel: Option<serde_yaml::Value>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatasetConfig {
    #[serde(default)]
    pub bpe_model: Option<String>,
    #[serde(default)]
    pub sample_rate: Option<usize>,
    #[serde(default)]
    pub mel: Option<DatasetMelConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DatasetMelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_length: usize,
    pub win_length: usize,
    pub n_mels: usize,
    #[serde(default)]
    pub mel_fmin: Option<f64>,
    #[serde(default)]
    pub normalize: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct GptConfig {
    pub model_dim: usize,
    pub max_mel_tokens: usize,
    pub max_text_tokens: usize,
    pub heads: usize,
    pub use_mel_codes_as_input: bool,
    pub mel_length_compression: usize,
    pub layers: usize,
    pub number_text_tokens: usize,
    pub number_mel_codes: usize,
    pub start_mel_token: usize,
    pub stop_mel_token: usize,
    pub start_text_token: usize,
    pub stop_text_token: usize,
    #[serde(default)]
    pub condition_type: Option<String>,
    #[serde(default)]
    pub condition_module: Option<ConditionModuleConfig>,
    #[serde(default)]
    pub emo_condition_module: Option<ConditionModuleConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ConditionModuleConfig {
    pub output_size: usize,
    pub linear_units: usize,
    pub attention_heads: usize,
    pub num_blocks: usize,
    #[serde(default)]
    pub input_layer: Option<String>,
    #[serde(default)]
    pub perceiver_mult: usize,
}

impl IndexTtsConfig {
    pub fn from_yaml(bytes: &[u8]) -> anyhow::Result<Self> {
        Ok(serde_yaml::from_slice(bytes)?)
    }
}

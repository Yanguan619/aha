//! TurboVLA 的配置：`model_config`（turbovla-libero/config.json）+ BERT / DINOv3 的超参。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

fn default_max_length() -> usize {
    256
}

#[derive(Debug, Clone, Deserialize)]
pub struct TextConfig {
    #[serde(default)]
    pub model_name_or_path: String,
    #[serde(default = "default_max_length")]
    pub max_length: usize,
    /// BERT 输入固定 pad 到这个长度（LIBERO 是 21）；None = pad 到 batch 内最长。
    #[serde(default)]
    pub padding_length: Option<usize>,
    #[serde(default)]
    pub sub_sentence_present: bool,
    #[serde(default)]
    pub zero_padded_tokens: bool,
    /// 少数指令有专属 pad 长度（LIBERO 语料），其余用 `padding_length`。
    #[serde(default)]
    pub padding_length_by_instruction: HashMap<String, usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    #[serde(default)]
    pub model_name_or_path: String,
    pub image_size: usize,
    #[serde(default = "default_two")]
    pub num_views: usize,
    /// "view"（LIBERO）或 "learned_patch"（RoboTwin）
    #[serde(default = "default_view")]
    pub position_embedding: String,
    #[serde(default)]
    pub encode_views_separately: bool,
    #[serde(default)]
    pub dropout: f32,
}

fn default_two() -> usize {
    2
}
fn default_view() -> String {
    "view".to_owned()
}

#[derive(Debug, Clone, Deserialize)]
pub struct InteractionConfig {
    pub hidden_dim: usize,
    pub nheads: usize,
    pub num_layers: usize,
    pub dim_feedforward: usize,
    pub enhancer_inner_dim: usize,
    #[serde(default)]
    pub padding_strategy: String,
    #[serde(default = "default_normalized")]
    pub residual_style: String,
    #[serde(default)]
    pub attention_backend: String,
}

fn default_normalized() -> String {
    "normalized".to_owned()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ActionConfig {
    pub action_dim: usize,
    pub state_dim: usize,
    pub horizon: usize,
    #[serde(default = "default_state_tokens")]
    pub num_state_tokens: usize,
    pub num_layers: usize,
    pub mlp_hidden_dim: usize,
    pub state_hidden_dim: usize,
}

fn default_state_tokens() -> usize {
    2
}

/// `model_config` 本体。
#[derive(Debug, Clone, Deserialize)]
pub struct TurboVlaConfig {
    #[serde(default)]
    pub name: String,
    pub text: TextConfig,
    pub vision: VisionConfig,
    pub interaction: InteractionConfig,
    pub action: ActionConfig,
}

/// BERT（bert-base-uncased）的超参，只取编码器需要的。
#[derive(Debug, Clone, Deserialize)]
pub struct BertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_type_vocab")]
    pub type_vocab_size: usize,
    #[serde(default = "default_bert_eps")]
    pub layer_norm_eps: f64,
}

fn default_max_pos() -> usize {
    512
}
fn default_type_vocab() -> usize {
    2
}
fn default_bert_eps() -> f64 {
    1e-12
}

/// DINOv3 ViT-B/16 的超参。
#[derive(Debug, Clone, Deserialize)]
pub struct DinoConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub patch_size: usize,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
    #[serde(default)]
    pub num_register_tokens: usize,
    #[serde(default = "default_dino_eps")]
    pub layer_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,
    #[serde(default = "default_true")]
    pub mlp_bias: bool,
    #[serde(default = "default_true")]
    pub query_bias: bool,
    #[serde(default)]
    pub key_bias: bool,
    #[serde(default = "default_true")]
    pub value_bias: bool,
    #[serde(default = "default_true")]
    pub proj_bias: bool,
    #[serde(default)]
    pub use_gated_mlp: bool,
}

fn default_image_size() -> usize {
    256
}
fn default_dino_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f32 {
    100.0
}
fn default_num_channels() -> usize {
    3
}
fn default_true() -> bool {
    true
}

/// 一个 TurboVLA 模型目录里的全部配置。
#[derive(Debug, Clone)]
pub struct ModelLayout {
    pub dir: PathBuf,
    pub model: TurboVlaConfig,
    pub text: BertConfig,
    pub vision: DinoConfig,
    pub tokenizer: PathBuf,
    pub weights: PathBuf,
}

impl ModelLayout {
    /// 从模型目录读齐四份文件（config.json / text_config.json / vision_config.json / tokenizer.json / model.safetensors）。
    pub fn open(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let read = |name: &str| -> Result<String> {
            let path = dir.join(name);
            std::fs::read_to_string(&path)
                .with_context(|| format!("读不到 {}（TurboVLA 模型目录不完整？）", path.display()))
        };
        let model: TurboVlaConfig =
            serde_json::from_str(&read("config.json")?).context("解析 config.json")?;
        let text: BertConfig =
            serde_json::from_str(&read("text_config.json")?).context("解析 text_config.json")?;
        let vision: DinoConfig =
            serde_json::from_str(&read("vision_config.json")?).context("解析 vision_config.json")?;
        let tokenizer = dir.join("tokenizer.json");
        if !tokenizer.is_file() {
            return Err(anyhow::anyhow!(
                "缺少 {}（BERT WordPiece 词表；用 tools/vla/convert_turbovla.py 生成）",
                tokenizer.display()
            ));
        }
        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(anyhow::anyhow!(
                "缺少 {}（用 tools/vla/convert_turbovla.py 从 .pth 转换）",
                weights.display()
            ));
        }
        Ok(Self {
            dir,
            model,
            text,
            vision,
            tokenizer,
            weights,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_names_are_consistent() {
        // 只守字段默认值（不依赖真实模型目录）
        let c: TurboVlaConfig = serde_json::from_str(
            r#"{
              "name":"TurboVLA",
              "text":{"model_name_or_path":"bert-base-uncased","max_length":256,"padding_length":21,
                      "sub_sentence_present":true,"zero_padded_tokens":false},
              "vision":{"model_name_or_path":"dino","image_size":256,"num_views":2,
                        "position_embedding":"view","encode_views_separately":true},
              "interaction":{"hidden_dim":256,"nheads":8,"num_layers":6,"dim_feedforward":2048,
                             "enhancer_inner_dim":1024,"padding_strategy":"key_padding_mask"},
              "action":{"action_dim":7,"state_dim":8,"horizon":12,"num_layers":3,
                        "mlp_hidden_dim":512,"state_hidden_dim":256}
            }"#,
        )
        .unwrap();
        assert_eq!(c.text.padding_length, Some(21));
        assert_eq!(c.vision.num_views, 2);
        assert_eq!(c.interaction.residual_style, "normalized");
        assert_eq!(c.action.num_state_tokens, 2);
    }
}

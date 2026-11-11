//! GPT-SoVITS（v2 / v2Pro / v2ProPlus）推理实现。
//!
//! 流水线见 `docs/plan-gpt-sovits.md`。当前进度：
//!
//! - [x] P0 权重转换（`scripts/gpt_sovits/convert_weights.py`）
//! - [x] P1 中文文本前端（[`text_frontend`]）
//! - [x] P2 HuBERT + RoBERTa（[`hubert`] / [`roberta`] / [`wordpiece`]，待数值对齐）
//! - [x] P3 T2S AR 解码（[`t2s`]，待 token 对齐）
//! - [x] P4 SoVITS 合成器（[`sovits`] / [`vits_attn`] / [`flow`] / [`hifigan`]），
//!   v2Pro 的 `eres2net` 说话人模型未实现（可用零向量跑通链路）
//! - [ ] P5 服务接入

#[cfg(feature = "cuda")]
pub mod ar_graph;
pub mod eres2net;
pub mod flow;
pub mod hifigan;
pub mod hubert;
pub mod pipeline;
pub mod quantizer;
pub mod roberta;
pub mod sovits;
pub mod style;
pub mod t2s;
pub mod text_frontend;
pub mod vits_attn;
pub mod wordpiece;

pub use eres2net::{ERes2NetV2, sv_fbank};
pub use flow::FlowBlock;
pub use hifigan::HifiGan;
pub use hubert::Hubert;
pub use pipeline::{GptSovitsModel, VoiceConfig};
pub use quantizer::SemanticQuantizer;
pub use roberta::RobertaEncoder;
pub use sovits::{Sovits, SovitsConfig, SynthOptions};
pub use style::{StyleEncoder, linear_spectrogram};
pub use t2s::{GenOptions, T2SModelConfig, Text2Semantic};
pub use text_frontend::{G2pOutput, ZhFrontend};
pub use wordpiece::WordPieceTokenizer;

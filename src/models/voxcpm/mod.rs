//! VoxCPM（0.5B / 1.5 / 2）的 TTS 实现。
//!
//! 结构：`processor` 负责文本/音频的前处理，`model` 是 DiT + CFM 主体，
//! `layers` 是主干里复用的几个算子（loc-DiT / loc-Enc / CFM / 标量量化），
//! `audio_vae` 是声码器，`minicpm4` 是文本侧的 MiniCPM 主干，`tokenizer` 是中文切词，
//! `generate` 把这一套包装成 `GenerateModel` 供 CLI / server 调用。

pub mod audio_vae;
pub mod config;
pub mod generate;
pub mod layers;
pub mod minicpm4;
pub mod model;
pub mod processor;
pub mod tokenizer;

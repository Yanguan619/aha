//! IndexTTS 2.5（audiocpp GGUF 打包）。

pub mod config;
pub mod emo;
pub mod frontend;
pub mod gpt;
#[allow(clippy::module_inception)]
pub mod pth_loader;
pub mod pipeline;
pub mod s2mel;
pub mod tokenizer;


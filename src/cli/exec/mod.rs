//! CLI exec module for direct model inference
//!
//! `aha run` 的每个模型一个模块，负责输入解析 + 调模型 + 落盘输出。
//!
//! 这里**故意没有 trait**：各模块只是 `pub fn run(..)` 的命名空间，没有多态需求。
//! 它们只服务 CLI，所以别在别处引用。

pub mod all_minilm_l6_v2;
pub mod deepseek_ocr;
pub mod fun_asr_nano;
pub mod glm_asr_nano;
pub mod glm_ocr;
#[cfg(feature = "gpt-sovits")]
pub mod gpt_sovits;
pub mod hunyuan_ocr;
#[cfg(feature = "sherpa-tts")]
pub mod kokoro;
pub mod lfm2;
pub mod lfm2vl;
#[cfg(feature = "llama")]
pub mod llama_cpp;
pub mod minicpm4;
pub mod minicpm5;
pub mod paddleocr_vl;
pub mod qwen2_5vl;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_asr;
pub mod qwen3_embedding;
pub mod qwen3_reranker;
pub mod qwen3vl;
pub mod rmbg2_0;
#[cfg(feature = "sherpa-asr")]
pub mod sense_voice;
pub mod voxcpm;

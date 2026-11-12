//! 用原始 PyTorch `.pth` 转出的 safetensors 权重替换 GGUF 里的 Q4_K 权重做精度对比。
//!
//! 原始权重的 key 前缀是 `net.`（s2mel）或无前缀（gpt/codec），
//! 和 GGUF 里的前缀一致，只是多了 `net.`。
//!
//! ```ignore
//! let pipe = IndexTtsPipeline::from_gguf(&gguf_path, &device)?;
//! let s2mel_st = r"E:\tmp\indextts25-orig\s2mel.safetensors";
//! let s2mel = pth_loader::load_s2mel_from_safetensors(s2mel_st, &device)?;
//! pipe.s2mel = s2mel; // 替换
//! ```

use anyhow::Result;
use candle_core::{DType, Device};
use candle_nn::VarBuilder;

use crate::models::index_tts::s2mel::S2Mel;

/// 从原始 s2mel.safetensors 构建一个 S2Mel（原始权重，无 Q4_K 损失）
pub fn load_s2mel_from_safetensors(path: &str, device: &Device) -> Result<S2Mel> {
    load_s2mel_from_safetensors_dtype(path, device, DType::F32)
}

/// 指定精度加载原始 s2mel 权重（F16 可以提速 ~2x，精度损失极小）
pub fn load_s2mel_from_safetensors_dtype(
    path: &str,
    device: &Device,
    dtype: DType,
) -> Result<S2Mel> {
    let model_list = vec![path.to_string()];
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, dtype, device)? };
    // safetensors 里的 key 是 net.cfm.estimator.xxx，GGUF 里是 cfm.estimator.xxx
    let vb_net = vb.pp("net");
    S2Mel::new(
        &vb_net, 80,   // in_channels
        512,  // dim
        8,    // n_head
        13,   // n_layers
        512,  // content_dim
        1024, // lr_in
        512,  // lr_channels
        4,    // lr_blocks
        8,    // wavenet_layers
        1280, // gpt_dim
    )
}

/// 从原始 gpt.safetensors 构建 GPT（原始 F32 权重）
#[allow(clippy::too_many_arguments)]
pub fn load_gpt_from_safetensors(
    path: &str,
    device: &Device,
) -> Result<crate::models::index_tts::gpt::IndexTtsGpt> {
    let config = crate::models::index_tts::config::GptConfig {
        model_dim: 1280,
        max_mel_tokens: 1815,
        max_text_tokens: 600,
        heads: 20,
        use_mel_codes_as_input: true,
        mel_length_compression: 1024,
        layers: 24,
        number_text_tokens: 60509,
        number_mel_codes: 8194,
        start_mel_token: 8192,
        stop_mel_token: 8193,
        start_text_token: 0,
        stop_text_token: 1,
        condition_type: Some("conformer_perceiver".to_string()),
        condition_module: None,
        emo_condition_module: None,
    };
    let model_list = vec![path.to_string()];
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&model_list, DType::F32, device)? };
    crate::models::index_tts::gpt::IndexTtsGpt::new(&vb, &config)
}

//! SoVITS 的语义量化器（`module/core_vq.py` 的 `ResidualVectorQuantizer`）。
//!
//! v2 系固定 `n_q = 1`、`bins = 1024`、`dim = 768`，所以这里只实现单层：
//!
//! ```text
//! HuBERT (1,768,T) → ssl_proj(Conv1d 768→768, k=2, s=2) → (1,768,T/2)
//!   → 每帧在 1024 个码字里取欧氏距离最近的那个 → codes (T/2,)
//! ```
//!
//! 参考音频的 prompt 语义就是这样得到的（`TTS.py::_set_prompt_semantic`）。

use anyhow::Result;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder, conv1d};

/// 码字数量（v2 系固定 1024）。
pub const NUM_CODES: usize = 1024;

pub struct SemanticQuantizer {
    ssl_proj: Conv1d,
    embed: Tensor, // (NUM_CODES, 768)
    device: Device,
}

impl SemanticQuantizer {
    pub fn load(vb: VarBuilder, dim: usize, device: &Device) -> Result<Self> {
        let embed = vb.get((NUM_CODES, dim), "quantizer.vq.layers.0._codebook.embed")?;
        let ssl_proj = conv1d(
            dim,
            dim,
            2,
            Conv1dConfig {
                stride: 2,
                ..Default::default()
            },
            vb.pp("ssl_proj"),
        )?;
        Ok(Self {
            ssl_proj,
            embed,
            device: device.clone(),
        })
    }

    /// 只做 `ssl_proj`（诊断用：看量化前的投影结果）
    pub fn project(&self, hubert: &Tensor) -> Result<Tensor> {
        Ok(self.ssl_proj.forward(hubert)?)
    }

    /// HuBERT 特征 `(1, 768, T)` → 语义 codes（长度 T/2）。
    pub fn encode(&self, hubert: &Tensor) -> Result<Vec<u32>> {
        let x = self.ssl_proj.forward(hubert)?; // (1, 768, T/2)
        let (_b, dim, frames) = x.dims3()?;
        let x = x.squeeze(0)?.transpose(0, 1)?.contiguous()?; // (T/2, 768)

        let x_norm = x.sqr()?.sum_keepdim(1)?; // (T/2, 1)
        let e_norm = self.embed.to_dtype(x.dtype())?.sqr()?.sum_keepdim(1)?.t()?; // (1, NUM_CODES)
        let dots = x.matmul(&self.embed.to_dtype(x.dtype())?.t()?.contiguous()?)?; // (T/2, NUM_CODES)
        let dist = x_norm.broadcast_add(&e_norm)?.broadcast_sub(&(dots * 2.0)?)?;

        let idx = dist.argmin(1)?; // (T/2,) u32
        let codes = idx.to_dtype(DType::U32)?.to_vec1::<u32>()?;
        debug_assert_eq!(codes.len(), frames);
        let _ = dim;
        Ok(codes)
    }

    /// codes → 量化后的连续向量 `(1, 768, T)`（`quantizer.decode`，合成阶段要用）。
    pub fn decode(&self, codes: &[u32]) -> Result<Tensor> {
        let idx = Tensor::from_vec(codes.to_vec(), codes.len(), &self.device)?;
        let out = self.embed.index_select(&idx, 0)?; // (T, 768)
        Ok(out.t()?.unsqueeze(0)?)
    }
}

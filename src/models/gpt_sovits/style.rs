//! 音色向量 `ge`：线性幅度谱 → `MelStyleEncoder`（`ref_enc`），v2Pro 再叠加说话人向量。
//!
//! 对齐 `module/modules.py::MelStyleEncoder` + `module/models.py::SynthesizerTrn.decode`：
//!
//! ```text
//! spec(1, 1025, T) --取前 704 bin--> ref_enc -->
//!     spectral: Linear(704→128) → Mish → Linear(128→128) → Mish
//!     temporal: Conv1dGLU(128, k5) ×2
//!     slf_attn: MultiHeadAttention(head=2, dk=dv=64, 温度=sqrt(128)) + 残差
//!     fc: Linear(128→1024)
//!     时间维取均值 → (1, 1024, 1)
//! --v2Pro--> + sv_emb(20480→1024) → PReLU
//! ```
//!
//! 推理时参考音频没有 padding，所以掩码恒为全 1，这里直接省略掩码分支。

use std::path::Path;

use anyhow::Result;
use candle_core::{Device, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Linear, VarBuilder, conv1d, linear};

/// 参考谱只取前 704 个频点（`refer[:, :704]`）。
pub const REF_SPEC_BINS: usize = 704;
/// `gin_channels`（v2Pro / v2ProPlus）。
pub const STYLE_DIM: usize = 1024;

fn mish(x: &Tensor) -> Result<Tensor> {
    // x * tanh(softplus(x))
    let softplus = x.exp()?.affine(1.0, 1.0)?.log()?;
    Ok((x * softplus.tanh()?)?)
}

struct Conv1dGlu {
    conv: Conv1d,
}

impl Conv1dGlu {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv.forward(x)?;
        let channels = h.dim(1)? / 2;
        let a = h.narrow(1, 0, channels)?;
        let b = h.narrow(1, channels, channels)?;
        let gated = (a * candle_nn::ops::sigmoid(&b)?)?;
        // residual + (a * sigmoid(b))
        Ok((x + gated)?)
    }
}

struct Mha {
    w_qs: Linear,
    w_ks: Linear,
    w_vs: Linear,
    fc: Linear,
    heads: usize,
    d_k: usize,
    d_v: usize,
    temperature: f64,
}

impl Mha {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, t, _) = x.dims3()?;
        let q = self.w_qs.forward(x)?;
        let k = self.w_ks.forward(x)?;
        let v = self.w_vs.forward(x)?;
        let shape = (b, t, self.heads, self.d_k);
        let q = q.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = v
            .reshape((b, t, self.heads, self.d_v))?
            .transpose(1, 2)?
            .contiguous()?;
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? / self.temperature)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.heads * self.d_v))?;
        // MultiHeadAttention 内部带残差
        Ok((x + self.fc.forward(&out)?)?)
    }
}

pub struct MelStyleEncoder {
    spectral_0: Linear,
    spectral_3: Linear,
    temporal: Vec<Conv1dGlu>,
    attn: Mha,
    fc: Linear,
}

impl MelStyleEncoder {
    pub fn load(vb: VarBuilder, mel_bins: usize, hidden: usize, out_dim: usize) -> Result<Self> {
        let temporal = (0..2)
            .map(|i| {
                Ok(Conv1dGlu {
                    conv: conv1d(
                        hidden,
                        hidden * 2,
                        5,
                        Conv1dConfig {
                            padding: 2,
                            ..Default::default()
                        },
                        vb.pp("temporal").pp(i).pp("conv1.conv"),
                    )?,
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let heads = 2;
        let d_k = hidden / heads;
        Ok(Self {
            spectral_0: linear(mel_bins, hidden, vb.pp("spectral.0.fc"))?,
            spectral_3: linear(hidden, hidden, vb.pp("spectral.3.fc"))?,
            temporal,
            attn: Mha {
                w_qs: linear(hidden, heads * d_k, vb.pp("slf_attn.w_qs"))?,
                w_ks: linear(hidden, heads * d_k, vb.pp("slf_attn.w_ks"))?,
                w_vs: linear(hidden, heads * d_k, vb.pp("slf_attn.w_vs"))?,
                fc: linear(heads * d_k, hidden, vb.pp("slf_attn.fc"))?,
                heads,
                d_k,
                d_v: d_k,
                temperature: (hidden as f64).sqrt(),
            },
            fc: linear(hidden, out_dim, vb.pp("fc.fc"))?,
        })
    }

    fn debug(&self) -> bool {
        std::env::var("AHA_GSV_DEBUG").is_ok()
    }

    fn log_stage(&self, label: &str, x: &Tensor) -> Result<()> {
        if self.debug() {
            let v: Vec<f32> = x.flatten_all()?.to_vec1()?;
            let n = v.len() as f64;
            let mean = v.iter().map(|a| *a as f64).sum::<f64>() / n;
            let std = (v.iter().map(|a| (*a as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
            eprintln!("    [ref_enc] {label:<10} dims={:?} mean={mean:+.4} std={std:.4}", x.dims());
        }
        Ok(())
    }

    /// `spec` 为线性幅度谱 `(1, freq, T)`，内部只取前 704 bin。
    pub fn forward(&self, spec: &Tensor) -> Result<Tensor> {
        let freq = spec.dim(1)?;
        let spec = if freq > REF_SPEC_BINS {
            spec.narrow(1, 0, REF_SPEC_BINS)?
        } else {
            spec.clone()
        };
        let x = spec.squeeze(0)?.t()?.contiguous()?; // (T, 704)

        // spectral
        self.log_stage("spec_in", &x)?;
        let x = mish(&self.spectral_0.forward(&x)?)?;
        self.log_stage("spectral0", &x)?;
        let x = mish(&self.spectral_3.forward(&x)?)?;
        self.log_stage("spectral3", &x)?;

        // temporal（Conv1dGLU 在通道维）
        let mut h = x.t()?.unsqueeze(0)?.contiguous()?; // (1, 128, T)
        for conv in &self.temporal {
            h = conv.forward(&h)?;
        }
        self.log_stage("temporal", &h)?;
        let h = h.squeeze(0)?.t()?.contiguous()?; // (T, 128)

        // self-attention + fc
        let h = self.attn.forward(&h.unsqueeze(0)?)?; // (1, T, 128)
        self.log_stage("slf_attn", &h)?;
        let h = self.fc.forward(&h)?; // (1, T, 1024)
        self.log_stage("fc", &h)?;

        // 时间维平均 → (1, 1024, 1)
        Ok(h.mean(1)?.unsqueeze(2)?)
    }
}

/// `ge` 的完整计算：`ref_enc` + （v2Pro）`sv_emb` + PReLU。
pub struct StyleEncoder {
    ref_enc: MelStyleEncoder,
    sv_proj: Option<Linear>,
    prelu: Option<Tensor>,
}

impl StyleEncoder {
    pub fn load(vb: VarBuilder, is_v2pro: bool) -> Result<Self> {
        let ref_enc = MelStyleEncoder::load(vb.pp("ref_enc"), REF_SPEC_BINS, 128, STYLE_DIM)?;
        let (sv_proj, prelu) = if is_v2pro {
            (
                Some(linear(20_480, STYLE_DIM, vb.pp("sv_emb"))?),
                Some(vb.get(STYLE_DIM, "prelu.weight")?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            ref_enc,
            sv_proj,
            prelu,
        })
    }

    pub fn needs_sv(&self) -> bool {
        self.sv_proj.is_some()
    }

    /// `spec`: (1, freq, T)；`sv`: (1, 20480)（v2Pro 必须给）。
    pub fn ge(&self, spec: &Tensor, sv: Option<&Tensor>) -> Result<Tensor> {
        let mut ge = self.ref_enc.forward(spec)?; // (1, 1024, 1)
        if let Some(proj) = &self.sv_proj {
            let sv = sv.ok_or_else(|| anyhow::anyhow!("v2Pro 需要说话人向量 sv_emb"))?;
            // (1, 20480) -> (1, 1024) -> (1, 1024, 1)
            let emb = proj.forward(sv)?.unsqueeze(2)?;
            ge = (&ge + emb)?;
            let weight = self.prelu.as_ref().expect("v2pro 一定有 prelu");
            ge = prelu(&ge, weight)?;
        }
        Ok(ge)
    }
}

/// `PReLU(num_parameters=C)`：`max(0,x) + w * min(0,x)`
fn prelu(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let w = weight.reshape((1, weight.dim(0)?, 1))?;
    let pos = x.relu()?;
    let neg = (x - &pos)?;
    Ok((pos + (neg * w)?)?)
}

// ---------------------------------------------------------------------------
// 参考谱（线性幅度谱）
// ---------------------------------------------------------------------------

/// 反射填充（torch 的 `pad(mode="reflect")`）。
fn reflect_pad(wave: &Tensor, pad: usize) -> Result<Tensor> {
    let len = wave.dim(0)?;
    let mut indices: Vec<u32> = Vec::with_capacity(len + 2 * pad);
    for i in (1..=pad).rev() {
        indices.push(i as u32);
    }
    indices.extend(0..len as u32);
    for i in (len - 1 - pad..len - 1).rev() {
        indices.push(i as u32);
    }
    let idx = Tensor::from_vec(indices, len + 2 * pad, wave.device())?;
    Ok(wave.index_select(&idx, 0)?)
}

/// 对齐 `module/mel_processing.py::spectrogram_torch`（center=False + 反射填充）。
///
/// 输入 32 kHz 波形 `(N,)`，输出 `(1, 1025, T)`。
pub fn linear_spectrogram(
    wave: &Tensor,
    n_fft: usize,
    hop: usize,
    win: usize,
    device: &Device,
) -> Result<Tensor> {
    let pad = (n_fft - hop) / 2;
    let padded = reflect_pad(wave, pad)?.unsqueeze(0)?; // (1, N)
    let frames = crate::utils::audio_utils::extract_frames(&padded, win, hop)?;
    // torch.hann_window 默认 periodic=True：w[i] = 0.5 - 0.5*cos(2πi/win)
    let window: Vec<f32> = (0..win)
        .map(|i| {
            let n = win as f64;
            0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / n).cos()
        })
        .map(|v| v as f32)
        .collect();
    let window = Tensor::from_vec(window, win, device)?;
    let frames = frames.broadcast_mul(&window)?;
    let frames = if n_fft > win {
        frames.pad_with_zeros(candle_core::D::Minus1, 0, n_fft - win)?
    } else {
        frames
    };
    // apply_stft 返回功率谱 (bs, n_frames, freq)
    let power = crate::utils::audio_utils::apply_stft(&frames)?;
    let eps = Tensor::new(1e-8_f32, device)?.to_dtype(power.dtype())?;
    let mag = (power.broadcast_add(&eps)?).sqrt()?;
    Ok(mag.transpose(1, 2)?.contiguous()?)
}

/// 从模型目录读取 `is_v2pro`（决定是否需要 `sv_emb`）。
pub fn is_v2pro(dir: impl AsRef<Path>) -> Result<bool> {
    #[derive(serde::Deserialize)]
    struct M {
        #[serde(default)]
        is_v2pro: bool,
    }
    let path = dir.as_ref().join("manifest.json");
    let m: M = serde_json::from_reader(std::io::BufReader::new(std::fs::File::open(path)?))?;
    Ok(m.is_v2pro)
}

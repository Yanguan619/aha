//! chinese-hubert-base（HuBERT，wav2vec2 结构）的 candle 实现。
//!
//! 与 transformers 4.51 的 `HubertModel` 严格对齐：
//!
//! ```text
//! 7 × Conv1d(stride 5/2/2/2/2/2/2) → 只有第 0 层带 GroupNorm(groups=C)
//!   → 逐层 GELU
//!   → feature_projection: LayerNorm(512) + Linear(512→768)
//!   → + conv_pos(kernel 128, groups 16, pad 64) → GELU
//!   → encoder.layer_norm          ← 注意：在进 transformer 之前，不是最后
//!   → 12 × [attn → +res → LN → FFN → +res → LN]   （post-norm）
//! ```
//!
//! 采样率必须是 16 kHz；输出帧率 = 输入采样点 / 320。

use std::path::Path;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{
    Activation, Conv1d, Conv1dConfig, GroupNorm, LayerNorm, Linear, VarBuilder, conv1d,
    conv1d_no_bias, group_norm, layer_norm,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct HubertConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub layer_norm_eps: f64,
    pub conv_dim: Vec<usize>,
    pub conv_kernel: Vec<usize>,
    pub conv_stride: Vec<usize>,
    pub num_conv_pos_embeddings: usize,
    pub num_conv_pos_embedding_groups: usize,
    #[serde(default)]
    pub conv_bias: bool,
    #[serde(default = "yes")]
    pub feat_proj_layer_norm: bool,
    /// `"group"`（默认，与官方 config 一致）或 `"layer"`
    #[serde(default = "group_norm_mode")]
    pub feat_extract_norm: String,
}

fn yes() -> bool {
    true
}

fn group_norm_mode() -> String {
    "group".to_string()
}

impl HubertConfig {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        serde_json::from_reader(std::io::BufReader::new(
            std::fs::File::open(path).with_context(|| format!("打开 {}", path.display()))?,
        ))
        .with_context(|| format!("解析 {}", path.display()))
    }
}

struct ConvLayer {
    conv: Conv1d,
    norm: Option<GroupNorm>,
    /// `feat_extract_norm == "layer"` 时的 LayerNorm（权重里通常只有第 0 层）
    layer_norm: Option<LayerNorm>,
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (b, s, _) = x.dims3()?;
        let shape = (b, s, self.heads, self.head_dim);
        // CUDA 的 matmul 只接受连续张量，transpose 后必须 contiguous
        let q = self.q.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let k = self.k.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = self.v.forward(x)?.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = probs.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, s, self.heads * self.head_dim))?;
        Ok(self.o.forward(&out)?)
    }
}

struct EncoderLayer {
    attn: Attention,
    ln1: LayerNorm,
    ff1: Linear,
    ff2: Linear,
    ln2: LayerNorm,
}

impl EncoderLayer {
    fn new(cfg: &HubertConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            attn: Attention {
                q: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attention.q_proj"))?,
                k: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attention.k_proj"))?,
                v: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attention.v_proj"))?,
                o: candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attention.out_proj"))?,
                heads: cfg.num_attention_heads,
                head_dim: cfg.hidden_size / cfg.num_attention_heads,
            },
            ln1: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm"))?,
            ff1: candle_nn::linear(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("feed_forward.intermediate_dense"),
            )?,
            ff2: candle_nn::linear(
                cfg.intermediate_size,
                cfg.hidden_size,
                vb.pp("feed_forward.output_dense"),
            )?,
            ln2: layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("final_layer_norm"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let attn = self.attn.forward(x)?;
        let x = self.ln1.forward(&(x + attn)?)?;
        let ff = Activation::Gelu.forward(&self.ff1.forward(&x)?)?;
        let ff = self.ff2.forward(&ff)?;
        Ok(self.ln2.forward(&(&x + ff)?)?)
    }
}

pub struct Hubert {
    conv_layers: Vec<ConvLayer>,
    proj_ln: Option<LayerNorm>,
    proj: Linear,
    pos_conv: Conv1d,
    encoder_ln: LayerNorm,
    layers: Vec<EncoderLayer>,
    device: Device,
    downsample: usize,
}

impl Hubert {
    pub fn load(dir: impl AsRef<Path>, device: &Device, dtype: DType) -> Result<Self> {
        let dir = dir.as_ref();
        let cfg = HubertConfig::load(dir.join("hubert_config.json"))?;
        let tensors = candle_core::safetensors::load(dir.join("hubert.safetensors"), device)?;
        let vb = VarBuilder::from_tensors(tensors, dtype, device);

        let conv_dim = &cfg.conv_dim;
        let group_mode = cfg.feat_extract_norm == "group";

        let mut conv_layers = Vec::with_capacity(conv_dim.len());
        for i in 0..conv_dim.len() {
            let in_c = if i == 0 { 1 } else { conv_dim[i - 1] };
            let conv_cfg = Conv1dConfig {
                stride: cfg.conv_stride[i],
                ..Default::default()
            };
            let conv_vb = vb.pp("feature_extractor.conv_layers").pp(i);
            let conv = if cfg.conv_bias {
                conv1d(in_c, conv_dim[i], cfg.conv_kernel[i], conv_cfg, conv_vb.pp("conv"))?
            } else {
                conv1d_no_bias(in_c, conv_dim[i], cfg.conv_kernel[i], conv_cfg, conv_vb.pp("conv"))?
            };
            // transformers 4.51: feat_extract_norm=="group" 时只有第 0 层是 GroupNorm，
            // 其余层是纯 Conv+激活（这正是这份权重里只有一层 layer_norm 的原因）
            let (norm, layer_norm) = if group_mode && i == 0 {
                (
                    Some(group_norm(
                        conv_dim[i],
                        conv_dim[i],
                        cfg.layer_norm_eps,
                        conv_vb.pp("layer_norm"),
                    )?),
                    None,
                )
            } else if !group_mode {
                (
                    None,
                    Some(layer_norm(
                        conv_dim[i],
                        cfg.layer_norm_eps,
                        conv_vb.pp("layer_norm"),
                    )?),
                )
            } else {
                (None, None)
            };
            conv_layers.push(ConvLayer {
                conv,
                norm,
                layer_norm,
            });
        }

        let pos_cfg = Conv1dConfig {
            padding: cfg.num_conv_pos_embeddings / 2,
            groups: cfg.num_conv_pos_embedding_groups,
            ..Default::default()
        };
        let downsample = cfg.conv_stride.iter().product();

        Ok(Self {
            conv_layers,
            proj_ln: if cfg.feat_proj_layer_norm {
                Some(layer_norm(
                    cfg.conv_dim[cfg.conv_dim.len() - 1],
                    cfg.layer_norm_eps,
                    vb.pp("feature_projection.layer_norm"),
                )?)
            } else {
                None
            },
            proj: candle_nn::linear(
                cfg.conv_dim[cfg.conv_dim.len() - 1],
                cfg.hidden_size,
                vb.pp("feature_projection.projection"),
            )?,
            pos_conv: conv1d(
                cfg.hidden_size,
                cfg.hidden_size,
                cfg.num_conv_pos_embeddings,
                pos_cfg,
                vb.pp("encoder.pos_conv_embed.conv"),
            )?,
            encoder_ln: layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("encoder.layer_norm"),
            )?,
            layers: (0..cfg.num_hidden_layers)
                .map(|i| EncoderLayer::new(&cfg, vb.pp("encoder.layers").pp(i)))
                .collect::<Result<Vec<_>>>()?,
            device: device.clone(),
            downsample,
        })
    }

    pub fn downsample(&self) -> usize {
        self.downsample
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
            let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
            let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("    [hubert] {label:<14} dims={:?} mean={mean:+.4} std={std:.4} min={min:+.3} max={max:+.3}", x.dims());
        }
        Ok(())
    }

    /// 输入 16 kHz 单声道波形 `(1, samples)`，输出 `(1, frames, hidden)`。
    pub fn forward(&self, wav: &Tensor) -> Result<Tensor> {
        // Conv1d 需要 (B, C, T)
        let mut x = match wav.rank() {
            2 => wav.unsqueeze(1)?,
            3 => wav.clone(),
            r => anyhow::bail!("HuBERT 输入应为 (B, T) 或 (B, 1, T)，实际 rank={r}"),
        };
        for layer in &self.conv_layers {
            x = layer.conv.forward(&x)?;
            if let Some(norm) = &layer.norm {
                x = norm.forward(&x)?;
            }
            if let Some(ln) = &layer.layer_norm {
                x = ln.forward(&x.transpose(1, 2)?.contiguous()?)?.transpose(1, 2)?;
            }
            x = Activation::Gelu.forward(&x)?;
        }

        self.log_stage("conv stack", &x)?;
        // (1, C, T') -> (1, T', C)
        let x = x.transpose(1, 2)?.contiguous()?;
        let x = match &self.proj_ln {
            Some(ln) => ln.forward(&x)?,
            None => x,
        };
        let hidden = self.proj.forward(&x)?;
        self.log_stage("feat_proj", &hidden)?;

        // 位置卷积（在通道维上跑 Conv1d）
        let seq_len = hidden.dim(1)?;
        let pos_in = hidden.transpose(1, 2)?.contiguous()?;
        let pos = self.pos_conv.forward(&pos_in)?;
        let pos = Activation::Gelu.forward(&pos)?;
        let pos = pos.narrow(2, 0, seq_len)?.transpose(1, 2)?.contiguous()?;
        self.log_stage("pos_conv", &pos)?;

        let mut hidden = self.encoder_ln.forward(&(&hidden + pos)?)?;
        self.log_stage("pre_layers(LN)", &hidden)?;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden)?;
            if matches!(i, 0 | 1 | 5 | 11) || i + 1 == self.layers.len() {
                self.log_stage(&format!("layer{i}"), &hidden)?;
            }
        }
        Ok(hidden)
    }

    /// 便捷入口：`&[f32]` 16 kHz 波形 → `(frames, hidden)`。
    pub fn forward_pcm(&self, samples: &[f32]) -> Result<Tensor> {
        let wav = Tensor::from_vec(samples.to_vec(), (1, samples.len()), &self.device)?;
        let out = self.forward(&wav)?;
        Ok(out.squeeze(0)?)
    }
}

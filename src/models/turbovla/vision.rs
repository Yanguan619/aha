//! DINOv3 ViT-B/16（TurboVLA 的视觉编码器）。
//!
//! 和 candle-transformers 的 `dinov2` 的区别：**RoPE**（不是绝对位置）、4 个 register token、
//! 每个 block 的 LayerScale。这里按 HF `DINOv3ViTModel` 的 forward 实现，只返回 patch tokens
//! （前面的 cls/register 被 TurboVLA 丢掉）。

use std::f32::consts::PI;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, VarBuilder};
use super::linear::{linear, linear_no_bias, Linear};

use super::config::DinoConfig;

struct Rope {
    cos: Tensor, // (P, head_dim) f32
    sin: Tensor,
}

impl Rope {
    fn new(cfg: &DinoConfig, device: &Device) -> Result<Self> {
        let gh = cfg.image_size / cfg.patch_size;
        let gw = gh;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let nfreq = head_dim / 4;
        // inv_freq = 1 / base ** arange(0, 1, 4 / head_dim)
        let inv: Vec<f32> = (0..nfreq)
            .map(|i| cfg.rope_theta.powf(-(4.0 * i as f32) / head_dim as f32))
            .collect();
        let mut angles = Vec::with_capacity(gh * gw * head_dim);
        for h in 0..gh {
            for w in 0..gw {
                // patch 中心坐标，归一化到 [-1, 1]
                let y = 2.0 * ((h as f32 + 0.5) / gh as f32) - 1.0;
                let x = 2.0 * ((w as f32 + 0.5) / gw as f32) - 1.0;
                let mut base = Vec::with_capacity(2 * nfreq);
                for c in [y, x] {
                    for &f in &inv {
                        base.push(2.0 * PI * c * f);
                    }
                }
                // tile(2)：y,x 各 16 个，再重复一遍 → 64
                angles.extend_from_slice(&base);
                angles.extend_from_slice(&base);
            }
        }
        let cos: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
        let sin: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
        Ok(Self {
            cos: Tensor::from_vec(cos, (gh * gw, head_dim), device)?,
            sin: Tensor::from_vec(sin, (gh * gw, head_dim), device)?,
        })
    }

    /// 转成 `candle_nn::rotary_emb::rope` 要的形状：`(1, 1, S, head_dim/2)`。
    ///
    /// 前缀（cls + register）位置填 `(cos=1, sin=0)`，等价于恒等变换——这样 RoPE
    /// 可以**整段一次算完**，不用 `narrow → 乘 → rotate_half → 乘 → 加 → cat` 那 7 个算子。
    fn prepared(&self, dtype: DType, num_prefix: usize) -> Result<(Tensor, Tensor)> {
        let half = self.cos.dim(1)? / 2;
        let dev = self.cos.device();
        // `Rope` 里存的是"半维重复两遍"的布局，`rope` 只需要前一半。
        let cos = self.cos.narrow(1, 0, half)?.to_dtype(dtype)?.contiguous()?;
        let sin = self.sin.narrow(1, 0, half)?.to_dtype(dtype)?.contiguous()?;
        let one = Tensor::ones((num_prefix, half), dtype, dev)?;
        let zero = Tensor::zeros((num_prefix, half), dtype, dev)?;
        // 2 维形式 `(S, head_dim/2)`：rope 接受 2D，batch/head 维度在 kernel 里自己广播。
        let cos = Tensor::cat(&[&one, &cos], 0)?;
        let sin = Tensor::cat(&[&zero, &sin], 0)?;
        Ok((cos, sin))
    }
}

struct Attention {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl Attention {
    fn load(cfg: &DinoConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let head_dim = h / heads;
        let vb_a = vb.pp("attention");
        let q = if cfg.query_bias {
            linear(h, h, vb_a.pp("q_proj"))?
        } else {
            linear_no_bias(h, h, vb_a.pp("q_proj"))?
        };
        let k = if cfg.key_bias {
            linear(h, h, vb_a.pp("k_proj"))?
        } else {
            linear_no_bias(h, h, vb_a.pp("k_proj"))?
        };
        let v = if cfg.value_bias {
            linear(h, h, vb_a.pp("v_proj"))?
        } else {
            linear_no_bias(h, h, vb_a.pp("v_proj"))?
        };
        let o = if cfg.proj_bias {
            linear(h, h, vb_a.pp("o_proj"))?
        } else {
            linear_no_bias(h, h, vb_a.pp("o_proj"))?
        };
        Ok(Self {
            q,
            k,
            v,
            o,
            heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let (b, s, _) = xs.dims3()?;
        let dtype = xs.dtype();
        let q = self
            .q
            .forward(xs)?
            .reshape((b, s, self.heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = self
            .k
            .forward(xs)?
            .reshape((b, s, self.heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = self
            .v
            .forward(xs)?
            .reshape((b, s, self.heads, self.head_dim))?
            .transpose(1, 2)?;

        // RoPE 一次算完（融合 kernel）。前缀位置的 cos/sin 是 (1, 0)，等价于不动。
        let q = candle_nn::rotary_emb::rope(&q.contiguous()?, cos, sin)?;
        let k = candle_nn::rotary_emb::rope(&k.contiguous()?, cos, sin)?;

        let attn = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let attn = (attn * self.scale)?;
        // softmax：candle 的 bf16 kernel 内部就是 f32 累加，原来那趟
        // `to_dtype(F32) → softmax → to_dtype(bf16)` 纯属多搬两遍内存
        // （nsys：cast_bf16_f32 + cast_f32_bf16 每帧 1.17ms）。
        // 需要和旧口径逐位对齐时用 TURBOVLA_SOFTMAX_F32=1 退回。
        let attn = if dtype == DType::F32 || std::env::var("TURBOVLA_SOFTMAX_F32").is_ok() {
            candle_nn::ops::softmax_last_dim(&attn.to_dtype(DType::F32)?)?.to_dtype(dtype)?
        } else {
            candle_nn::ops::softmax_last_dim(&attn)?
        };
        // v 必须 contiguous：cublas 只接受「最后两维是 dense 行」的布局，
        // `(B,S,H,D).transpose(1,2)` 出来的 stride 两条分支都不满足，
        // CUDA 上会直接报 MatMulNonContiguous（CPU 能跑，所以只在 GPU 上暴露）。
        let v = v.contiguous()?;
        let out = attn.matmul(&v)?;
        let out = out.transpose(1, 2)?.reshape((b, s, self.heads * self.head_dim))?;
        self.o.forward(&out)
    }
}

struct Mlp {
    up: Linear,
    down: Linear,
}

impl Mlp {
    fn load(cfg: &DinoConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let vb_m = vb.pp("mlp");
        Ok(Self {
            up: if cfg.mlp_bias {
                linear(h, i, vb_m.pp("up_proj"))?
            } else {
                linear_no_bias(h, i, vb_m.pp("up_proj"))?
            },
            down: if cfg.mlp_bias {
                linear(i, h, vb_m.pp("down_proj"))?
            } else {
                linear_no_bias(i, h, vb_m.pp("down_proj"))?
            },
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = Activation::Gelu.forward(&self.up.forward(xs)?)?;
        self.down.forward(&x)
    }
}

struct Layer {
    norm1: LayerNorm,
    attn: Attention,
    ls1: Tensor,
    norm2: LayerNorm,
    mlp: Mlp,
    ls2: Tensor,
}

impl Layer {
    fn load(cfg: &DinoConfig, vb: VarBuilder) -> Result<Self> {
        let eps = cfg.layer_norm_eps;
        let h = cfg.hidden_size;
        Ok(Self {
            norm1: candle_nn::layer_norm(h, LayerNormConfig { eps, ..Default::default() }, vb.pp("norm1"))?,
            attn: Attention::load(cfg, vb.clone())?,
            ls1: vb.get(h, "layer_scale1.lambda1")?,
            norm2: candle_nn::layer_norm(h, LayerNormConfig { eps, ..Default::default() }, vb.pp("norm2"))?,
            mlp: Mlp::load(cfg, vb.clone())?,
            ls2: vb.get(h, "layer_scale2.lambda1")?,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let residual = xs;
        let h = self.norm1.forward(xs)?;
        let h = self.attn.forward(&h, cos, sin)?;
        let h = h.broadcast_mul(&self.ls1)?;
        let h = (h + residual)?;

        let residual = &h;
        let h = self.norm2.forward(&h)?;
        let h = self.mlp.forward(&h)?;
        let h = h.broadcast_mul(&self.ls2)?;
        h + residual
    }
}

/// DINOv3 ViT backbone。`load` 的 `vb` 应指向 `vision_encoder.backbone`。
pub struct DinoV3 {
    patch: Conv2d,
    cls: Tensor,
    registers: Option<Tensor>,
    layers: Vec<Layer>,
    norm: LayerNorm,
    rope: Rope,
    num_prefix: usize,
    num_patches: usize,
    hidden: usize,
}

impl DinoV3 {
    pub fn load(cfg: &DinoConfig, vb: VarBuilder, device: &Device) -> Result<Self> {
        let h = cfg.hidden_size;
        let gh = cfg.image_size / cfg.patch_size;
        let num_patches = gh * gh;
        let vb_e = vb.pp("embeddings");
        let patch = candle_nn::conv2d(
            cfg.num_channels,
            h,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            },
            vb_e.pp("patch_embeddings"),
        )?;
        let cls = vb_e.get((1, 1, h), "cls_token")?;
        let registers = if cfg.num_register_tokens > 0 {
            Some(vb_e.get((1, cfg.num_register_tokens, h), "register_tokens")?)
        } else {
            None
        };
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Layer::load(cfg, vb.pp(format!("layer.{i}")))?);
        }
        let norm = candle_nn::layer_norm(
            h,
            LayerNormConfig {
                eps: cfg.layer_norm_eps,
                ..Default::default()
            },
            vb.pp("norm"),
        )?;
        let num_prefix = 1 + cfg.num_register_tokens;
        Ok(Self {
            patch,
            cls,
            registers,
            layers,
            norm,
            rope: Rope::new(cfg, device)?,
            num_prefix,
            num_patches,
            hidden: h,
        })
    }

    /// 输入 `(B, 3, H, W)`，返回 `(B, num_patches, hidden)`（丢掉 cls/register）。
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let (b, _, _, _) = pixel_values.dims4()?;
        let x = self.patch.forward(pixel_values)?; // (B, H, gh, gw)
        let x = x.flatten(2, 3)?.transpose(1, 2)?; // (B, P, H)
        let prefix = match &self.registers {
            Some(r) => {
                let r = r.broadcast_as((b, r.dim(1)?, self.hidden))?;
                Tensor::cat(&[&self.cls.broadcast_as((b, 1, self.hidden))?, &r], 1)?
            }
            None => self.cls.broadcast_as((b, 1, self.hidden))?,
        };
        let mut xs = Tensor::cat(&[&prefix, &x], 1)?; // (B, 1+R+P, H)
        debug_assert_eq!(xs.dim(1)?, self.num_prefix + self.num_patches);
        // cos/sin 只按 dtype 转一次，别每层重算
        let (cos, sin) = self.rope.prepared(xs.dtype(), self.num_prefix)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &cos, &sin)?;
        }
        let xs = self.norm.forward(&xs)?;
        xs.narrow(1, self.num_prefix, self.num_patches)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rope_shape_matches_head_dim() {
        let cfg = DinoConfig {
            hidden_size: 768,
            num_attention_heads: 12,
            num_hidden_layers: 1,
            intermediate_size: 3072,
            patch_size: 16,
            image_size: 256,
            num_register_tokens: 4,
            layer_norm_eps: 1e-5,
            rope_theta: 100.0,
            num_channels: 3,
            mlp_bias: true,
            query_bias: true,
            key_bias: false,
            value_bias: true,
            proj_bias: true,
            use_gated_mlp: false,
        };
        let dev = Device::Cpu;
        let rope = Rope::new(&cfg, &dev).unwrap();
        assert_eq!(rope.cos.dims(), &[256, 64]);
        assert_eq!(rope.sin.dims(), &[256, 64]);
    }
}

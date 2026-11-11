//! IndexTTS 2.5 的情绪条件编码器。
//!
//! 结构（对照上游 `indextts/gpt/model_v2_5.py` + `conformer_encoder.py` + `perceiver.py`）：
//!
//! ```text
//! w2v-bert 特征 (b, T, 1024)
//!   └─ ConformerEncoder(conv2d2 下采样 / 相对位置 / 4 层 / 512 维)
//!        ├─ Conv2d(1→512, k3, s2) → ReLU → Linear(512*511 → 512) → x*sqrt(512)
//!        └─ 每层: x += attn(norm_mha(x)); x += conv(norm_conv(x)); x += ff(norm_ff(x)); x = norm_final(x)
//!   └─ PerceiverResampler(dim=1024, ctx=512, 1 latent, depth=2, heads=4, ff_mult=2)
//!        └─ RMSNorm → (b, 1, 1024)
//!   └─ emovec_layer(1024→1280) → emo_layer(1280→1280) → emo_vec (b, 1280)
//! ```

use anyhow::Result;
use candle_core::{D, Tensor};
use candle_nn::{Conv1d, Conv2d, LayerNorm, Linear, Module, VarBuilder, linear};

use crate::models::common::modules::{get_conv1d, get_conv2d, get_layer_norm};

/// 相对位置注意力的正弦位置表（和上游 `PositionalEncoding` 完全一致的公式）
fn sinusoidal_pe(max_len: usize, d_model: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut data = vec![0f32; max_len * d_model];
    let log_timescale = -(10000f64).ln() / d_model as f64;
    for pos in 0..max_len {
        for i in 0..d_model / 2 {
            let angle = pos as f64 * (log_timescale * (2 * i) as f64).exp();
            data[pos * d_model + 2 * i] = angle.sin() as f32;
            data[pos * d_model + 2 * i + 1] = angle.cos() as f32;
        }
    }
    Ok(Tensor::from_vec(data, (1, max_len, d_model), device)?)
}

/// 相对位置多头注意力（wenet/funasr 版，注意本实现**没有** rel_shift）
struct RelPosAttention {
    h: usize,
    d_k: usize,
    linear_q: Linear,
    linear_k: Linear,
    linear_v: Linear,
    linear_out: Linear,
    linear_pos: Linear,
    pos_bias_u: Tensor,
    pos_bias_v: Tensor,
}

impl RelPosAttention {
    fn new(vb: &VarBuilder, n_feat: usize, n_head: usize) -> Result<Self> {
        let d_k = n_feat / n_head;
        let pos_bias_u = vb
            .get((n_head, d_k), "pos_bias_u")?
            .reshape((1, n_head, 1, d_k))?;
        let pos_bias_v = vb
            .get((n_head, d_k), "pos_bias_v")?
            .reshape((1, n_head, 1, d_k))?;
        Ok(Self {
            h: n_head,
            d_k,
            linear_q: linear(n_feat, n_feat, vb.pp("linear_q"))?,
            linear_k: linear(n_feat, n_feat, vb.pp("linear_k"))?,
            linear_v: linear(n_feat, n_feat, vb.pp("linear_v"))?,
            linear_out: linear(n_feat, n_feat, vb.pp("linear_out"))?,
            linear_pos: linear_no_bias(n_feat, n_feat, vb.pp("linear_pos"))?,
            pos_bias_u,
            pos_bias_v,
        })
    }

    /// `xs`: (b, t, d)；`pos`: (1, t, d)
    fn forward(&self, xs: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let (b, t, _) = xs.dims3()?;
        let (h, d_k) = (self.h, self.d_k);
        let split = |t_: Tensor| -> Result<Tensor> {
            Ok(t_.reshape((b, t, h, d_k))?.transpose(1, 2)?.contiguous()?)
        };
        let q = split(self.linear_q.forward(xs)?)?; // (b,h,t,d_k)
        let k = split(self.linear_k.forward(xs)?)?;
        let v = split(self.linear_v.forward(xs)?)?;
        let kt = k.transpose(2, 3)?.contiguous()?; // (b,h,d_k,t)
        let vt = v;

        let (pb, pt, _) = pos.dims3()?;
        let p = self
            .linear_pos
            .forward(pos)?
            .reshape((pb, pt, h, d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let pt = p.transpose(2, 3)?.contiguous()?; // (1,h,d_k,t)

        let qu = q.broadcast_add(&self.pos_bias_u)?; // (b,h,t,d_k)
        let qv = q.broadcast_add(&self.pos_bias_v)?;

        let matrix_ac = qu.matmul(&kt)?; // (b,h,t,t)
        let matrix_bd = qv.matmul(&pt)?; // (b,h,t,t)（pos 只有 1 个 batch，靠广播）
        let scores = ((matrix_ac + matrix_bd)? / f64::sqrt(d_k as f64))?;
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&vt)?; // (b,h,t,d_k)
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, h * d_k))?;
        Ok(self.linear_out.forward(&out)?)
    }
}

fn linear_no_bias(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((out_dim, in_dim), "weight")?;
    Ok(Linear::new(weight, None))
}

/// Conformer 的卷积模块（GLU + depthwise + LayerNorm + SiLU + pointwise）
struct ConvModule {
    pointwise_conv1: Conv1d,
    depthwise_conv: Conv1d,
    norm: LayerNorm,
    pointwise_conv2: Conv1d,
}

impl ConvModule {
    fn new(vb: &VarBuilder, channels: usize, kernel_size: usize) -> Result<Self> {
        Ok(Self {
            pointwise_conv1: get_conv1d(
                vb.pp("pointwise_conv1"),
                channels,
                2 * channels,
                1,
                0,
                1,
                1,
                1,
                true,
            )?,
            depthwise_conv: get_conv1d(
                vb.pp("depthwise_conv"),
                channels,
                channels,
                kernel_size,
                (kernel_size - 1) / 2,
                1,
                1,
                channels,
                true,
            )?,
            norm: get_layer_norm(vb.pp("norm"), 1e-5, channels, true)?,
            pointwise_conv2: get_conv1d(
                vb.pp("pointwise_conv2"),
                channels,
                channels,
                1,
                0,
                1,
                1,
                1,
                true,
            )?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.transpose(1, 2)?; // (b,C,t)
        let xs = self.pointwise_conv1.forward(&xs)?; // (b,2C,t)
        let parts = xs.chunk(2, 1)?;
        // F.glu: 前半 * sigmoid(后半)
        let xs = (candle_nn::ops::sigmoid(&parts[0])? * &parts[1])?;
        let xs = self.depthwise_conv.forward(&xs)?;
        let xs = xs.transpose(1, 2)?;
        let xs = self.norm.forward(&xs)?;
        let xs = xs.silu()?;
        let xs = xs.transpose(1, 2)?;
        let xs = self.pointwise_conv2.forward(&xs)?;
        Ok(xs.transpose(1, 2)?.contiguous()?)
    }
}

struct FeedForward {
    w_1: Linear,
    w_2: Linear,
}

impl FeedForward {
    fn new(vb: &VarBuilder, idim: usize, hidden: usize) -> Result<Self> {
        Ok(Self {
            w_1: linear(idim, hidden, vb.pp("w_1"))?,
            w_2: linear(hidden, idim, vb.pp("w_2"))?,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.w_1.forward(xs)?.silu()?;
        Ok(self.w_2.forward(&xs)?)
    }
}

struct ConformerLayer {
    norm_mha: LayerNorm,
    attn: RelPosAttention,
    norm_conv: LayerNorm,
    conv: ConvModule,
    norm_ff: LayerNorm,
    ff: FeedForward,
    norm_final: LayerNorm,
}

impl ConformerLayer {
    fn new(vb: &VarBuilder, dim: usize, heads: usize, lin_units: usize) -> Result<Self> {
        Ok(Self {
            norm_mha: get_layer_norm(vb.pp("norm_mha"), 1e-5, dim, true)?,
            attn: RelPosAttention::new(&vb.pp("self_attn"), dim, heads)?,
            norm_conv: get_layer_norm(vb.pp("norm_conv"), 1e-5, dim, true)?,
            conv: ConvModule::new(&vb.pp("conv_module"), dim, 15)?,
            norm_ff: get_layer_norm(vb.pp("norm_ff"), 1e-5, dim, true)?,
            ff: FeedForward::new(&vb.pp("feed_forward"), dim, lin_units)?,
            norm_final: get_layer_norm(vb.pp("norm_final"), 1e-5, dim, true)?,
        })
    }

    fn forward(&self, xs: &Tensor, pos: &Tensor) -> Result<Tensor> {
        let x = self.norm_mha.forward(xs)?;
        let x = self.attn.forward(&x, pos)?;
        let xs = (xs + x)?;

        let x = self.norm_conv.forward(&xs)?;
        let x = self.conv.forward(&x)?;
        let xs = (xs + x)?;

        let x = self.norm_ff.forward(&xs)?;
        let x = self.ff.forward(&x)?;
        let xs = (xs + x)?;

        Ok(self.norm_final.forward(&xs)?)
    }
}

/// conv2d2 下采样 + 4 层 conformer + after_norm
pub struct ConformerEncoder {
    conv: Conv2d,
    out: Linear,
    pe: Tensor,
    layers: Vec<ConformerLayer>,
    after_norm: LayerNorm,
    dim: usize,
}

impl ConformerEncoder {
    /// `input_size` 是特征维（1024 = w2v-bert hidden）
    pub fn new(
        vb: &VarBuilder,
        input_size: usize,
        dim: usize,
        heads: usize,
        lin_units: usize,
        num_blocks: usize,
    ) -> Result<Self> {
        let embed = vb.pp("embed");
        let conv = get_conv2d(embed.pp("conv").pp(0), 1, dim, 3, 0, 2, 1, 1, true)?;
        let freq_out = input_size.saturating_sub(1) / 2;
        let out = linear(dim * freq_out, dim, embed.pp("out").pp(0))?;
        let pe = sinusoidal_pe(5000, dim, vb.device())?;

        let vb_layers = vb.pp("encoders");
        let mut layers = Vec::with_capacity(num_blocks);
        for i in 0..num_blocks {
            layers.push(ConformerLayer::new(
                &vb_layers.pp(i),
                dim,
                heads,
                lin_units,
            )?);
        }
        let after_norm = get_layer_norm(vb.pp("after_norm"), 1e-5, dim, true)?;
        Ok(Self {
            conv,
            out,
            pe,
            layers,
            after_norm,
            dim,
        })
    }

    /// `feats`: (b, T, input_size) -> (b, T', dim)
    pub fn forward(&self, feats: &Tensor) -> Result<Tensor> {
        let xs = feats.unsqueeze(1)?; // (b,1,T,F)
        let xs = self.conv.forward(&xs)?; // (b,C,T',F')
        let (b, c, t, f) = xs.dims4()?;
        // 和上游一致：transpose(1,2).view(b,t,c*f) —— 通道在前、频率在后
        let xs = xs.transpose(1, 2)?.contiguous()?.reshape((b, t, c * f))?;
        let xs = self.out.forward(&xs)?;
        let xs = (xs * f64::sqrt(self.dim as f64))?;
        let pos = self.pe.narrow(1, 0, t)?; // (1,t,dim)
        let mut xs = xs;
        for layer in &self.layers {
            xs = layer.forward(&xs, &pos)?;
        }
        Ok(self.after_norm.forward(&xs)?)
    }
}

/// perceiver 里的一个 cross-attention block（含 queries 拼进 context 的细节）
struct PerceiverAttention {
    h: usize,
    d_k: usize,
    to_q: Linear,
    to_kv: Linear,
    to_out: Linear,
}

impl PerceiverAttention {
    fn new(vb: &VarBuilder, dim: usize, heads: usize) -> Result<Self> {
        // 从权重形状反推 dim_inner（= heads * dim_head），避免把 head 数写死
        let (dim_inner, _) = vb.get_unchecked("to_q.weight")?.dims2()?;
        Ok(Self {
            h: heads,
            d_k: dim_inner / heads,
            to_q: linear_no_bias(dim, dim_inner, vb.pp("to_q"))?,
            to_kv: linear_no_bias(dim, dim_inner * 2, vb.pp("to_kv"))?,
            to_out: linear_no_bias(dim_inner, dim, vb.pp("to_out"))?,
        })
    }

    /// `latents`: (b,1,dim)；`context`: (b,T,dim) —— 按上游把 queries 拼到 context 前面
    fn forward(&self, latents: &Tensor, context: &Tensor) -> Result<Tensor> {
        let (b, n, _) = latents.dims3()?;
        let (h, d_k) = (self.h, self.d_k);
        let kv_input = Tensor::cat(&[latents, context], 1)?; // (b,1+T,dim)
        let (_, m, _) = kv_input.dims3()?;

        let q = self
            .to_q
            .forward(latents)?
            .reshape((b, n, h, d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let kv = self.to_kv.forward(&kv_input)?;
        let kv = kv.chunk(2, 2)?;
        let k = kv[0]
            .reshape((b, m, h, d_k))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = kv[1]
            .reshape((b, m, h, d_k))?
            .transpose(1, 2)?
            .contiguous()?;

        let scale = 1f64 / f64::sqrt(d_k as f64);
        let out = candle_nn::ops::softmax_last_dim(
            &(q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?,
        )?;
        let out = out.matmul(&v)?;
        let out = out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, n, h * d_k))?;
        Ok(self.to_out.forward(&out)?)
    }
}

/// GEGLU + Linear 的前馈
struct PerceiverFeedForward {
    linear_in: Linear,
    linear_out: Linear,
    dim_inner: usize,
}

impl PerceiverFeedForward {
    fn new(vb: &VarBuilder, dim: usize, ff_mult: usize) -> Result<Self> {
        let dim_inner = dim * ff_mult * 2 / 3;
        Ok(Self {
            linear_in: linear(dim, dim_inner * 2, vb.pp(0))?,
            linear_out: linear(dim_inner, dim, vb.pp(2))?,
            dim_inner,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear_in.forward(xs)?;
        let parts = xs.chunk(2, D::Minus1)?;
        // GEGLU: x * gelu(gate)
        let xs = (&parts[0] * parts[1].gelu()?)?;
        let _ = self.dim_inner;
        Ok(self.linear_out.forward(&xs)?)
    }
}

/// perceiver resampler：把变长条件序列压成 `num_latents` 个向量
pub struct PerceiverResampler {
    proj_context: Option<Linear>,
    latents: Tensor,
    attn: Vec<PerceiverAttention>,
    ff: Vec<PerceiverFeedForward>,
    gamma: Tensor,
    dim: usize,
}

impl PerceiverResampler {
    pub fn new(
        vb: &VarBuilder,
        dim: usize,
        dim_context: usize,
        num_latents: usize,
        depth: usize,
        heads: usize,
        ff_mult: usize,
    ) -> Result<Self> {
        let proj_context = if dim_context != dim {
            Some(linear(dim_context, dim, vb.pp("proj_context"))?)
        } else {
            None
        };
        let latents = vb
            .get((num_latents, dim), "latents")?
            .reshape((1, num_latents, dim))?;
        let mut attn = Vec::with_capacity(depth);
        let mut ff = Vec::with_capacity(depth);
        for i in 0..depth {
            let vb_layer = vb.pp("layers").pp(i);
            attn.push(PerceiverAttention::new(&vb_layer.pp(0), dim, heads)?);
            ff.push(PerceiverFeedForward::new(&vb_layer.pp(1), dim, ff_mult)?);
        }
        let gamma = vb.get(dim, "norm.gamma")?;
        Ok(Self {
            proj_context,
            latents,
            attn,
            ff,
            gamma,
            dim,
        })
    }

    /// `context`: (b, T, dim_context) -> (b, num_latents, dim)
    pub fn forward(&self, context: &Tensor) -> Result<Tensor> {
        let (b, _, _) = context.dims3()?;
        let context = match &self.proj_context {
            Some(p) => p.forward(context)?,
            None => context.clone(),
        };
        let mut latents = self
            .latents
            .broadcast_as((b, self.latents.dim(1)?, self.dim))?;
        for (attn, ff) in self.attn.iter().zip(self.ff.iter()) {
            latents = (attn.forward(&latents, &context)? + &latents)?;
            latents = (ff.forward(&latents)? + &latents)?;
        }
        // RMSNorm：F.normalize(x) * sqrt(dim) * gamma
        let norm = latents.sqr()?.sum_keepdim(D::Minus1)?.sqrt()?;
        let latents = latents.broadcast_div(&norm)?;
        let latents = (latents * f64::sqrt(self.dim as f64))?;
        Ok(latents.broadcast_mul(&self.gamma)?)
    }
}

/// 完整的情绪条件链路：conformer → perceiver → emovec_layer → emo_layer
pub struct EmoConditioner {
    conformer: ConformerEncoder,
    perceiver: PerceiverResampler,
    emovec_layer: Linear,
    emo_layer: Linear,
}

impl EmoConditioner {
    pub fn new(
        vb: &VarBuilder,
        input_size: usize,
        cond_dim: usize,
        heads: usize,
        lin_units: usize,
        num_blocks: usize,
        perceiver_dim: usize,
        perceiver_mult: usize,
        model_dim: usize,
    ) -> Result<Self> {
        let conformer = ConformerEncoder::new(
            &vb.pp("emo_conditioning_encoder"),
            input_size,
            cond_dim,
            heads,
            lin_units,
            num_blocks,
        )?;
        let perceiver = PerceiverResampler::new(
            &vb.pp("emo_perceiver_encoder"),
            perceiver_dim,
            cond_dim,
            1,
            2,
            heads,
            perceiver_mult,
        )?;
        let emovec_layer = linear(perceiver_dim, model_dim, vb.pp("emovec_layer"))?;
        let emo_layer = linear(model_dim, model_dim, vb.pp("emo_layer"))?;
        Ok(Self {
            conformer,
            perceiver,
            emovec_layer,
            emo_layer,
        })
    }

    /// `feats`: (b, T, 1024) w2v-bert 特征 -> (b, model_dim) 情绪向量
    pub fn forward(&self, feats: &Tensor) -> Result<Tensor> {
        let xs = self.conformer.forward(feats)?; // (b,T',512)
        let latents = self.perceiver.forward(&xs)?; // (b,1,1024)
        let latents = latents.squeeze(1)?; // (b,1024)
        let emo = self.emovec_layer.forward(&latents)?;
        Ok(self.emo_layer.forward(&emo)?)
    }
}

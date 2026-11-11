//! VITS 的注意力实现（`module/attentions.py` + `module/mrte_model.py`）。
//!
//! 与普通 MHA 的区别：
//!
//! 1. Q/K/V 是 **1×1 Conv1d**（不是 Linear）；
//! 2. `enc_p` 的三个 Encoder 开着 **相对位置注意力**（`emb_rel_k` / `emb_rel_v`，
//!    window=4），scores 要加一项 `scores_local`，输出要加一项相对值的贡献；
//! 3. `FFN` 是两层 Conv1d + ReLU，padding 是 `(k-1)/2` 左 `k/2` 右（same padding）；
//! 4. LayerNorm 在**通道维**上做（等价 PyTorch 的 `LayerNorm` on channel-last）。

use anyhow::Result;
use candle_core::Tensor;
use candle_nn::Module as _;
use candle_nn::{Conv1d, Conv1dConfig, LayerNorm, VarBuilder, conv1d};


/// 诊断用：把张量写成裸 f32 到 `<AHA_GPT_SOVITS_DIR>/ref_dump/rust_<name>.bin`
pub fn dump_tensor(name: &str, t: &Tensor) -> Result<()> {
    if std::env::var("AHA_GSV_DUMP").is_err() {
        return Ok(());
    }
    let Ok(dir) = std::env::var("AHA_GPT_SOVITS_DIR") else {
        return Ok(());
    };
    let dir = std::path::Path::new(&dir).join("ref_dump");
    std::fs::create_dir_all(&dir)?;
    let v: Vec<f32> = t.flatten_all()?.to_vec1()?;
    let mut bytes = Vec::with_capacity(v.len() * 4);
    for x in v {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(dir.join(format!("rust_{name}.bin")), bytes)?;
    Ok(())
}

/// 通道维 LayerNorm：`(B, C, T)` ↔ `(B, T, C)`
pub fn ln_channel_last(x: &Tensor, ln: &LayerNorm) -> Result<Tensor> {
    Ok(ln
        .forward(&x.transpose(1, 2)?.contiguous()?)?
        .transpose(1, 2)?
        .contiguous()?)
}

fn conv1x1(in_c: usize, out_c: usize, vb: VarBuilder) -> Result<Conv1d> {
    Ok(conv1d(in_c, out_c, 1, Conv1dConfig::default(), vb)?)
}

/// `module/modules.py::LayerNorm`：通道维归一化，参数名是 **gamma/beta**
/// （不是 `nn.LayerNorm` 的 weight/bias——按 weight/bias 读，candle 会用默认值静默补上）。
fn vits_layer_norm(channels: usize, eps: f64, vb: VarBuilder) -> Result<LayerNorm> {
    let gamma = vb
        .get(channels, "gamma")
        .or_else(|_| vb.get(channels, "weight"))?;
    if std::env::var("AHA_GSV_DEBUG").is_ok() {
        let mean = gamma.flatten_all()?.mean_all()?.to_scalar::<f32>()?;
        eprintln!("    [ln] gamma mean={mean:+.5} channels={channels}");
    }
    let beta = vb
        .get(channels, "beta")
        .or_else(|_| vb.get(channels, "bias"))?;
    Ok(LayerNorm::new(gamma, beta, eps))
}

/// `MultiHeadAttention`（带可选的相对位置编码）
pub struct RelMultiHeadAttention {
    conv_q: Conv1d,
    conv_k: Conv1d,
    conv_v: Conv1d,
    conv_o: Conv1d,
    emb_rel_k: Option<Tensor>,
    emb_rel_v: Option<Tensor>,
    heads: usize,
    k_channels: usize,
    window_size: Option<usize>,
}

impl RelMultiHeadAttention {
    pub fn new(
        channels: usize,
        out_channels: usize,
        heads: usize,
        window_size: Option<usize>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let k_channels = channels / heads;
        let (emb_rel_k, emb_rel_v) = match window_size {
            Some(w) => (
                Some(vb.get((1, w * 2 + 1, k_channels), "emb_rel_k")?),
                Some(vb.get((1, w * 2 + 1, k_channels), "emb_rel_v")?),
            ),
            None => (None, None),
        };
        Ok(Self {
            conv_q: conv1x1(channels, channels, vb.pp("conv_q"))?,
            conv_k: conv1x1(channels, channels, vb.pp("conv_k"))?,
            conv_v: conv1x1(channels, channels, vb.pp("conv_v"))?,
            conv_o: conv1x1(channels, out_channels, vb.pp("conv_o"))?,
            emb_rel_k,
            emb_rel_v,
            heads,
            k_channels,
            window_size,
        })
    }

    /// `x` 为 query `(B, C, Tq)`，`c` 为 key/value `(B, C, Tk)`，`mask` 为 `(B, Tq, Tk)`（0/1）。
    pub fn forward(&self, x: &Tensor, c: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let q = self.conv_q.forward(x)?;
        let k = self.conv_k.forward(c)?;
        let v = self.conv_v.forward(c)?;

        let out = self.attention(&q, &k, &v, mask)?;
        Ok(self.conv_o.forward(&out)?)
    }

    fn attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, _d, t_t) = q.dims3()?;
        let t_s = k.dim(2)?;
        let h = self.heads;
        let kc = self.k_channels;

        let q = q.reshape((b, h, kc, t_t))?.transpose(2, 3)?.contiguous()?; // (B,H,Tt,Kc)
        let k = k.reshape((b, h, kc, t_s))?.transpose(2, 3)?.contiguous()?;
        let v = v.reshape((b, h, kc, t_s))?.transpose(2, 3)?.contiguous()?;

        let scale = (kc as f64).sqrt();
        let q_scaled = (q / scale)?;
        let mut scores = q_scaled.matmul(&k.transpose(2, 3)?.contiguous()?)?;

        if let (Some(window), Some(emb_k), Some(emb_v)) =
            (self.window_size, &self.emb_rel_k, &self.emb_rel_v)
        {
            anyhow::ensure!(t_s == t_t, "相对位置注意力只支持自注意力");
            let rel_k = get_relative_embeddings(emb_k, window, t_s)?; // (1, 2T-1, Kc)
            // (B,H,T,Kc) x (1,Kc,2T-1) -> (B,H,T,2T-1)，需要广播 batch 维
            let rel_logits = q_scaled.broadcast_matmul(&rel_k.transpose(1, 2)?.contiguous()?)?;
            let scores_local = relative_to_absolute(&rel_logits)?;
            scores = (&scores + scores_local)?;

            // 相对值那一项在 softmax 之后加
            let probs = self.masked_softmax(&scores, mask)?;
            let out = probs.matmul(&v)?;
            let rel_weights = absolute_to_relative(&probs)?; // (B,H,T,2T-1)
            let rel_v = get_relative_embeddings(emb_v, window, t_s)?;
            let rel_out = rel_weights.broadcast_matmul(&rel_v)?;
            let out = (&out + rel_out)?;
            return self.merge_heads(&out, b, t_t);
        }

        let probs = self.masked_softmax(&scores, mask)?;
        let out = probs.matmul(&v)?;
        self.merge_heads(&out, b, t_t)
    }

    fn masked_softmax(&self, scores: &Tensor, mask: Option<&Tensor>) -> Result<Tensor> {
        let scores = match mask {
            None => scores.clone(),
            Some(mask) => {
                // PyTorch: scores.masked_fill(mask == 0, -1e4)
                let mask = mask.broadcast_as(scores.shape())?;
                let keep = mask.gt(0.0)?;
                let on_false = Tensor::new(-1e4f32, scores.device())?
                    .to_dtype(scores.dtype())?
                    .broadcast_as(scores.shape())?;
                keep.where_cond(scores, &on_false)?
            }
        };
        Ok(candle_nn::ops::softmax_last_dim(&scores)?)
    }

    fn merge_heads(&self, out: &Tensor, b: usize, t: usize) -> Result<Tensor> {
        // (B,H,T,Kc) -> (B,T,H*Kc) -> (B,C,T)
        // 注意是 transpose(1,2)（把 head 挪到时间后面），写成 transpose(2,3) 会把
        // 通道和时间轴搅在一起，输出与参考几乎正交。
        Ok(out
            .transpose(1, 2)?
            .contiguous()?
            .reshape((b, t, self.heads * self.k_channels))?
            .transpose(1, 2)?
            .contiguous()?)
    }
}

/// `_get_relative_embeddings`：把 (1, 2w+1, kc) 扩成 (1, 2L-1, kc)
fn get_relative_embeddings(emb: &Tensor, window: usize, length: usize) -> Result<Tensor> {
    let margin = window + 1;
    let pad_length = length.saturating_sub(margin);
    let slice_start = margin.saturating_sub(length);
    let slice_end = slice_start + 2 * length - 1;

    let padded = if pad_length > 0 {
        emb.pad_with_zeros(1, pad_length, pad_length)?
    } else {
        emb.clone()
    };
    Ok(padded.narrow(1, slice_start, slice_end - slice_start)?)
}

/// `_relative_position_to_absolute_position`：(B,H,L,2L-1) -> (B,H,L,L)
fn relative_to_absolute(x: &Tensor) -> Result<Tensor> {
    let (b, h, l, _) = x.dims4()?;
    let x = x.pad_with_zeros(3, 0, 1)?; // (B,H,L,2L)
    let x = x.reshape((b, h, l * 2 * l))?;
    let x = x.pad_with_zeros(2, 0, l - 1)?;
    let x = x.reshape((b, h, l + 1, 2 * l - 1))?;
    let x = x.narrow(2, 0, l)?;
    Ok(x.narrow(3, l - 1, l)?.contiguous()?)
}

/// `_absolute_position_to_relative_position`：(B,H,L,L) -> (B,H,L,2L-1)
fn absolute_to_relative(x: &Tensor) -> Result<Tensor> {
    let (b, h, l, _) = x.dims4()?;
    let x = x.pad_with_zeros(3, 0, l - 1)?;
    let x = x.reshape((b, h, l * l + l * (l - 1)))?;
    let x = x.pad_with_zeros(2, l, 0)?;
    let x = x.reshape((b, h, l, 2 * l))?;
    Ok(x.narrow(3, 1, 2 * l - 1)?.contiguous()?)
}

/// `FFN`：Conv1d → ReLU → Conv1d，same padding
pub struct VitsFfn {
    conv_1: Conv1d,
    conv_2: Conv1d,
    kernel_size: usize,
}

impl VitsFfn {
    pub fn new(
        in_c: usize,
        out_c: usize,
        filter_c: usize,
        kernel_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            conv_1: conv1d(in_c, filter_c, kernel_size, Conv1dConfig::default(), vb.pp("conv_1"))?,
            conv_2: conv1d(
                filter_c,
                out_c,
                kernel_size,
                Conv1dConfig::default(),
                vb.pp("conv_2"),
            )?,
            kernel_size,
        })
    }

    fn same_pad(&self, x: &Tensor) -> Result<Tensor> {
        if self.kernel_size == 1 {
            return Ok(x.clone());
        }
        let pad_l = (self.kernel_size - 1) / 2;
        let pad_r = self.kernel_size / 2;
        Ok(x.pad_with_zeros(2, pad_l, pad_r)?)
    }

    pub fn forward(&self, x: &Tensor, x_mask: &Tensor) -> Result<Tensor> {
        let h = self.conv_1.forward(&self.same_pad(&(x * x_mask)?)?)?;
        let h = h.relu()?;
        let h = self.conv_2.forward(&self.same_pad(&(h * x_mask)?)?)?;
        Ok((h * x_mask)?)
    }

    /// 无 padding 版本（mask 恒为全 1）
    pub fn forward_plain(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.conv_1.forward(&self.same_pad(x)?)?;
        let h = h.relu()?;
        Ok(self.conv_2.forward(&self.same_pad(&h)?)?)
    }
}

/// `attentions.Encoder`（`encoder_ssl` / `encoder_text` / `encoder2`）
pub struct VitsEncoder {
    attn: Vec<RelMultiHeadAttention>,
    ln1: Vec<LayerNorm>,
    ffn: Vec<VitsFfn>,
    ln2: Vec<LayerNorm>,
}

impl VitsEncoder {
    pub fn new(
        hidden: usize,
        filter: usize,
        heads: usize,
        n_layers: usize,
        kernel_size: usize,
        window_size: Option<usize>,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut attn = Vec::with_capacity(n_layers);
        let mut ln1 = Vec::with_capacity(n_layers);
        let mut ffn = Vec::with_capacity(n_layers);
        let mut ln2 = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            let layer = vb.pp(format!("attn_layers.{}", i));
            attn.push(RelMultiHeadAttention::new(
                hidden,
                hidden,
                heads,
                window_size,
                layer,
            )?);
            ln1.push(vits_layer_norm(hidden, eps, vb.pp(format!("norm_layers_1.{}", i)))?);
            ffn.push(VitsFfn::new(
                hidden,
                hidden,
                filter,
                kernel_size,
                vb.pp(format!("ffn_layers.{}", i)),
            )?);
            ln2.push(vits_layer_norm(hidden, eps, vb.pp(format!("norm_layers_2.{}", i)))?);
        }
        Ok(Self {
            attn,
            ln1,
            ffn,
            ln2,
        })
    }

    /// `x`: (B, C, T)，`x_mask`: (B, 1, T)
    pub fn forward(&self, x: &Tensor, x_mask: &Tensor) -> Result<Tensor> {
        let attn_mask = x_mask.unsqueeze(2)?.broadcast_mul(&x_mask.unsqueeze(3)?)?; // (B,1,T,T)
        let attn_mask = attn_mask.squeeze(1)?; // (B,T,T)
        let mut x = (x * x_mask)?;
        for i in 0..self.attn.len() {
            let y = self.attn[i].forward(&x, &x, Some(&attn_mask))?;
            x = ln_channel_last(&(&x + y)?, &self.ln1[i])?;
            let y = self.ffn[i].forward(&x, x_mask)?;
            x = ln_channel_last(&(&x + y)?, &self.ln2[i])?;
        }
        Ok((x * x_mask)?)
    }

    /// 无 padding 的等价版本（推理时 mask 恒为全 1）。
    ///
    /// 省掉 `x * x_mask` 和 `masked_fill`，数值上完全一致（乘 1 / 不屏蔽）。
    pub fn forward_plain(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.attn.len() {
            let y = self.attn[i].forward(&x, &x, None)?;
            x = ln_channel_last(&(&x + y)?, &self.ln1[i])?;
            let y = self.ffn[i].forward_plain(&x)?;
            x = ln_channel_last(&(&x + y)?, &self.ln2[i])?;
        }
        Ok(x)
    }
}

/// `MRTE`：SSL 特征与文本特征做交叉注意力，再加音色向量
pub struct Mrte {
    c_pre: Conv1d,
    text_pre: Conv1d,
    c_post: Conv1d,
    attn: RelMultiHeadAttention,
}

impl Mrte {
    pub fn new(
        content_channels: usize,
        hidden: usize,
        out_channels: usize,
        heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            c_pre: conv1x1(content_channels, hidden, vb.pp("c_pre"))?,
            text_pre: conv1x1(content_channels, hidden, vb.pp("text_pre"))?,
            c_post: conv1x1(hidden, out_channels, vb.pp("c_post"))?,
            attn: RelMultiHeadAttention::new(hidden, hidden, heads, None, vb.pp("cross_attention"))?,
        })
    }

    /// `ssl`/`text`: (B,C,T)；`ssl_mask`: (B,1,T1)；`text_mask`: (B,1,T2)；`ge`: (B,ge_channels,1)
    pub fn forward(
        &self,
        ssl: &Tensor,
        ssl_mask: &Tensor,
        text: &Tensor,
        text_mask: &Tensor,
        ge: &Tensor,
    ) -> Result<Tensor> {
        // (B, T1, T2)
        let attn_mask = text_mask
            .unsqueeze(2)?
            .broadcast_mul(&ssl_mask.unsqueeze(3)?)?;
        let attn_mask = attn_mask.squeeze(1)?;

        let ssl_enc = self.c_pre.forward(&(ssl * ssl_mask)?)?;
        let text_enc = self.text_pre.forward(&(text * text_mask)?)?;

        let x = self
            .attn
            .forward(&(ssl_enc.clone() * ssl_mask)?, &(text_enc * text_mask)?, Some(&attn_mask))?;
        // 注意力输出 + c_pre(ssl) + ge（ge 是 (B,512,1)，广播到时间维）
        // ge 是 (B, C, 1)，要广播到时间维
        let x = (&x + &ssl_enc)?.broadcast_add(ge)?;
        Ok(self.c_post.forward(&(x * ssl_mask)?)?)
    }

    /// 无 padding 版本（mask 恒为全 1）
    pub fn forward_plain(&self, ssl: &Tensor, text: &Tensor, ge: &Tensor) -> Result<Tensor> {
        let ssl_enc = self.c_pre.forward(ssl)?;
        let text_enc = self.text_pre.forward(text)?;
        let x = self.attn.forward(&ssl_enc, &text_enc, None)?;
        let x = (&x + &ssl_enc)?.broadcast_add(ge)?;
        Ok(self.c_post.forward(&x)?)
    }
}

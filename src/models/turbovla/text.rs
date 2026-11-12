//! TurboVLA 的文本编码器：BERT-base（frozen）+ 子句级 3D attention mask + 自定义 position_ids。
//!
//! 和 candle-transformers 的 `bert` 的权重命名一致（`text_encoder.bert.*`）；区别是
//! TurboVLA 用 `[CLS]`/`[SEP]`/`.`/`?` 把指令切成子句，每段独立位置编码、段内全连接。

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, LayerNormConfig, VarBuilder};
use super::linear::{linear, Linear};
use tokenizers::{Tokenizer, TruncationParams};

use super::config::BertConfig;

struct BertLayer {
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    attn_ln: LayerNorm,
    inter: Linear,
    out: Linear,
    out_ln: LayerNorm,
}

pub struct BertEncoder {
    word: Embedding,
    pos: Embedding,
    tok: Embedding,
    ln: LayerNorm,
    layers: Vec<BertLayer>,
    heads: usize,
    head_dim: usize,
    scale: f64,
}

impl BertEncoder {
    pub fn load(cfg: &BertConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let heads = cfg.num_attention_heads;
        let vb_e = vb.pp("embeddings");
        let word = candle_nn::embedding(cfg.vocab_size, h, vb_e.pp("word_embeddings"))?;
        let pos = candle_nn::embedding(cfg.max_position_embeddings, h, vb_e.pp("position_embeddings"))?;
        let tok = candle_nn::embedding(cfg.type_vocab_size, h, vb_e.pp("token_type_embeddings"))?;
        let ln = candle_nn::layer_norm(
            h,
            LayerNormConfig { eps: cfg.layer_norm_eps, ..Default::default() },
            vb_e.pp("LayerNorm"),
        )?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let vb_l = vb.pp(format!("encoder.layer.{i}"));
            let vb_a = vb_l.pp("attention");
            layers.push(BertLayer {
                q: linear(h, h, vb_a.pp("self").pp("query"))?,
                k: linear(h, h, vb_a.pp("self").pp("key"))?,
                v: linear(h, h, vb_a.pp("self").pp("value"))?,
                o: linear(h, h, vb_a.pp("output").pp("dense"))?,
                attn_ln: candle_nn::layer_norm(
                    h,
                    LayerNormConfig { eps: cfg.layer_norm_eps, ..Default::default() },
                    vb_a.pp("output").pp("LayerNorm"),
                )?,
                inter: linear(h, cfg.intermediate_size, vb_l.pp("intermediate").pp("dense"))?,
                out: linear(cfg.intermediate_size, h, vb_l.pp("output").pp("dense"))?,
                out_ln: candle_nn::layer_norm(
                    h,
                    LayerNormConfig { eps: cfg.layer_norm_eps, ..Default::default() },
                    vb_l.pp("output").pp("LayerNorm"),
                )?,
            });
        }
        Ok(Self {
            word,
            pos,
            tok,
            ln,
            layers,
            heads,
            head_dim: h / heads,
            scale: ((h / heads) as f64).powf(-0.5),
        })
    }

    /// `input_ids`/`position_ids`/`token_type_ids`: (B,S) u32；`mask`: (B,S,S) bool（true=可见）。
    pub fn forward(
        &self,
        input_ids: &Tensor,
        position_ids: &Tensor,
        token_type_ids: &Tensor,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let (b, s) = input_ids.dims2()?;
        let x = (self.word.forward(input_ids)?
            + self.pos.forward(position_ids)?
            + self.tok.forward(token_type_ids)?)?;
        let mut x = self.ln.forward(&x)?;

        // additive mask: (1 - m) * -10000，形状 (B,1,S,S)
        let m = mask.to_dtype(DType::F32)?;
        let additive = ((m.ones_like()? - &m)? * -10000.0)?
            .unsqueeze(1)?
            .to_dtype(x.dtype())?;

        for layer in &self.layers {
            let residual = &x;
            let q = layer.q.forward(&x)?.reshape((b, s, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
            let k = layer.k.forward(&x)?.reshape((b, s, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
            let v = layer.v.forward(&x)?.reshape((b, s, self.heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
            let attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * self.scale)?;
            let attn = attn.broadcast_add(&additive)?;
            let attn = candle_nn::ops::softmax_last_dim(&attn)?;
            let out = attn.matmul(&v)?;
            let out = out.transpose(1, 2)?.reshape((b, s, self.heads * self.head_dim))?;
            let out = layer.o.forward(&out)?;
            x = layer.attn_ln.forward(&(out + residual)?)?;

            let residual = &x;
            let inter = candle_nn::Activation::Gelu.forward(&layer.inter.forward(&x)?)?;
            let out = layer.out.forward(&inter)?;
            x = layer.out_ln.forward(&(out + residual)?)?;
        }
        Ok(x)
    }
}

/// 一条指令 tokenize 后的结果（定长 pad）。
pub struct EncodedInstruction {
    pub input_ids: Vec<u32>,
    pub position_ids: Vec<u32>,
    pub attention_mask: Vec<u32>,
    /// (S,S) bool：true = 可见
    pub self_attention: Vec<bool>,
    pub length: usize,
}

/// BERT tokenizer + 子句级 mask 的生成器。
pub struct TextTokenizer {
    tokenizer: Tokenizer,
    special: Vec<u32>,
    sep_id: u32,
    max_len: usize,
}

impl TextTokenizer {
    /// `max_len` = 模型允许的最长 token 数（LIBERO 是 21）。
    pub fn from_file(path: &str, max_len: usize) -> anyhow::Result<Self> {
        let mut tokenizer = Tokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("加载 tokenizer.json 失败：{e}"))?;
        tokenizer
            .with_truncation(Some(TruncationParams {
                max_length: max_len,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("设置 truncation 失败：{e}"))?;
        let mut special = Vec::new();
        for token in ["[CLS]", "[SEP]", ".", "?"] {
            if let Some(id) = tokenizer.token_to_id(token) {
                special.push(id);
            }
        }
        let sep_id = tokenizer
            .token_to_id("[SEP]")
            .ok_or_else(|| anyhow::anyhow!("tokenizer 里没有 [SEP]"))?;
        Ok(Self {
            tokenizer,
            special,
            sep_id,
            max_len,
        })
    }

    /// 一条指令 → ids / position_ids / 2D pad mask / 3D 子句 mask，定长 pad 到 `pad_len`。
    pub fn encode(&self, text: &str, pad_len: usize) -> anyhow::Result<EncodedInstruction> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenize 失败：{e}"))?;
        let mut ids = encoding.get_ids().to_vec();
        // 和 HF `truncation=True, max_length=pad_len` 对齐：保留结尾 [SEP]
        if ids.len() > pad_len {
            ids.truncate(pad_len);
            if let Some(last) = ids.last_mut() {
                *last = self.sep_id;
            }
        }
        let s = pad_len;
        let mut input_ids = vec![0u32; s];
        let mut attention = vec![0u32; s];
        for (i, &id) in ids.iter().take(s).enumerate() {
            input_ids[i] = id;
            attention[i] = 1;
        }

        // 子句切分：和 turbovla/text/bert.py 的 generate_masks_with_special_tokens 一致
        let mut self_attn = vec![false; s * s];
        let mut position_ids = vec![0u32; s];
        for i in 0..s {
            self_attn[i * s + i] = true;
        }
        let mut previous = 0usize;
        for col in 0..s {
            if !self.special.contains(&input_ids[col]) {
                continue;
            }
            if col == 0 || col == s - 1 {
                self_attn[col * s + col] = true;
                position_ids[col] = 0;
            } else {
                for row in previous + 1..=col {
                    for c in previous + 1..=col {
                        self_attn[row * s + c] = true;
                    }
                }
                for (k, node) in (previous + 1..=col).enumerate() {
                    position_ids[node] = k as u32;
                }
            }
            previous = col;
        }
        let length = attention.iter().filter(|&&v| v == 1).count();
        Ok(EncodedInstruction {
            input_ids,
            position_ids,
            attention_mask: attention,
            self_attention: self_attn,
            length,
        })
    }

    pub fn max_len(&self) -> usize {
        self.max_len
    }
}

/// 把一批 `EncodedInstruction` 拼成张量。
pub struct TextBatch {
    pub input_ids: Tensor,
    pub position_ids: Tensor,
    pub token_type_ids: Tensor,
    pub self_attention: Tensor,
    /// (B,S) u8，1 = 真实 token，0 = padding
    pub key_padding: Tensor,
    /// (B,S) u8，1 = padding（给融合层的 mask 用）
    pub pad_mask: Tensor,
    pub lengths: Vec<usize>,
}

impl TextBatch {
    /// `group_len` 是这条指令在 Python 里实际 pad 到的长度（LIBERO 是 11/14/21），
    /// `full_len` 是整体长度（`padding_length`）。组外行在 Python 里是**全零**。
    pub fn build(
        items: &[EncodedInstruction],
        device: &Device,
        group_len: usize,
        full_len: usize,
    ) -> Result<Self> {
        let b = items.len();
        let mut ids = vec![0u32; b * full_len];
        let mut pos = vec![0u32; b * full_len];
        let typ = vec![0u32; b * full_len];
        let mut am = vec![0u32; b * full_len];
        let mut pads = vec![1u8; b * full_len];
        // 组内块从 items 拷过来，组外保持单位阵
        let mut sam = vec![0u8; b * full_len * full_len];
        for (bi, it) in items.iter().enumerate() {
            let base = bi * full_len;
            for i in 0..group_len.min(it.input_ids.len()) {
                ids[base + i] = it.input_ids[i];
                pos[base + i] = it.position_ids[i];
                am[base + i] = it.attention_mask[i];
                pads[base + i] = if it.attention_mask[i] == 1 { 0 } else { 1 };
            }
            for r in 0..full_len {
                for c in 0..full_len {
                    let v = if r < group_len && c < group_len {
                        it.self_attention[r * group_len + c]
                    } else {
                        r == c
                    };
                    sam[base + r * full_len + c] = if v { 1 } else { 0 };
                }
            }
        }
        Ok(Self {
            input_ids: Tensor::from_vec(ids, (b, full_len), device)?,
            position_ids: Tensor::from_vec(pos, (b, full_len), device)?,
            token_type_ids: Tensor::from_vec(typ, (b, full_len), device)?,
            self_attention: Tensor::from_vec(sam, (b, full_len, full_len), device)?
                .to_dtype(DType::U8)?,
            key_padding: Tensor::from_vec(am, (b, full_len), device)?,
            pad_mask: Tensor::from_vec(pads, (b, full_len), device)?,
            lengths: items.iter().map(|i| i.length).collect(),
        })
    }
}

/// 一条指令实际要 pad 到的组长度（`padding_length_by_instruction`），没有就退回默认。
pub fn group_len_for(
    layout: &std::collections::HashMap<String, usize>,
    instruction: &str,
    default: usize,
) -> usize {
    layout.get(instruction).copied().unwrap_or(default).min(default)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_len_prefers_layout() {
        let mut layout = std::collections::HashMap::new();
        layout.insert("put the bowl on the plate".to_string(), 11usize);
        assert_eq!(group_len_for(&layout, "put the bowl on the plate", 21), 11);
        assert_eq!(group_len_for(&layout, "turn on the stove", 21), 21);
    }

    #[test]
    fn sub_sentence_mask_is_block_diagonal() {
        // 不依赖真实 tokenizer：直接验证算法在 [CLS]a b[SEP] 上的形状
        let s = 5usize;
        let ids = [1u32, 7, 8, 2, 0];
        let special = [1u32, 2, 3, 4];
        let mut attn = vec![false; s * s];
        let mut pos = vec![0u32; s];
        for i in 0..s {
            attn[i * s + i] = true;
        }
        let mut previous = 0usize;
        for col in 0..s {
            if !special.contains(&ids[col]) {
                continue;
            }
            if col == 0 || col == s - 1 {
                attn[col * s + col] = true;
            } else {
                for r in previous + 1..=col {
                    for c in previous + 1..=col {
                        attn[r * s + c] = true;
                    }
                }
                for (k, node) in (previous + 1..=col).enumerate() {
                    pos[node] = k as u32;
                }
            }
            previous = col;
        }
        // [CLS]=1 独立；a,b 与 [SEP] 同段（2..=3）互相可见
        assert!(attn[1 * s + 1] && attn[1 * s + 2] && attn[2 * s + 3] && attn[3 * s + 1]);
        // 位置在段内从 0 开始（块 = previous+1..=col，含 [SEP]）
        assert_eq!(pos[1], 0);
        assert_eq!(pos[2], 1);
        assert_eq!(pos[3], 2);
    }
}

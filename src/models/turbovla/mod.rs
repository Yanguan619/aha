//! TurboVLA（Vision-Language-Action）模型。
//!
//! 结构（见 `third-party/vla/turbovla/models/`）：
//! - 文本：BERT-base（frozen，子句级 3D attention mask + 自定义 position_ids）→ 投影到 256；
//! - 视觉：DINOv3 ViT-B/16（RoPE + 4 register + LayerScale），2 个视角各自编码后取 patch tokens；
//! - 融合：6 层 BiAttentionBlock（视觉↔文本双向）+ 文本 TransformerEncoderLayer；
//! - 动作：state 投影成 2 个 token，和融合后的 token 一起进 3 层 TransformerDecoder，
//!   12 个 action query 输出 12×7 的动作 chunk（tanh）。
//!
//! 权重来自 `tools/vla/convert_turbovla.py` 转出的 `model.safetensors`（键名与 Python 一致）。

pub mod action;
pub mod config;
pub mod fusion;
pub mod text;
pub mod vision;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::{Linear, VarBuilder};
use image::imageops::FilterType;

use self::action::ActionHead;
use self::config::ModelLayout;
use self::fusion::{Interaction, VisionProjection};
use self::text::{group_len_for, BertEncoder, TextBatch, TextTokenizer};
use self::vision::DinoV3;

pub use config::TurboVlaConfig;

const IMAGE_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGE_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// LIBERO 的 proprio 归一化（和 Python `normalize_state` 一致）。
const PROPRIO_MEAN: [f32; 8] = [
    -0.04190646484494209,
    0.03539437800645828,
    0.8257066607475281,
    2.908315658569336,
    -0.5562158823013306,
    -0.16649103164672852,
    0.02831534668803215,
    -0.028561558574438095,
];
const PROPRIO_STD: [f32; 8] = [
    0.10743443667888641,
    0.14424759149551392,
    0.25723373889923096,
    0.34413808584213257,
    1.234430193901062,
    0.35798805952072144,
    0.013308786787092686,
    0.013174591585993767,
];

fn normalize_state(state: &[f32]) -> Vec<f32> {
    state
        .iter()
        .enumerate()
        .map(|(i, v)| (v - PROPRIO_MEAN[i]) / (PROPRIO_STD[i] + 1e-6))
        .collect()
}

/// 常驻的 TurboVLA 模型。
pub struct TurboVlaModel {
    text: TextTokenizer,
    bert: BertEncoder,
    text_proj: Linear,
    vision: DinoV3,
    vision_proj: VisionProjection,
    view_embedding: Tensor,
    interaction: Interaction,
    action: ActionHead,
    device: Device,
    dtype: DType,
    image_size: usize,
    num_views: usize,
    state_dim: usize,
    action_dim: usize,
    horizon: usize,
    pad_len: usize,
    pad_len_by_instruction: std::collections::HashMap<String, usize>,
    zero_fill: bool,
}

impl TurboVlaModel {
    pub fn load(path: &str, device: &Device, dtype: DType) -> Result<Self> {
        let layout = ModelLayout::open(path)?;
        let cfg = &layout.model;
        let hidden = cfg.interaction.hidden_dim;
        let pad_len = cfg.text.padding_length.unwrap_or(cfg.text.max_length);
        let tokenizer_path = layout
            .tokenizer
            .to_str()
            .context("tokenizer 路径不是 UTF-8")?;
        let text = TextTokenizer::from_file(tokenizer_path, cfg.text.max_length)?;

        // SAFETY: safetensors 只读映射；文件在模型生命周期内不会变。
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[layout.weights.clone()], dtype, device)?
        };

        let bert = BertEncoder::load(&layout.text, vb.pp("text_encoder").pp("bert"))?;
        let text_proj = candle_nn::linear(
            layout.text.hidden_size,
            hidden,
            vb.pp("text_encoder").pp("text_projection"),
        )?;
        let vision = DinoV3::load(
            &layout.vision,
            vb.pp("vision_encoder").pp("backbone"),
            device,
        )?;
        let vision_proj = VisionProjection::load(layout.vision.hidden_size, hidden, vb.clone())?;
        let view_embedding = vb.get((1, cfg.vision.num_views, hidden), "view_embedding")?;
        let interaction = Interaction::load(&cfg.interaction, vb.clone())?;
        let action = ActionHead::load(
            &cfg.action,
            hidden,
            cfg.interaction.nheads,
            cfg.interaction.dim_feedforward,
            vb.pp("action_head"),
        )?;

        Ok(Self {
            text,
            bert,
            text_proj,
            vision,
            vision_proj,
            view_embedding,
            interaction,
            action,
            device: device.clone(),
            dtype,
            image_size: cfg.vision.image_size,
            num_views: cfg.vision.num_views,
            state_dim: cfg.action.state_dim,
            action_dim: cfg.action.action_dim,
            horizon: cfg.action.horizon,
            pad_len,
            pad_len_by_instruction: cfg.text.padding_length_by_instruction.clone(),
            zero_fill: cfg.interaction.padding_strategy == "zero_fill",
        })
    }

    pub fn action_dim(&self) -> usize {
        self.action_dim
    }    pub fn horizon(&self) -> usize {
        self.horizon
    }
    pub fn num_views(&self) -> usize {
        self.num_views
    }
    pub fn state_dim(&self) -> usize {
        self.state_dim
    }

    /// 只跑融合层第 `i` 层（对齐用）。
    pub fn debug_fusion_layer(
        &self,
        i: usize,
        visual: &Tensor,
        text: &Tensor,
        mask_l: &Tensor,
        self_mask: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        self.interaction
            .forward_layer(i, visual, text, Some(mask_l), Some(self_mask), self.zero_fill)
            .map_err(anyhow::Error::from)
    }

    /// 只跑动作头（对齐用）：`condition` = (1, tokens, hidden)。
    pub fn action_from_tokens(&self, condition: Tensor, state: &[f32]) -> Result<Tensor> {
        let st = Tensor::from_vec(normalize_state(state), (1, self.state_dim), &self.device)?
            .to_dtype(self.dtype)?;
        self.action
            .forward(&condition, &st)
            .map_err(anyhow::Error::from)
    }

    /// 一帧一个视角：`images` 是每帧的 JPEG/PNG 字节（顺序和训练一致），
    /// `state` 是 state_dim 维（LIBERO 是 8）。返回 `(1, horizon, action_dim)`。
    pub fn predict(&self, images: &[&[u8]], instruction: &str, state: &[f32]) -> Result<Tensor> {
        Ok(self.stages(images, instruction, state)?.actions)
    }

    /// 跑一遍并保留各阶段张量（对齐 Python 参考时用）。
    pub fn stages(
        &self,
        images: &[&[u8]],
        instruction: &str,
        state: &[f32],
    ) -> Result<Stages> {
        if images.len() != self.num_views {
            bail!("需要 {} 个视角，收到 {}", self.num_views, images.len());
        }
        if state.len() != self.state_dim {
            bail!("state 需要 {} 维，收到 {}", self.state_dim, state.len());
        }

        let frames: Vec<Tensor> = images
            .iter()
            .map(|bytes| preprocess(bytes, self.image_size, &self.device, self.dtype))
            .collect::<Result<_>>()?;
        let pixel = Tensor::stack(&frames, 0)?.unsqueeze(0)?; // (1,V,3,H,W)
        let pad_len = self.pad_len;
        let group_len = group_len_for(&self.pad_len_by_instruction, instruction, pad_len);

        let mut vis = Vec::with_capacity(self.num_views);
        for v in 0..self.num_views {
            let view = pixel.narrow(1, v, 1)?.squeeze(1)?; // (1,3,H,W)
            vis.push(self.vision.forward(&view)?); // (1,P,768)
        }
        let vis = Tensor::stack(&vis, 1)?; // (1,V,P,768)
        let vis = self.vision_proj.forward(&vis)?; // (1,V,P,hidden)
        let view = self.view_embedding.unsqueeze(2)?; // (1,V,1,hidden)
        let vis = vis.broadcast_add(&view)?; // + view embedding
        let vis = vis.flatten(1, 2)?; // (1,V*P,hidden)

        let enc = self.text.encode(instruction, group_len)?;
        let batch = TextBatch::build(&[enc], &self.device, group_len, pad_len)?;
        let mut hidden = self.bert.forward(
            &batch.input_ids,
            &batch.position_ids,
            &batch.token_type_ids,
            &batch.self_attention,
        )?;
        if group_len < pad_len {
            // Python 里组外行是 768 维零向量（过投影后只剩 bias）
            let mut keep = vec![0f32; pad_len];
            keep[..group_len].fill(1.0);
            let keep =
                Tensor::from_vec(keep, (1, pad_len, 1), &self.device)?.to_dtype(hidden.dtype())?;
            hidden = hidden.broadcast_mul(&keep)?;
        }
        let txt = self.text_proj.forward(&hidden)?;

        let (v, l) = self.interaction.forward(
            &vis,
            &txt,
            Some(&batch.pad_mask),
            Some(&batch.self_attention),
            self.zero_fill,
        )?;
        let condition = Tensor::cat(&[&v, &l], 1)?;

        let st = Tensor::from_vec(normalize_state(state), (1, self.state_dim), &self.device)?
            .to_dtype(self.dtype)?;
        let actions = self
            .action
            .forward(&condition, &st)
            .map_err(anyhow::Error::from)?;
        Ok(Stages {
            pixel,
            input_ids: batch.input_ids.clone(),
            position_ids: batch.position_ids.clone(),
            self_attention: batch.self_attention.clone(),
            key_padding: batch.key_padding.clone(),
            pad_mask: batch.pad_mask.clone(),
            text: txt.clone(),
            visual: vis.clone(),
            fused_v: v,
            fused_l: l,
            actions,
        })
    }
}

/// 前向各阶段的张量，方便和 Python 参考逐段对比。
pub struct Stages {
    pub pixel: Tensor,
    pub input_ids: Tensor,
    pub position_ids: Tensor,
    pub self_attention: Tensor,
    pub key_padding: Tensor,
    pub pad_mask: Tensor,
    pub text: Tensor,
    pub visual: Tensor,
    pub fused_v: Tensor,
    pub fused_l: Tensor,
    pub actions: Tensor,
}

/// 一张图 → (3,size,size) 的归一化张量（和 HF `preprocessor_config` 一致）。
fn preprocess(bytes: &[u8], size: usize, device: &Device, dtype: DType) -> Result<Tensor> {
    let img = image::load_from_memory(bytes)
        .map_err(|e| anyhow::anyhow!("图片解码失败：{e}"))?;
    // 和 PIL 对齐：尺寸已经一致时不做重采样（resize_exact 会模糊）。
    let rgb = if img.width() as usize == size && img.height() as usize == size {
        img.to_rgb8()
    } else {
        img.resize_exact(size as u32, size as u32, FilterType::CatmullRom)
            .to_rgb8()
    };
    let mut data = vec![0f32; 3 * size * size];
    for (x, y, px) in rgb.enumerate_pixels() {
        for c in 0..3 {
            data[c * size * size + y as usize * size + x as usize] =
                (px[c] as f32 / 255.0 - IMAGE_MEAN[c]) / IMAGE_STD[c];
        }
    }
    Ok(Tensor::from_vec(data, (3, size, size), device)?.to_dtype(dtype)?)
}

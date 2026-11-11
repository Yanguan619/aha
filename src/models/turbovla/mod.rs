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
#[cfg(feature = "cuda")]
pub mod graph;
#[cfg(feature = "cuda")]
pub mod heads_cuda;
pub mod linear;
pub mod ops;
pub mod text;
pub mod vision;

use anyhow::{Context, Result, bail};
use candle_core::{DType, Device, Module, Tensor};
use crate::models::turbovla::linear::{linear, Linear};
use candle_nn::VarBuilder;
use image::imageops::FilterType;
use std::collections::HashMap;
use std::sync::Mutex;

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

/// 分段计时器：`laps` 为 `None` 时所有调用都是空操作（零开销）。
struct Timer<'a> {
    laps: Option<&'a mut Vec<(&'static str, f64)>>,
    device: Option<&'a Device>,
    start: std::time::Instant,
    last: std::time::Instant,
}

impl<'a> Timer<'a> {
    fn disabled() -> Self {
        Self {
            laps: None,
            device: None,
            start: std::time::Instant::now(),
            last: std::time::Instant::now(),
        }
    }

    fn new(device: &'a Device, laps: Option<&'a mut Vec<(&'static str, f64)>>) -> Self {
        let now = std::time::Instant::now();
        Self {
            laps,
            device: Some(device),
            start: now,
            last: now,
        }
    }

    /// 记一段耗时。先同步设备，否则 CUDA 的异步队列会把上一段的时间算进下一段。
    fn lap(&mut self, name: &'static str) -> Result<()> {
        if let Some(laps) = self.laps.as_mut() {
            if let Some(dev) = self.device {
                dev.synchronize()?;
            }
            let now = std::time::Instant::now();
            laps.push((name, (now - self.last).as_secs_f64() * 1e3));
            self.last = now;
        }
        Ok(())
    }

    fn total(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1e3
    }
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
    /// 指令 → 文本侧的各阶段张量（BERT 段只跟指令有关，可直接复用）。
    text_cache: Mutex<HashMap<(String, usize), TextFeat>>,
    /// `text_cache` 的访问序（队尾 = 最近用过），用于 LRU 驱逐。
    text_order: Mutex<Vec<(String, usize)>>,
    /// CUDA graph 默认开启（CUDA 上每帧只做 H2D + 一次 launch）；
    /// `TURBOVLA_GRAPH=0` 关掉，捕获失败自动退回逐算子。
    graph_on: std::sync::atomic::AtomicBool,
    /// 已捕获的 CUDA graph（LRU，最多 [`MAX_GRAPH_SLOTS`] 个槽）。
    #[cfg(feature = "cuda")]
    graphs: Mutex<Vec<GraphSlot>>,
}

/// 最多同时保留多少张已捕获的图。
///
/// 桌宠的指令是一个小集合（"看屏幕"、"走两步"…），2 张就够覆盖交替；给到 3
/// 留一点余量。每张图占一份图私有显存池 + 文本特征（几 MB 量级），再多就该
/// 考虑按使用频次驱逐了。
#[cfg(feature = "cuda")]
const MAX_GRAPH_SLOTS: usize = 3;

/// 文本特征缓存容量。每条是一个 BERT 段的结果（几 MB 显存）。
const TEXT_CACHE_CAP: usize = 8;

#[cfg(feature = "cuda")]
struct GraphSlot {
    /// (instruction, group_len, 视角数)：参数变了就重捕。
    key: (String, usize, usize),
    /// 图里烧死了这些文本特征的地址，必须一直活着。
    #[allow(dead_code)]
    feat: TextFeat,
    runner: crate::models::turbovla::graph::GraphRunner,
}

/// 文本侧的常量输出（同一指令、同一组长度下完全确定）。
#[derive(Clone)]
struct TextFeat {
    input_ids: Tensor,
    position_ids: Tensor,
    self_attention: Tensor,
    key_padding: Tensor,
    pad_mask: Tensor,
    text: Tensor,
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
        let text_proj = linear(
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
            text_cache: Mutex::new(HashMap::new()),
            text_order: Mutex::new(Vec::new()),
            graph_on: std::sync::atomic::AtomicBool::new(
                std::env::var("TURBOVLA_GRAPH").map(|v| v != "0").unwrap_or(true),
            ),

            #[cfg(feature = "cuda")]
            graphs: Mutex::new(Vec::new()),
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
        #[cfg(feature = "cuda")]
        if self.graph_on.load(std::sync::atomic::Ordering::Relaxed) && self.device.is_cuda() {
            match self.predict_graph(images, instruction, state) {
                Ok(t) => return Ok(t),
                Err(e) => {
                    eprintln!("[turbovla] CUDA graph 不可用，改用逐算子：{e:#}");
                    self.graph_on
                        .store(false, std::sync::atomic::Ordering::Relaxed);
                }
            }
        }
        Ok(self.stages(images, instruction, state)?.actions)
    }

    /// CUDA graph 路径：首次捕获，之后每帧只做 H2D + 一次 launch。
    #[cfg(feature = "cuda")]
    fn predict_graph(
        &self,
        images: &[&[u8]],
        instruction: &str,
        state: &[f32],
    ) -> Result<Tensor> {
        if images.len() != self.num_views {
            bail!("需要 {} 个视角，收到 {}", self.num_views, images.len());
        }
        if state.len() != self.state_dim {
            bail!("state 需要 {} 维，收到 {}", self.state_dim, state.len());
        }
        let pad_len = self.pad_len;
        let group_len = group_len_for(&self.pad_len_by_instruction, instruction, pad_len);
        let key = (instruction.to_string(), group_len, images.len());

        let pixel_host = self.pixel_host(images)?;
        let state_host = normalize_state(state);

        let mut graphs = self.graphs.lock().unwrap();
        // 命中：把这一槽挪到末尾（LRU 语义），再重放。
        if let Some(pos) = graphs.iter().position(|s| s.key == key) {
            let slot = graphs.remove(pos);
            slot.runner.upload(&pixel_host, &state_host)?;
            let out = slot.runner.replay()?;
            graphs.push(slot);
            return Ok(out);
        }

        // 未命中：捕一张新图。参数（指令 / 组长 / 视角数）变了就必须重捕——
        // 图把张量地址和形状都烧死了，没法复用。
        let feat = self.text_features(instruction, group_len, pad_len)?;
        let runner = crate::models::turbovla::graph::GraphRunner::try_capture(
            &self.device,
            self.dtype,
            (1, self.num_views, 3, self.image_size, self.image_size),
            &pixel_host,
            (1, self.state_dim),
            &state_host,
            |pixel, st| {
                let mut timer = Timer::disabled();
                Ok(self.compute(pixel.tensor(), &feat, st.tensor(), &mut timer)?.1.2)
            },
        )?;

        // 捕获时用的就是这一帧的输入，直接重放一次拿到结果
        let first = runner.replay()?;
        // 超出容量就淘汰最久没用的那一槽（Vec 头部 = 最久未命中）。
        if graphs.len() >= MAX_GRAPH_SLOTS {
            graphs.remove(0);
        }
        graphs.push(GraphSlot {
            key,
            feat,
            runner,
        });
        Ok(first)
    }

    /// 跑一遍并保留各阶段张量（对齐 Python 参考时用）。
    pub fn stages(
        &self,
        images: &[&[u8]],
        instruction: &str,
        state: &[f32],
    ) -> Result<Stages> {
        let mut timer = Timer::disabled();
        self.run(images, instruction, state, &mut timer)
    }

    /// 分段计时（毫秒）。每段结束显式同步，否则 CUDA 的异步执行会把时间记到下一段。
    pub fn stages_timed(
        &self,
        images: &[&[u8]],
        instruction: &str,
        state: &[f32],
    ) -> Result<(Stages, Vec<(&'static str, f64)>)> {
        let mut laps = Vec::new();
        let mut timer = Timer::new(&self.device, Some(&mut laps));
        let stages = self.run(images, instruction, state, &mut timer)?;
        let total = timer.total();
        laps.push(("total", total));
        Ok((stages, laps))
    }

    fn run(
        &self,
        images: &[&[u8]],
        instruction: &str,
        state: &[f32],
        timer: &mut Timer<'_>,
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
        timer.lap("preprocess")?;

        let feat = self.text_features(instruction, group_len, pad_len)?;
        timer.lap("bert")?;

        let st = Tensor::from_vec(normalize_state(state), (1, self.state_dim), &self.device)?
            .to_dtype(self.dtype)?;
        let (vis, (v, l, actions)) = self.compute(&pixel, &feat, &st, timer)?;
        Ok(Stages {
            pixel,
            input_ids: feat.input_ids.clone(),
            position_ids: feat.position_ids.clone(),
            self_attention: feat.self_attention.clone(),
            key_padding: feat.key_padding.clone(),
            pad_mask: feat.pad_mask.clone(),
            text: feat.text.clone(),
            visual: vis,
            fused_v: v,
            fused_l: l,
            actions,
        })
    }

    /// 纯张量计算：图捕获的就是这一段，**不能有主机同步/主机内存分配**。
    ///
    /// 返回 `(vision tokens, (fused_v, fused_l, actions))`。
    fn compute(
        &self,
        pixel: &Tensor,
        feat: &TextFeat,
        st: &Tensor,
        timer: &mut Timer<'_>,
    ) -> Result<(Tensor, (Tensor, Tensor, Tensor))> {
        // 两个视角**一次前向**：ViT 逐样本独立，合批只是把 kernel 数量减半。
        let vis = self.vision.forward(&pixel.squeeze(0)?)?; // (V,P,768)
        let vis = vis.unsqueeze(0)?; // (1,V,P,768)
        let vis = self.vision_proj.forward(&vis)?; // (1,V,P,hidden)
        let view = self.view_embedding.unsqueeze(2)?; // (1,V,1,hidden)
        let vis = vis.broadcast_add(&view)?; // + view embedding
        let vis = vis.flatten(1, 2)?; // (1,V*P,hidden)
        timer.lap("vision")?;

        let (v, l) = self.interaction.forward(
            &vis,
            &feat.text,
            Some(&feat.pad_mask),
            Some(&feat.self_attention),
            self.zero_fill,
        )?;
        timer.lap("fusion")?;
        let condition = Tensor::cat(&[&v, &l], 1)?;

        let actions = self
            .action
            .forward(&condition, st)
            .map_err(anyhow::Error::from)?;
        timer.lap("action")?;
        Ok((vis, (v, l, actions)))
    }

    /// 图像字节 → `(V,3,H,W)` 的 f32 主机缓冲（图模式下直接 H2D 到静态显存）。
    fn pixel_host(&self, images: &[&[u8]]) -> Result<Vec<f32>> {
        let mut all = Vec::with_capacity(images.len() * 3 * self.image_size * self.image_size);
        for bytes in images {
            all.extend(preprocess_host(bytes, self.image_size)?);
        }
        Ok(all)
    }

    /// 文本侧各阶段：同一 (instruction, group_len) 只算一次。
    ///
    /// 缓存是 LRU 语义（`HashMap` + 插入顺序无能为力，所以用一个旁路队列记访问序），
    /// 溢出时只淘汰**最久没用的一条**，而不是像早期那样 `clear()` 全清。
    /// 桌宠的指令是小集合但会来回切换，全清等于每次切回来都要重跑 BERT
    /// （一条 ~100ms），比不缓存还糟。
    fn text_features(&self, instruction: &str, group_len: usize, pad_len: usize) -> Result<TextFeat> {
        let key = (instruction.to_string(), group_len);
        {
            let mut cache = self.text_cache.lock().unwrap();
            if let Some(hit) = cache.get(&key) {
                let feat = hit.clone();
                // 刷新访问序：把它挪到队尾。
                self.touch_text_key(&key);
                return Ok(feat);
            }
        }

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
        let feat = TextFeat {
            input_ids: batch.input_ids,
            position_ids: batch.position_ids,
            self_attention: batch.self_attention,
            key_padding: batch.key_padding,
            pad_mask: batch.pad_mask,
            text: self.text_proj.forward(&hidden)?,
        };

        let mut cache = self.text_cache.lock().unwrap();
        if cache.len() >= TEXT_CACHE_CAP && !cache.contains_key(&key) {
            // 淘汰最久没用的那条（`order` 头部）。order 里可能有已被淘汰的陈旧键，跳过。
            let mut order = self.text_order.lock().unwrap();
            while let Some(oldest) = order.first().cloned() {
                order.remove(0);
                if cache.remove(&oldest).is_some() {
                    break;
                }
            }
        }
        cache.insert(key.clone(), feat.clone());
        self.text_order.lock().unwrap().push(key);
        Ok(feat)
    }

    /// 把某个键挪到访问序的队尾（LRU 的「刚刚用过」）。
    fn touch_text_key(&self, key: &(String, usize)) {
        let mut order = self.text_order.lock().unwrap();
        if let Some(pos) = order.iter().position(|k| k == key) {
            let k = order.remove(pos);
            order.push(k);
        }
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
    let data = preprocess_host(bytes, size)?;
    Ok(Tensor::from_vec(data, (3, size, size), device)?.to_dtype(dtype)?)
}

/// 同上，但只出主机侧的 f32 缓冲（graph 模式直接 H2D 进静态显存）。
fn preprocess_host(bytes: &[u8], size: usize) -> Result<Vec<f32>> {
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
    Ok(data)
}

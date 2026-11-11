//! 把 GGUF 权重接到 candle 的 `VarBuilder` 上。
//!
//! aha 里现有模型（`bigvgan` / `campplus` / `w2v_bert_2_0` / `gpt2` / `qwen3` …）
//! 全是按 `VarBuilder` 名字取权重的。GGUF 只是"名字 + 张量"的另一种容器，所以
//! 只要实现一个 `SimpleBackend`，这些模型的 `new(vb, cfg)` **一行都不用改**就能
//! 直接吃 GGUF（含 Q4_K 等量化权重，取的时候 dequantize）。
//!
//! 名字约定：GGUF 里是 `组件/内部名`（如 `bigvgan/conv_pre.weight_v`），而 candle
//! 的 `vb.pp()` 用 `.` 拼内部名。所以这里用 `prefix + "/" + candle名` 即可对上，
//! 对不上的场景可以用 [`GgufVarBuilder::with_alias`] 显式覆盖。
//!
//! ```no_run
//! let gguf = Gguf::new(lc.into_content(), file, device.clone());
//! let vb = VarBuilder::from_backend(
//!     Box::new(GgufVarBuilder::new(gguf, "bigvgan")),
//!     DType::F32,
//!     device,
//! );
//! let bigvgan = BigVGAN::new(vb, &cfg)?;
//! ```

use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::Mutex;

use candle_core::{DType, Device, Shape, Tensor};
use candle_nn::Init;
use candle_nn::var_builder::SimpleBackend;

use crate::models::common::gguf::Gguf;

fn err(msg: impl Into<String>) -> candle_core::Error {
    candle_core::Error::Msg(msg.into())
}

/// 基于 GGUF 的 VarBuilder 后端
pub struct GgufVarBuilder<R: Read + Seek + Send> {
    gguf: Mutex<Gguf<R>>,
    prefix: Option<String>,
    /// candle 名 -> GGUF 名，用于名字对不上的个别张量
    alias: HashMap<String, String>,
}

impl<R: Read + Seek + Send> GgufVarBuilder<R> {
    /// `prefix` 是 GGUF 里的组件前缀（如 `bigvgan`）
    pub fn new(gguf: Gguf<R>, prefix: impl Into<String>) -> Self {
        Self {
            gguf: Mutex::new(gguf),
            prefix: Some(prefix.into()),
            alias: HashMap::new(),
        }
    }

    /// 不加前缀（GGUF 里的名字就是 candle 侧的名字）
    pub fn new_unprefixed(gguf: Gguf<R>) -> Self {
        Self {
            gguf: Mutex::new(gguf),
            prefix: None,
            alias: HashMap::new(),
        }
    }

    /// candle 侧的名字 `from` 实际对应 GGUF 里的 `to`
    pub fn with_alias(mut self, from: &str, to: &str) -> Self {
        self.alias.insert(from.to_string(), to.to_string());
        self
    }

    /// candle `VarBuilder` 传进来的名字 -> GGUF 里的名字（默认拼接方式）
    pub fn gguf_name(&self, name: &str) -> String {
        if let Some(to) = self.alias.get(name) {
            return to.clone();
        }
        match &self.prefix {
            Some(p) if !p.is_empty() => format!("{p}/{name}"),
            _ => name.to_string(),
        }
    }

    /// 真正去 GGUF 里找：`prefix/name`、`prefix.name`、`name` 依次尝试，
    /// 因为不同打包器的分隔符不一样（audiocpp 的 bigvgan 用 `/`，
    /// campplus 是 `campplus/speaker_encoder.head.conv1.weight` 这种混着来）
    fn lookup(&self, gguf: &Gguf<R>, name: &str) -> String {
        if let Some(to) = self.alias.get(name) {
            return to.clone();
        }
        if let Some(p) = &self.prefix
            && !p.is_empty()
        {
            for sep in ["/", "."] {
                let cand = format!("{p}{sep}{name}");
                if gguf.contains(&cand) {
                    return cand;
                }
                // candle 的 `pp()` 用 `.` 拼，GGUF 的组件名后面可能又用 `.`（如
                // `campplus/speaker_encoder.head...`、`qwen_emotion/model.layers...`），
                // 所以再试一次把第一个 `.` 换成 `/`
                let mixed = cand.replacen('.', "/", 1);
                if mixed != cand && gguf.contains(&mixed) {
                    return mixed;
                }
            }
        }
        name.to_string()
    }

    /// 取回里层 `Gguf`（例如后面还要用 `qmatmul` / `metadata()`）
    pub fn into_inner(self) -> Gguf<R> {
        self.gguf.into_inner().unwrap_or_else(|e| e.into_inner())
    }

    fn load(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let mut gguf = self
            .gguf
            .lock()
            .map_err(|e| err(format!("GgufVarBuilder 锁已中毒: {e}")))?;
        let full = self.lookup(&gguf, name);
        if !gguf.contains(&full) {
            return Err(err(format!(
                "GGUF 里没有张量 {full}（candle 侧名字 {name}）"
            )));
        }
        let t = gguf
            .get_dequantized(&full)
            .map_err(|e| err(format!("读取 {full} 失败: {e}")))?;
        let t = t.to_dtype(dtype)?;
        if t.device().same_device(dev) {
            Ok(t)
        } else {
            t.to_device(dev)
        }
    }
}

impl<R: Read + Seek + Send> std::fmt::Debug for GgufVarBuilder<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GgufVarBuilder")
            .field("prefix", &self.prefix)
            .field("alias", &self.alias.len())
            .finish()
    }
}

impl<R: Read + Seek + Send> SimpleBackend for GgufVarBuilder<R> {
    fn get(
        &self,
        s: Shape,
        name: &str,
        _h: Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let t = self.load(name, dtype, dev)?;
        if t.dims() == s.dims() {
            return Ok(t);
        }
        // 元素数一致就直接 reshape（GGUF 维度已翻成 PyTorch 序，通常不会走到这）
        if t.elem_count() == s.elem_count() {
            return t.reshape(s);
        }
        Err(err(format!(
            "张量 {} 形状不匹配：GGUF 是 {:?}，candle 侧期望 {:?}",
            self.gguf_name(name),
            t.dims(),
            s.dims()
        )))
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        self.load(name, dtype, dev)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        match self.gguf.lock() {
            Ok(g) => {
                let full = self.lookup(&g, name);
                g.contains(&full)
            }
            Err(_) => false,
        }
    }
}

//! GPT-SoVITS v2Pro 的 SoVITS 解码链在 QNN（HTP）上的运行时封装。
//!
//! 图由 `scripts/gpt_sovits_export_onnx.py` 导出（`ref_enc + quantizer.decode + enc_p +
//! flow + dec` 一起），再由 `scripts/qnn_convert.py` 转成 context binary。输入固定：
//!
//! | 名字 | 类型 | 形状 | 含义 |
//! |---|---|---|---|
//! | `codes` | i32 | `[1, 64, 1]` | 语义 codes，固定 64 帧 |
//! | `text` | i32 | `[1, 96]` | 音素 id，定长 96 |
//! | `refer` | f32 | `[1, 704, 704]` | 参考幅度谱前 704 个频点 × 704 帧 |
//! | `sv_emb` | f32 | `[1, 20480]` | v2Pro 的说话人向量 |
//!
//! 输出 `audio`（f32，`(1, 1, N)` 展平）。
//!
//! 这张图是**定长**的：超过 64 帧的句子必须由调用方回退到 candle 实现。

use std::path::Path;

use anyhow::{Result, anyhow};

use crate::models::common::runtime::qnn::{self, Runtime};

/// 图里 codes 的固定帧数。
pub const NUM_CODES: usize = 64;
/// 图里 text 的固定长度。
pub const TEXT_LEN: usize = 96;
/// 参考谱保留的频点数（`ref_enc = MelStyleEncoder(704, ...)`，上游取 `refer[:, :704]`）。
pub const REFER_BINS: usize = 704;
/// 图里参考谱的固定帧数。
pub const REFER_FRAMES: usize = 704;
/// v2Pro 说话人向量维度。
pub const SV_DIM: usize = 20480;

/// 一个已打开的 SoVITS QNN 图。
pub struct SovitsQnn {
    rt: Runtime,
    graph: usize,
}

impl SovitsQnn {
    /// 打开 context binary（`--backend/--system` 是设备上的 QNN 库名）。
    pub fn open(context_bin: impl AsRef<Path>, backend: &str, system: &str) -> Result<Self> {
        Self::from_runtime(Runtime::open(context_bin.as_ref(), backend, system)?)
    }

    /// 打开模型库（`lib<name>.so` + 同目录的 `<name>.bin`）。
    ///
    /// SoVITS 的图 IO 全是 float32，而 context binary 的 float32 IO 在 HTP 上会
    /// 报 `Dma execution failed`；走模型库由 QNN 自己管理缓冲，实测可执行。
    pub fn open_model_lib(model_lib: impl AsRef<Path>, backend: &str) -> Result<Self> {
        Self::from_runtime(Runtime::open_model_lib(model_lib.as_ref(), backend)?)
    }

    fn from_runtime(rt: Runtime) -> Result<Self> {
        let graph = rt
            .graphs()
            .iter()
            .position(|g| g.name.contains("vits") || g.outputs.iter().any(|t| t.name == "audio"))
            .or_else(|| (!rt.graphs().is_empty()).then_some(0))
            .ok_or_else(|| anyhow!("context binary 里没有图"))?;

        // 打开时就核对一次图契约，别等到推理时才炸。
        let g = &rt.graphs()[graph];
        for (name, elements) in [
            ("codes", NUM_CODES as i64),
            ("text", TEXT_LEN as i64),
            ("refer", (REFER_BINS * REFER_FRAMES) as i64),
            ("sv_emb", SV_DIM as i64),
        ] {
            let t = g.inputs.iter().find(|t| t.name == name).ok_or_else(|| {
                anyhow!(
                    "图 {:?} 缺少输入 {name}（有：{}）",
                    g.name,
                    names(&g.inputs)
                )
            })?;
            if t.elements != elements {
                return Err(anyhow!(
                    "图 {:?} 的输入 {name} 有 {} 个元素，程序按 {elements} 准备（形状不匹配，需要重新导出）",
                    g.name,
                    t.elements
                ));
            }
        }
        if !g.outputs.iter().any(|t| t.name == "audio") {
            return Err(anyhow!(
                "图 {:?} 没有 audio 输出（有：{}）",
                g.name,
                names(&g.outputs)
            ));
        }
        Ok(Self { rt, graph })
    }

    /// 按 `AHA_SOVITS_QNN` 环境变量尝试打开。
    ///
    /// 值是 `lib<name>.so` 时按模型库打开（SoVITS 的 float32 IO 只有这条路能跑），
    /// 其余按 context binary 打开。
    ///
    /// `AHA_SOVITS_QNN_BACKEND` / `AHA_SOVITS_QNN_SYSTEM` 可覆盖设备上的库名，
    /// 默认 `libQnnHtp.so` / `libQnnSystem.so`。
    pub fn from_env() -> Option<Result<Self>> {
        let path = std::env::var("AHA_SOVITS_QNN").ok()?;
        if path.trim().is_empty() {
            return None;
        }
        let path = path.trim();
        let backend =
            std::env::var("AHA_SOVITS_QNN_BACKEND").unwrap_or_else(|_| "libQnnHtp.so".to_string());
        let system = std::env::var("AHA_SOVITS_QNN_SYSTEM")
            .unwrap_or_else(|_| "libQnnSystem.so".to_string());
        Some(if is_model_lib(Path::new(path)) {
            Self::open_model_lib(path, &backend)
        } else {
            Self::open(path, &backend, &system)
        })
    }

    /// 是否可用（垫片 + context binary 都到位）。
    pub fn is_available() -> bool {
        qnn::shim_path().is_some() && std::env::var("AHA_SOVITS_QNN").is_ok()
    }

    /// 这份 codes 能否交给 QNN 图跑（定长约束）。
    pub fn can_run(&self, num_codes: usize) -> bool {
        num_codes <= NUM_CODES
    }

    /// codes/text/refer/sv_emb → 波形 `(1, 1, N)` 的展平数据。
    ///
    /// - `codes`：语义 codes，长度 ≤ [`NUM_CODES`]（不足补 0）；
    /// - `text_ids`：音素 id，长度 ≤ [`TEXT_LEN`]（不足补 0）；
    /// - `refer`：**已按 `[bins, frames]` 行主序**排好的参考谱，长度必须
    ///   `bins * frames` 且 `bins <= REFER_BINS`、`frames <= REFER_FRAMES`；
    /// - `sv`：说话人向量，长度 ≤ [`SV_DIM`]。
    pub fn synthesize(
        &self,
        codes: &[u32],
        text_ids: &[u32],
        refer: &[f32],
        refer_bins: usize,
        refer_frames: usize,
        sv: &[f32],
    ) -> Result<Vec<f32>> {
        if codes.len() > NUM_CODES {
            return Err(anyhow!(
                "SoVITS QNN 图只接受 {NUM_CODES} 帧 codes，收到 {}",
                codes.len()
            ));
        }
        if text_ids.len() > TEXT_LEN {
            return Err(anyhow!(
                "SoVITS QNN 图只接受 {TEXT_LEN} 个音素，收到 {}",
                text_ids.len()
            ));
        }
        if refer_bins > REFER_BINS || refer_frames > REFER_FRAMES {
            return Err(anyhow!(
                "参考谱 {refer_bins}×{refer_frames} 超过图上限 {REFER_BINS}×{REFER_FRAMES}"
            ));
        }
        if refer.len() != refer_bins * refer_frames {
            return Err(anyhow!(
                "参考谱长度 {} 与 {refer_bins}×{refer_frames} 不一致",
                refer.len()
            ));
        }

        let g = &self.rt.graphs()[self.graph];
        // 按图里声明的输入顺序准备缓冲区（`Runtime::run` 按下标对应）。
        let mut inputs: Vec<Vec<u8>> = Vec::with_capacity(g.inputs.len());
        for t in &g.inputs {
            let buf = match t.name.as_str() {
                "codes" => ints_to_bytes(codes, NUM_CODES),
                "text" => ints_to_bytes(text_ids, TEXT_LEN),
                "refer" => {
                    let mut v = vec![0f32; REFER_BINS * REFER_FRAMES];
                    for b in 0..refer_bins {
                        v[b * REFER_FRAMES..b * REFER_FRAMES + refer_frames]
                            .copy_from_slice(&refer[b * refer_frames..(b + 1) * refer_frames]);
                    }
                    f32_to_bytes(&v)
                }
                "sv_emb" => {
                    let mut v = vec![0f32; SV_DIM];
                    let n = sv.len().min(SV_DIM);
                    v[..n].copy_from_slice(&sv[..n]);
                    f32_to_bytes(&v)
                }
                other => return Err(anyhow!("图里出现了未知输入 {other}")),
            };
            let want = self.rt.bytes_per_element(t.data_type) * t.elements as usize;
            if buf.len() != want {
                return Err(anyhow!(
                    "输入 {} 准备了 {} 字节，图要 {want} 字节",
                    t.name,
                    buf.len()
                ));
            }
            inputs.push(buf);
        }

        let mut outputs: Vec<Vec<u8>> = g
            .outputs
            .iter()
            .map(|t| vec![0u8; self.rt.bytes_per_element(t.data_type) * t.elements as usize])
            .collect();
        self.rt.run(self.graph, &mut inputs, &mut outputs)?;

        let out = g
            .outputs
            .iter()
            .position(|t| t.name == "audio")
            .ok_or_else(|| anyhow!("图里没有 audio 输出"))?;
        Ok(bytes_to_f32(&outputs[out]))
    }
}

/// 路径是不是模型库（`.so`）：SoVITS 的 float32 IO 只能走模型库，其余按 context binary。
fn is_model_lib(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("so"))
}

fn names(ts: &[crate::models::common::runtime::qnn::TensorInfo]) -> String {
    ts.iter()
        .map(|t| format!("{} {:?}", t.name, t.dims))
        .collect::<Vec<_>>()
        .join(", ")
}

fn ints_to_bytes(src: &[u32], fixed: usize) -> Vec<u8> {
    let mut v = vec![0i32; fixed];
    for (dst, s) in v.iter_mut().zip(src.iter()) {
        *dst = *s as i32;
    }
    let mut out = Vec::with_capacity(fixed * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}

fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// 把 aha 的线性谱 `(1, freq, T)` 收敛成 QNN 图要的形状。
///
/// 上游是 `ge = ref_enc(refer[:, :704] * mask, mask)`，所以只取前 704 个频点，
/// 帧数超过 704 时截断（官方参考音频都很短）。
pub fn trim_spec(spec: &[f32], freq: usize, frames: usize) -> (Vec<f32>, usize, usize) {
    let bins = freq.min(REFER_BINS);
    let f = frames.min(REFER_FRAMES);
    let mut out = Vec::with_capacity(bins * f);
    for b in 0..bins {
        out.extend_from_slice(&spec[b * frames..b * frames + f]);
    }
    (out, bins, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ints_are_padded_and_little_endian() {
        let b = ints_to_bytes(&[1, 2], 4);
        assert_eq!(b.len(), 16);
        assert_eq!(&b[0..4], &1i32.to_le_bytes());
        assert_eq!(&b[8..12], &0i32.to_le_bytes());
    }

    #[test]
    fn trim_spec_keeps_row_major_layout() {
        // 3 个频点 × 2 帧 → 每行 2 个数
        let spec = vec![1., 2., 3., 4., 5., 6.];
        let (out, bins, frames) = trim_spec(&spec, 3, 2);
        assert_eq!((bins, frames), (3, 2));
        assert_eq!(out, spec);
    }

    #[test]
    fn model_lib_paths_are_recognised_by_extension() {
        assert!(is_model_lib(Path::new("/data/local/tmp/vitsF/libvitsF.so")));
        assert!(is_model_lib(Path::new("LIBVITSF.SO")));
        assert!(!is_model_lib(Path::new("vitsF_ctx.bin")));
        assert!(!is_model_lib(Path::new("libvitsF.so.bin")));
    }

    #[test]
    fn trim_spec_drops_extra_frequencies_and_frames() {
        let (out, bins, frames) = trim_spec(&vec![0.5; 800 * 800], 800, 800);
        assert_eq!((bins, frames), (REFER_BINS, REFER_FRAMES));
        assert_eq!(out.len(), REFER_BINS * REFER_FRAMES);
    }
}

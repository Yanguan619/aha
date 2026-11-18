//! candle 侧"运行时"设施：CUDA graph 抓图 / 重放。
//!
//! 和 `runtime/qnn.rs`、`runtime/llama_cpp/` 一样，这一层不认识具体模型：模型只负责把自己的
//! 一段前向编排成图的形状，复用这里的抓图、静态缓冲和重放原语。

/// 抓图 / 重放 / 静态缓冲：把一段前向录成 CUDA graph，之后每步只换输入再重放。
///
/// 依赖 candle 的 `cuda` 后端（`--features cuda`），其他设备上没有意义。
#[cfg(feature = "cuda")]
pub mod cuda_graph;

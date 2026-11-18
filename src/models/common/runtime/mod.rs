//! 推理运行时后端：把"跑图 / 跑生成"交给外部 C 运行时。
//!
//! 这一层**不认识任何具体模型**：QNN 只按 context binary 里的元数据跑静态图，llama.cpp 只把
//! 文本生成交给它的 kernel（能解析 candle 不支持的新架构 / 新量化，比如 STQ）。模型自己的
//! prompt 模板、特殊 token 由模型模块通过钩子注入（见 `PromptRenderer`），这里不写
//! `if 模型 == X` 分支。

/// candle 自己的运行时设施（CUDA graph 抓图 / 重放，只在 `cuda` 下有意义）。
#[cfg(feature = "cuda")]
pub mod candle;

/// QNN 运行时：加载 QAIRT context binary 跑任意静态图。C 垫片在 `native/qnn/`。
///
/// 桌面构建默认不编（`--features qnn` 才编），实际接线只在 Android 上成立。
#[cfg(feature = "qnn")]
pub mod qnn;

/// llama.cpp 后端：把生成交给 llama.cpp，能跑 candle 不支持的新架构 / 新量化（如 STQ）。
#[cfg(feature = "llama")]
pub mod llama_cpp;

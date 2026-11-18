// 端点请求 / 响应类型，按端点一文件。
//
// 这里装的全是 **HTTP payload**：库和 exec 也用得上的（`chat` / `shared`）给 `pub`，
// 只有服务端端点用得上的给 `pub(crate)`。`#[allow(unused)]` 是因为某些 feature 组合
// 下对应的端点不会被编译（比如没开 sherpa 就没有 `/audio/speech` 的 TTS 分支）。
#[allow(dead_code)]
pub mod actions;
#[allow(dead_code)]
pub mod asr;
pub mod chat;
#[allow(dead_code)]
pub mod embedding;
#[allow(dead_code)]
pub mod health;
#[allow(dead_code)]
pub mod models;
#[allow(dead_code)]
pub mod rerank;
pub mod shared;
#[allow(dead_code)]
pub mod shutdown;
#[allow(dead_code)]
pub mod speech;

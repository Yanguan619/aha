// 端点请求 / 响应类型，按端点一文件。
//
// 这里装的全是 **HTTP payload**：库和 exec 也用得上的（`chat` / `shared`）给 `pub`，
// 只有服务端端点用得上的给 `pub(crate)`。`#[allow(unused)]` 是因为某些 feature 组合
// 下对应的端点不会被编译（比如没开 sherpa 就没有 `/v1/tts`）。
#[allow(unused)]
pub mod actions;
#[allow(unused)]
pub mod asr;
pub mod chat;
#[allow(unused)]
pub mod decisions;
#[allow(unused)]
pub mod embedding;
#[allow(unused)]
pub mod health;
#[allow(unused)]
pub mod models;
#[allow(unused)]
pub mod rerank;
pub mod shared;
#[allow(unused)]
pub mod shutdown;
#[allow(unused)]
pub mod tts;

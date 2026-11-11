//! `GET /health` 的响应体。
//!
//! 桌宠靠它判断"每一路好了没"：`models` 里出现哪个角色，就说明哪个端点能用。

use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct HealthResponse {
    pub(crate) status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error: Option<String>,
    /// 已加载的角色 / 模型 key。
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) models: Vec<String>,
}

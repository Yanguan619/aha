//! `GET /models` 与 `GET /v1/models` 的响应体（OpenAI 兼容）。

use serde::Serialize;

/// OpenAI-compatible model object.
#[derive(Debug, Serialize)]
pub(crate) struct ModelObject {
    pub(crate) id: String,
    pub(crate) object: String,
    pub(crate) created: Option<i64>,
    pub(crate) owned_by: String,
}

/// OpenAI-compatible models list response.
#[derive(Debug, Serialize)]
pub(crate) struct ModelsListResponse {
    pub(crate) object: String,
    pub(crate) data: Vec<ModelObject>,
}

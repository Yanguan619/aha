//! `POST /v1/decisions` 的请求体（桌宠 → Laya 决策）。
//!
//! 只要一句话：判断该走 VLA（身体动作）还是 LLM（对话）。

use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
pub(crate) struct DecisionRequest {
    #[serde(default)]
    pub(crate) text: String,
}

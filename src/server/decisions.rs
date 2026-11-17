//! `POST /v1/decisions` —— Laya（System-1 决策模型）端点，**标准 Jev 形状**。
//!
//! 和 Ollama 的 `/v1/systemone` 是同一套（TypeSafe 的 Jev API）：题面（`questions`）
//! 由调用方在请求里给，aha 只负责把 `state` + 题面交给模型、把 `answers` 原样返回。
//!
//! ```text
//! POST /v1/decisions
//!   {"state":"跳一下",
//!    "questions":{"route":{"type":"choice","instructions":"…",
//!                          "criteria":{"vla":"…","llm":"…"}}}}
//!   → {"model":"convaiinnovations/laya-multilingual",
//!      "answers":{"route":{"type":"choice","choice":"vla",
//!                          "probabilities":{"vla":0.9722,"llm":0.0278},
//!                          "confidence":0.8168}},
//!      "usage":{"input_tokens":30,"output_tokens":0}}
//! ```
//!
//! 问题类型支持 `choice` / `noul` / `score`，一次最多 [`MAX_QUESTIONS`] 个
//! （受 sys1 batcher 的批内槽位限制；Jev 标准本身允许 1–64）。
//!
//! **阈值、关键词短路、答案改写都是调用方策略**，不在这里——`LayaRouteModel::route`
//! 那套三层判定现在只给 `aha probe --selftest` 和启动预热用。
//!
//! 同一个模型一次只放一个请求进推理（sys1 不是并发安全的），所以拿写锁。

use axum::{
    Json,
    body::Bytes,
    http::StatusCode,
    response::{IntoResponse, Response},
};

use crate::server::error;
use aha::models::laya::MAX_QUESTIONS;

pub async fn handle_route(body: Bytes) -> Response {
    let request: sys1::schema::DecisionRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => {
            return error::contract_error(
                StatusCode::BAD_REQUEST,
                format!("body 必须是 Jev 形状 {{state,questions}}：{error}"),
            );
        }
    };
    if let Some(message) = check_decision_request(&request) {
        return error::contract_error(StatusCode::BAD_REQUEST, message);
    }

    let Some(model_ref) = crate::server::api::model_by_key("system1")
        .or_else(|| crate::server::api::model_by_key("laya"))
    else {
        return error::contract_error(StatusCode::SERVICE_UNAVAILABLE, "模型没有初始化");
    };
    let guard = model_ref.write().await;
    let Some(model) = guard.instance.route_model() else {
        return error::contract_error(
            StatusCode::BAD_REQUEST,
            format!(
                "当前模型 {} 不是 Laya，无法判定",
                guard.which_model.as_string()
            ),
        );
    };
    // 模型挂了要如实报错：桌宠 Go 侧据此 fail-open，别把失败伪装成一个答案
    match model.predict(request).await {
        Ok(response) => Json(response).into_response(),
        Err(message) => error::contract_error(StatusCode::INTERNAL_SERVER_ERROR, message),
    }
}

/// 标准只要求两点：`state` 非空、`questions` 非空。问题数上限提在这里，
/// 免得超限的请求落到 batcher 里变成 500。
fn check_decision_request(request: &sys1::schema::DecisionRequest) -> Option<String> {
    if request.questions.is_empty() {
        return Some(format!(
            "questions 不能为空（1–{MAX_QUESTIONS} 个命名问题）"
        ));
    }
    if request.questions.len() > MAX_QUESTIONS {
        return Some(format!(
            "questions 最多 {MAX_QUESTIONS} 个（收到 {}）",
            request.questions.len()
        ));
    }
    let state_is_empty = match &request.state {
        serde_json::Value::Null => true,
        serde_json::Value::String(text) => text.trim().is_empty(),
        serde_json::Value::Array(items) => items.is_empty(),
        serde_json::Value::Object(map) => map.is_empty(),
        _ => false,
    };
    if state_is_empty {
        return Some("state 不能为空".to_owned());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request(body: &str) -> sys1::schema::DecisionRequest {
        serde_json::from_slice(body.as_bytes()).expect("测试体的 JSON 是手写的")
    }

    #[test]
    fn rejects_empty_state_or_questions() {
        assert!(check_decision_request(&request(r#"{"state":"走","questions":{}}"#)).is_some());
        assert!(
            check_decision_request(&request(
                r#"{"state":"  ","questions":{"q":{"type":"noul"}}}"#
            ))
            .is_some()
        );
        assert!(check_decision_request(&request(r#"{"state":{},"questions":{"q":{}}}"#)).is_some());
    }

    #[test]
    fn rejects_too_many_questions() {
        let body = r#"{"state":"hi","questions":{"a":{},"b":{},"c":{},"d":{},"e":{}}}"#;
        let message = check_decision_request(&request(body)).unwrap();
        assert!(message.contains("最多 4 个"), "got {message}");
    }

    #[test]
    fn accepts_standard_shape() {
        let ok = request(
            r#"{"model":"laya","state":"跳一下","questions":{"route":{"type":"choice",
                "instructions":"怎么回应？","criteria":{"vla":"动作","llm":"对话"}}}}"#,
        );
        assert!(check_decision_request(&ok).is_none());
    }
}

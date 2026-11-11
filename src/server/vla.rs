//! 桌宠的 VLA（视觉-语言-动作）端点：`POST /v1/actions`。
//!
//! 请求：`{"instruction":"…","state":[8 个 float],"images":["<base64 jpeg/png>", …]}`
//! 响应：`{"actions":[[7 个 float] × 12],"horizon":12,"action_dim":7}`
//!
//! `laya`/`asr`/`tts` 的接口是"文本/音频"；VLA 的输出是**动作 chunk**，所以单独一条。

use axum::{
    Json,
    body::Bytes,
    http::StatusCode,
    response::{IntoResponse, Response},
};
use base64::{Engine, prelude::BASE64_STANDARD};
use serde_json::json;

use crate::params::actions::ActionsRequest;
use crate::server::api::model_by_key;

pub async fn handle_actions(body: Bytes) -> Response {
    if body.is_empty() {
        return fail(StatusCode::BAD_REQUEST, "请求体是空的");
    }
    let req: ActionsRequest = match serde_json::from_slice(&body) {
        Ok(req) => req,
        Err(error) => {
            return fail(StatusCode::BAD_REQUEST, format!("JSON 不对：{error}"));
        }
    };
    let instruction = req.instruction.trim();
    if instruction.is_empty() {
        return fail(StatusCode::BAD_REQUEST, "instruction 不能为空");
    }
    let mut images = Vec::with_capacity(req.images.len());
    for image in &req.images {
        let raw = image.split(',').next_back().unwrap_or(image);
        match BASE64_STANDARD.decode(raw) {
            Ok(bytes) => images.push(bytes),
            Err(error) => {
                return fail(
                    StatusCode::BAD_REQUEST,
                    format!("图片 base64 解不开：{error}"),
                );
            }
        }
    }

    let Some(model_ref) = model_by_key("vla").or_else(|| model_by_key("OpenBMB/TurboVLA")) else {
        return fail(
            StatusCode::SERVICE_UNAVAILABLE,
            "VLA 模型没有加载（用 aha multi --vla <dir> 启动）",
        );
    };
    let guard = model_ref.write().await;
    let Some(model) = guard.instance.vla_model() else {
        return fail(
            StatusCode::BAD_REQUEST,
            format!(
                "当前模型 {} 不是 TurboVLA，无法输出动作",
                guard.which_model.as_string()
            ),
        );
    };
    let refs: Vec<&[u8]> = images.iter().map(|bytes| bytes.as_slice()).collect();
    match model.predict(&refs, instruction, &req.state) {
        Ok(actions) => {
            let horizon = model.horizon();
            let action_dim = model.action_dim();
            // 模型在 CUDA 上跑 bf16，动作张量要先转回 f32 才能 to_vec2。
            match actions
                .squeeze(0)
                .and_then(|t| t.to_dtype(candle_core::DType::F32))
                .and_then(|t| t.to_vec2::<f32>()) {
                Ok(rows) => Json(json!({
                    "actions": rows,
                    "horizon": horizon,
                    "action_dim": action_dim,
                }))
                .into_response(),
                Err(error) => fail(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("动作张量转换失败：{error}"),
                ),
            }
        }
        Err(error) => fail(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("VLA 推理失败：{error:#}"),
        ),
    }
}

fn fail(status: StatusCode, message: impl Into<String>) -> Response {
    (status, Json(json!({ "error": message.into() }))).into_response()
}

//! realtime 协议层：事件 envelope、item 序列化、请求体拼装。
//!
//! 纯 JSON 形状转换，不碰 socket、不碰状态机；会话与编排在父模块 `mod.rs`。

use serde_json::{Value, json};

use aha::params::chat::ChatCompletionParameters;

use super::{Item, SessionConfig};

/// 一帧事件的外壳：`event_id` + `type` + 载荷。
pub(super) fn envelope(kind: &str, body: Value) -> Value {
    let mut map = match body {
        Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    map.insert("event_id".to_owned(), json!(new_id("event")));
    map.insert("type".to_owned(), json!(kind));
    Value::Object(map)
}

/// 新 id，形状和 OpenAI 一样：`<prefix>_<32 位 hex>`。
pub(super) fn new_id(prefix: &str) -> String {
    format!("{prefix}_{}", uuid::Uuid::new_v4().simple())
}

pub(super) fn response_shell(response_id: &str, cfg: &SessionConfig, status: &str) -> Value {
    json!({
        "id": response_id,
        "object": "realtime.response",
        "status": status,
        "status_details": Value::Null,
        "output": [],
        "usage": Value::Null,
        "conversation_id": Value::Null,
        "modalities": cfg.modalities,
        "voice": cfg.voice,
        "output_audio_format": "pcm16",
        "temperature": cfg.temperature,
        "max_output_tokens": cfg.max_output_tokens.map(|n| json!(n)).unwrap_or_else(|| json!("inf")),
    })
}

pub(super) fn item_to_json(item: &Item) -> Value {
    json!({
        "id": item.id,
        "object": "realtime.item",
        "type": "message",
        "status": "completed",
        "role": item.role,
        "content": item.content,
    })
}

/// 把 `conversation.item.create` 里的 content 归一化成 chat 形状的 content。
///
/// 认这几种：纯字符串、`input_text`/`text`、`input_image`/`image_url`
/// （视频就是按帧塞 `input_image`）。返回 `None` 表示一个字都没解析出来。
pub(super) fn normalize_content(content: &Value) -> Option<Value> {
    if let Some(text) = content.as_str() {
        let text = text.trim();
        return (!text.is_empty()).then(|| Value::String(text.to_owned()));
    }
    let parts = content.as_array()?;
    let mut normalized = Vec::new();
    let mut has_media = false;
    for part in parts {
        let kind = part.get("type").and_then(Value::as_str).unwrap_or_default();
        match kind {
            "input_text" | "text" | "output_text" => {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    normalized.push(json!({ "type": "text", "text": text }));
                }
            }
            "input_image" | "image_url" => {
                // OpenAI Realtime 是 `image_url: "data:..."`；chat 形状是 `{url, detail}`。
                let url = part
                    .get("image_url")
                    .and_then(|image| {
                        image.as_str().map(ToOwned::to_owned).or_else(|| {
                            image
                                .get("url")
                                .and_then(Value::as_str)
                                .map(ToOwned::to_owned)
                        })
                    })
                    .or_else(|| {
                        part.get("url")
                            .and_then(Value::as_str)
                            .map(ToOwned::to_owned)
                    })?;
                let detail = part
                    .get("image_url")
                    .and_then(|image| image.get("detail"))
                    .and_then(Value::as_str)
                    .unwrap_or("auto");
                normalized.push(json!({
                    "type": "image_url",
                    "image_url": { "url": url, "detail": detail },
                }));
                has_media = true;
            }
            _ => {}
        }
    }
    if normalized.is_empty() {
        return None;
    }
    // 没有多模态内容时退化成纯字符串，兼容只吃字符串 content 的模型。
    if has_media {
        Some(Value::Array(normalized))
    } else {
        let text = normalized
            .iter()
            .filter_map(|part| part.get("text").and_then(Value::as_str))
            .collect::<Vec<_>>()
            .join("\n");
        Some(Value::String(text))
    }
}

/// 会话 + 指令 → chat 请求体（走 `serde_json` 填字段，省得手工构造 30 多个 Option）。
pub(super) fn build_request(
    cfg: &SessionConfig,
    items: &[Item],
) -> Result<ChatCompletionParameters, String> {
    let mut messages: Vec<Value> = Vec::new();
    if !cfg.instructions.trim().is_empty() {
        messages.push(json!({ "role": "system", "content": cfg.instructions }));
    }
    for item in items {
        messages.push(json!({ "role": item.role, "content": item.content }));
    }
    if messages.is_empty() {
        return Err("对话是空的".to_owned());
    }
    let mut body = json!({
        "model": cfg.model.clone().unwrap_or_default(),
        "messages": messages,
        "stream": true,
    });
    if let Some(temperature) = cfg.temperature {
        body["temperature"] = json!(temperature);
    }
    if let Some(max_tokens) = cfg.max_output_tokens {
        body["max_completion_tokens"] = json!(max_tokens);
    }
    // 语音对话里不要思考链：Qwen3 默认会先想几千 token 才开口。
    body["enable_thinking"] = json!(cfg.enable_thinking);
    serde_json::from_value(body).map_err(|error| format!("请求体构造失败：{error}"))
}

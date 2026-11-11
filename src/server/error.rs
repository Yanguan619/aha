//! Unified OpenAI-compatible error responses.
//!
//! Every error leaving the HTTP layer — whether raised by a handler, an extractor
//! or the router itself — is serialized as:
//!
//! ```json
//! { "error": { "message": "...", "type": "...", "param": "...", "code": "..." } }
//! ```

use axum::{
    Json,
    extract::Request,
    http::{StatusCode, header},
    middleware::Next,
    response::{IntoResponse, Response},
};
use serde::Serialize;

use crate::params::shared::{ApiErrorBody, ApiErrorDetail};

/// Build an OpenAI-compatible error payload.
pub(crate) fn api_error(
    status: StatusCode,
    error_type: &str,
    message: impl Into<String>,
    param: Option<&str>,
    code: Option<&str>,
) -> (StatusCode, Json<serde_json::Value>) {
    let body = ApiErrorBody {
        error: ApiErrorDetail {
            message: message.into(),
            error_type: error_type.to_string(),
            param: param.map(str::to_string),
            code: code.map(str::to_string),
        },
    };
    (
        status,
        Json(serde_json::to_value(body).expect("error body is always serializable")),
    )
}

/// 400 — the request itself is malformed or carries invalid parameters.
pub(crate) fn invalid_request(
    message: impl Into<String>,
    param: Option<&str>,
    code: Option<&str>,
) -> (StatusCode, Json<serde_json::Value>) {
    api_error(
        StatusCode::BAD_REQUEST,
        "invalid_request_error",
        message,
        param,
        code,
    )
}

/// 500 — the request was valid but the server failed to fulfil it.
pub(crate) fn internal_error(message: impl Into<String>) -> (StatusCode, Json<serde_json::Value>) {
    api_error(
        StatusCode::INTERNAL_SERVER_ERROR,
        "server_error",
        message,
        None,
        Some("internal_error"),
    )
}

/// 503 — the model has not been loaded yet.
pub(crate) fn model_not_loaded() -> (StatusCode, Json<serde_json::Value>) {
    api_error(
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable",
        "Model not initialized",
        None,
        Some("model_not_loaded"),
    )
}

/// 404 fallback for unknown routes.
pub(crate) async fn fallback_not_found(uri: axum::http::Uri) -> Response {
    let path = uri.path().to_string();
    api_error(
        StatusCode::NOT_FOUND,
        "invalid_request_error",
        format!("Invalid URL ({}): no such endpoint", path),
        None,
        Some("unknown_endpoint"),
    )
    .into_response()
}

/// 405 fallback for known paths used with the wrong HTTP method.
pub(crate) async fn fallback_method_not_allowed(
    method: axum::http::Method,
    uri: axum::http::Uri,
) -> Response {
    let path = uri.path().to_string();
    api_error(
        StatusCode::METHOD_NOT_ALLOWED,
        "invalid_request_error",
        format!("Method {method} is not allowed for {path}"),
        None,
        Some("method_not_allowed"),
    )
    .into_response()
}

/// Convert a JSON extractor rejection into the OpenAI error envelope.
///
/// `422 Unprocessable Entity` (well-formed JSON with invalid fields) is mapped to
/// `400 Bad Request`, which is what OpenAI clients expect; oversized bodies keep
/// their `413` status.
pub(crate) fn from_json_rejection(
    rejection: axum::extract::rejection::JsonRejection,
) -> (StatusCode, Json<serde_json::Value>) {
    let origin = rejection.status();
    let (status, message, code) = match origin {
        StatusCode::PAYLOAD_TOO_LARGE => (
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body is too large".to_string(),
            "payload_too_large",
        ),
        StatusCode::UNSUPPORTED_MEDIA_TYPE => (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be application/json".to_string(),
            "unsupported_media_type",
        ),
        StatusCode::UNPROCESSABLE_ENTITY => (
            StatusCode::BAD_REQUEST,
            rejection.body_text(),
            "invalid_body",
        ),
        _ => (
            StatusCode::BAD_REQUEST,
            rejection.body_text(),
            "invalid_body",
        ),
    };
    api_error(status, "invalid_request_error", message, None, Some(code))
}

/// Convert a multipart extractor rejection into the OpenAI error envelope.
pub(crate) fn from_multipart_rejection(
    rejection: axum::extract::multipart::MultipartRejection,
) -> (StatusCode, Json<serde_json::Value>) {
    let origin = rejection.status();
    let (status, message, code) = match origin {
        StatusCode::PAYLOAD_TOO_LARGE => (
            StatusCode::PAYLOAD_TOO_LARGE,
            "Request body is too large".to_string(),
            "payload_too_large",
        ),
        StatusCode::UNSUPPORTED_MEDIA_TYPE => (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "Content-Type must be multipart/form-data".to_string(),
            "unsupported_media_type",
        ),
        _ => (
            StatusCode::BAD_REQUEST,
            rejection.body_text(),
            "invalid_multipart",
        ),
    };
    api_error(status, "invalid_request_error", message, None, Some(code))
}

/// Last-resort normaliser: any error response that is not already the
/// OpenAI-compatible JSON envelope is rewritten into one.
///
/// This covers framework-generated responses such as `413 Payload Too Large`
/// from `DefaultBodyLimit` or `415 Unsupported Media Type`.
pub(crate) async fn normalize_error_response(request: Request, next: Next) -> Response {
    let response = next.run(request).await;
    let status = response.status();
    if !status.is_client_error() && !status.is_server_error() {
        return response;
    }

    let already_json = response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .is_some_and(|value| value.to_ascii_lowercase().contains("application/json"));
    if already_json {
        return response;
    }

    let error_type = if status.is_server_error() {
        "server_error"
    } else {
        "invalid_request_error"
    };
    let message = match status {
        StatusCode::PAYLOAD_TOO_LARGE => "Request body is too large".to_string(),
        StatusCode::UNSUPPORTED_MEDIA_TYPE => {
            "Content-Type is not supported for this endpoint".to_string()
        }
        StatusCode::UNPROCESSABLE_ENTITY => "Request body could not be parsed".to_string(),
        _ => status
            .canonical_reason()
            .unwrap_or("Request failed")
            .to_string(),
    };

    api_error(status, error_type, message, None, None).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_envelope_shape() {
        let (status, json) = invalid_request("Audio file is required", None, Some("missing_file"));
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let value = json.0;
        assert_eq!(value["error"]["message"], "Audio file is required");
        assert_eq!(value["error"]["type"], "invalid_request_error");
        assert_eq!(value["error"]["code"], "missing_file");
        // `param` is omitted when unset.
        assert!(value["error"].get("param").is_none());
    }

    #[test]
    fn test_error_includes_param_when_set() {
        let (_, json) = invalid_request("bad temperature", Some("temperature"), None);
        assert_eq!(json.0["error"]["param"], "temperature");
    }

    #[test]
    fn test_model_not_loaded_shape() {
        let (status, json) = model_not_loaded();
        assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(json.0["error"]["type"], "service_unavailable");
        assert_eq!(json.0["error"]["code"], "model_not_loaded");
    }
}

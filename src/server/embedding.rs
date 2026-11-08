use axum::{Json, extract::rejection::JsonRejection, http::StatusCode};
use serde_json::Value;

use crate::{
    params::embedding::{EmbeddingData, EmbeddingRequest, EmbeddingResponse},
    server::{api::MODEL, error},
};

fn parse_embedding_input(input: &Value) -> anyhow::Result<Vec<String>> {
    match input {
        Value::String(s) => Ok(vec![s.clone()]),
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for v in arr {
                let s = v.as_str().ok_or_else(|| {
                    anyhow::anyhow!("embedding input array must contain only strings")
                })?;
                out.push(s.to_string());
            }
            if out.is_empty() {
                return Err(anyhow::anyhow!("embedding input cannot be empty"));
            }
            Ok(out)
        }
        _ => Err(anyhow::anyhow!(
            "embedding input must be a string or an array of strings"
        )),
    }
}

pub(crate) async fn embeddings(
    payload: Result<Json<EmbeddingRequest>, JsonRejection>,
) -> (StatusCode, Json<Value>) {
    let req = match payload {
        Ok(Json(req)) => req,
        Err(rejection) => {
            return error::from_json_rejection(rejection);
        }
    };

    let texts = match parse_embedding_input(&req.input) {
        Ok(v) => v,
        Err(e) => {
            return error::invalid_request(e.to_string(), Some("input"), Some("invalid_input"));
        }
    };
    let model_ref = match MODEL.get().cloned() {
        Some(v) => v,
        None => return error::model_not_loaded(),
    };
    let mut guard = model_ref.write().await;
    let embeddings = match guard.instance.embedding(&texts) {
        Ok(v) => v,
        Err(e) => {
            return error::invalid_request(e.to_string(), Some("input"), Some("embedding_failed"));
        }
    };
    let model_name = guard.which_model.as_string();
    let data = embeddings
        .into_iter()
        .enumerate()
        .map(|(index, embedding)| EmbeddingData {
            object: "embedding".to_string(),
            index,
            embedding,
        })
        .collect::<Vec<_>>();
    let response = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: model_name,
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap()),
    )
}

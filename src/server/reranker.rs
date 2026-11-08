use axum::{Json, extract::rejection::JsonRejection, http::StatusCode};
use serde_json::Value;

use crate::{
    params::rerank::{RerankRequest, RerankResponse, RerankResult},
    server::{api::stored_model, error},
};

fn validate_rerank_input(query: &str, documents: &[String]) -> anyhow::Result<()> {
    if query.trim().is_empty() {
        return Err(anyhow::anyhow!("rerank query cannot be empty"));
    }
    if documents.is_empty() {
        return Err(anyhow::anyhow!("rerank documents cannot be empty"));
    }
    if documents.iter().any(|doc| doc.trim().is_empty()) {
        return Err(anyhow::anyhow!(
            "rerank documents cannot contain empty strings"
        ));
    }
    Ok(())
}

pub(crate) async fn rerank(
    payload: Result<Json<RerankRequest>, JsonRejection>,
) -> (StatusCode, Json<Value>) {
    let req = match payload {
        Ok(Json(req)) => req,
        Err(rejection) => {
            return error::from_json_rejection(rejection);
        }
    };

    if let Err(e) = validate_rerank_input(&req.query, &req.documents) {
        return error::invalid_request(e.to_string(), None, Some("invalid_input"));
    }

    let model_ref = match stored_model() {
        Some(v) => v,
        None => return error::model_not_loaded(),
    };

    let mut guard = model_ref.write().await;
    let scores = match guard.instance.rerank(&req.query, &req.documents) {
        Ok(v) => v,
        Err(e) => {
            return error::invalid_request(e.to_string(), None, Some("rerank_failed"));
        }
    };

    let mut results = scores
        .into_iter()
        .enumerate()
        .map(|(index, relevance_score)| RerankResult {
            index,
            relevance_score,
            document: req.documents[index].clone(),
        })
        .collect::<Vec<_>>();
    results.sort_by(|a, b| b.relevance_score.total_cmp(&a.relevance_score));
    if let Some(top_n) = req.top_n {
        results.truncate(top_n.min(results.len()));
    }

    let response = RerankResponse {
        object: "list".to_string(),
        model: guard.which_model.as_string(),
        results,
    };
    (
        StatusCode::OK,
        Json(serde_json::to_value(response).unwrap()),
    )
}

//! `POST /shutdown` 的响应体。

use serde::Serialize;

#[derive(Debug, Serialize)]
pub(crate) struct ShutdownResponse {
    pub(crate) message: String,
}

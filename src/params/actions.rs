//! `POST /v1/actions` 的请求体（桌宠 → VLA）。
//!
//! 一帧画面（每个视角一张图）+ 一句自然语言指令 + 机器人当前状态，
//! 返回未来 `horizon` 步的动作序列。

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(crate) struct ActionsRequest {
    /// 自然语言指令，比如"把碗放到盘子上"。
    #[serde(default)]
    pub(crate) instruction: String,
    /// 机器人当前状态（LIBERO 是 8 维）。
    #[serde(default)]
    pub(crate) state: Vec<f32>,
    /// 每个视角一张图（JPEG/PNG；容忍 `data:image/...;base64,` 前缀）。
    #[serde(default)]
    pub(crate) images: Vec<String>,
}

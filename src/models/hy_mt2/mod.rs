//! 腾讯混元翻译模型 Hy-MT2 1.8B 的 1.25bit GGUF。
//!
//! 这个权重文件用的是 STQ 自定义量化（`STQ_0` / `STQ1_0`），candle 侧没有对应
//! kernel，只能交给 llama.cpp（需要带 STQ 支持的构建，见 `native/scripts/build-llama.sh`）。
//! 本模块只放模型自己的东西：推荐采样参数 + chat template；权重的仓库 id / 文件名
//! 与下载都在 [`crate::models::mirrors`] 和 [`crate::models::download`]。

/// 官方推荐的采样参数（README 里给的是 temperature 0.7 / top_p 0.6 / top_k 20 /
/// repetition_penalty 1.05 / max_tokens 4096）。`aha run` 的默认值用这一组。
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
pub const DEFAULT_TOP_P: f32 = 0.6;
pub const DEFAULT_TOP_K: usize = 20;
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.05;

/// Hy-MT2 的 chat template（`tokenizer.chat_template` 里的 jinja 原文）：
/// system 消息会被夹在 BOS 与 `<｜hy_place▁holder▁no▁3｜>` 之间，用户轮以
/// `<｜hy_User｜>` 开头，助手轮以 `<｜hy_Assistant｜>` 开头、以
/// `<｜hy_place▁holder▁no▁2｜>` 结束，最后补一个 `<｜hy_Assistant｜>` 让模型续写。
///
/// llama.cpp 只认几种内置模板，所以由本模块按模型自己的模板渲染，再通过
/// [`crate::models::common::runtime::llama_cpp::LlamaCppModel::init_with_prompt_renderer`]
/// 挂到通用 llama.cpp 后端上——后端本身不需要知道 Hy-MT2。
#[cfg(feature = "llama")]
mod chat_template {
    use std::ffi::CString;

    use crate::models::common::runtime::llama_cpp::RenderedPrompt;

    pub const BOS: &str = "<｜hy_begin▁of▁sentence｜>";
    pub const USER: &str = "<｜hy_User｜>";
    pub const ASSISTANT: &str = "<｜hy_Assistant｜>";
    pub const END_TURN: &str = "<｜hy_place▁holder▁no▁2｜>";
    pub const END_SYSTEM: &str = "<｜hy_place▁holder▁no▁3｜>";

    /// 渲染对话。模板文本里已经带了 BOS 特殊 token，`add_special` 固定为 `false`，
    /// 否则 tokenizer 会再补一个 BOS。
    pub fn render(roles: &[CString], contents: &[CString]) -> Option<RenderedPrompt> {
        let mut text = String::from(BOS);
        let mut start = 0;
        if roles
            .first()
            .is_some_and(|r| r.to_string_lossy() == "system")
        {
            text.push_str(&contents[0].to_string_lossy());
            text.push_str(END_SYSTEM);
            start = 1;
        }
        for (role, content) in roles[start..].iter().zip(contents[start..].iter()) {
            match role.to_string_lossy().as_ref() {
                "user" => {
                    text.push_str(USER);
                    text.push_str(&content.to_string_lossy());
                }
                "assistant" => {
                    text.push_str(ASSISTANT);
                    text.push_str(&content.to_string_lossy());
                    text.push_str(END_TURN);
                }
                _ => {}
            }
        }
        text.push_str(ASSISTANT);
        Some(RenderedPrompt {
            text,
            add_special: false,
        })
    }
}

/// 供 `load_gguf_model` 挂到 llama.cpp 后端上的 prompt 渲染器。
#[cfg(feature = "llama")]
pub use chat_template::render as render_prompt;

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "llama")]
    #[test]
    fn renders_hy_chat_template() {
        let roles = vec![
            std::ffi::CString::new("system").unwrap(),
            std::ffi::CString::new("user").unwrap(),
        ];
        let contents = vec![
            std::ffi::CString::new("只翻译，不解释").unwrap(),
            std::ffi::CString::new("Hello").unwrap(),
        ];

        let rendered = render_prompt(&roles, &contents).expect("renderer");

        // BOS 开头、system 段有结束符、user 轮有角色 token、结尾是 Assistant 提示。
        assert!(rendered.text.starts_with(chat_template::BOS));
        assert!(rendered.text.contains(chat_template::END_SYSTEM));
        assert!(rendered.text.contains(chat_template::USER));
        assert!(rendered.text.ends_with(chat_template::ASSISTANT));
        // 模板已含 BOS，不能再让 tokenizer 补。
        assert!(!rendered.add_special);
    }
}

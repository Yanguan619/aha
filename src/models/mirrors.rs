//! 模型 → ModelScope 仓库的唯一真值表。
//!
//! `--model` / `-m` 传的是 [`WhichModel`](crate::models::common::model_mapping::WhichModel)
//! 的 value name，也就是沿用的历史 id。它跟真正托管权重的 ModelScope 仓库不总是同名：
//!
//! - sherpa-onnx 的两个（SenseVoice / Kokoro）是社区镜像，官方 `csukuangfj/...` 在
//!   ModelScope 上没有同名仓库；
//! - TurboVLA / GPT-SoVITS / Qwen3.5 GGUF 是本地转换/挑文件后传上去的镜像。
//!
//! 下载路径（CLI、FFI、各模型模块的 `ensure_downloaded`）都从这里取仓库名，避免同一个
//! 仓库 id 散落在几个地方各写一份、改一处漏一处。
//!
//! `files` 是"推理真正要用的文件"：`Some` 表示只下这些，`None` 表示整个仓库都要
//! （那种仓库里除了权重只有 README / config 这类小文件）。

/// 一个模型在 ModelScope 上的落点。
pub struct Mirror {
    /// 就是 `WhichModel::as_string()` 的值，也是 `-m` 接收的 id、以及本地目录名。
    pub model_id: &'static str,
    /// ModelScope 仓库 id（`<namespace>/<name>`）。
    pub repo: &'static str,
    /// 需要下载的文件；`None` = 整仓下载。
    pub files: Option<&'static [&'static str]>,
}

/// SenseVoice（sherpa-onnx ASR）的 ModelScope 镜像。
pub const SENSE_VOICE_MODEL_ID: &str =
    "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17";
pub const SENSE_VOICE_REPO: &str = "WEAAEW/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17";
/// 只下 int8 那份（fp32 的 model.onnx 有 938 MB，CPU 上没必要）。
pub const SENSE_VOICE_FILES: &[&str] = &["model.int8.onnx", "tokens.txt"];

/// Kokoro（sherpa-onnx TTS）的 ModelScope 镜像。
pub const KOKORO_MODEL_ID: &str = "csukuangfj/kokoro-multi-lang-v1_1";
pub const KOKORO_REPO: &str = "journey0ad/kokoro-multi-lang-v1_1";
/// 推理必需的文件；`espeak-ng-data/`（几百个小文件）和别的语言词表不下。
pub const KOKORO_FILES: &[&str] = &[
    "model.onnx",
    "voices.bin",
    "tokens.txt",
    "lexicon-us-en.txt",
    "lexicon-zh.txt",
    "dict/jieba.dict.utf8",
];

/// Laya（路由/决策模型）的 ModelScope 仓库。
pub const LAYA_MODEL_ID: &str = "convaiinnovations/laya-multilingual";
pub const LAYA_REPO: &str = "convaiinnovations/laya-multilingual";
/// 跟 HF 上 `allow_patterns` 一致：只要这四个，README / configuration.json 不下。
pub const LAYA_FILES: &[&str] = &[
    "encoder/config.json",
    "model.safetensors",
    "rl_agent_config.json",
    "tokenizer/tokenizer.json",
];

/// TurboVLA（视觉-语言-动作）在 ModelScope 上的转换产物镜像。
pub const TURBOVLA_MODEL_ID: &str = "OpenBMB/TurboVLA";
pub const TURBOVLA_REPO: &str = "Yanguan/turbovla-libero";
/// 转换产出目录里的全部文件。
pub const TURBOVLA_FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "text_config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "vision_config.json",
    "vocab.txt",
];

/// GPT-SoVITS（v2Pro 中文）在 ModelScope 上的转换产物镜像。
pub const GPT_SOVITS_MODEL_ID: &str = "gpt-sovits-v2pro-zh";
pub const GPT_SOVITS_REPO: &str = "Yanguan/gpt-sovits-v2pro-zh";
/// 推理需要的全部文件（`gpt_sovits::pipeline` 只校验 `manifest.json`，这里列全）。
pub const GPT_SOVITS_FILES: &[&str] = &[
    "bert.safetensors",
    "bert_config.json",
    "bert_tokenizer/tokenizer.json",
    "dict.txt",
    "frontend_zh.json",
    "gpt.safetensors",
    "hubert.safetensors",
    "hubert_config.json",
    "manifest.json",
    "ref.txt",
    "ref.wav",
    "sovits.safetensors",
    "sv_eres2net.safetensors",
    "symbols.json",
    "voices.json",
];

/// Qwen3.5 GGUF 的镜像：`-m qwen3.5-gguf` 默认下 0.8B Q4_K_M + mmproj-F16。
pub const QWEN3_5_GGUF_MODEL_ID: &str = "qwen3.5-gguf";
pub const QWEN3_5_GGUF_REPO: &str = "Yanguan/qwen3.5-0.8b-gguf";
pub const QWEN3_5_GGUF_FILE: &str = "Qwen3.5-0.8B-Q4_K_M.gguf";
pub const QWEN3_5_MMPROJ_FILE: &str = "mmproj-F16.gguf";
pub const QWEN3_5_GGUF_FILES: &[&str] = &[QWEN3_5_GGUF_FILE, QWEN3_5_MMPROJ_FILE];

/// Hy-MT2（混元翻译 1.8B，1.25bit STQ GGUF）：官方仓库，整个仓库就一个权重文件可下。
pub const HY_MT2_MODEL_ID: &str = "hy-mt2-1.8b-1.25bit-gguf";
pub const HY_MT2_REPO: &str = "Tencent-Hunyuan/Hy-MT2-1.8B-1.25bit-GGUF";

/// 所有带 ModelScope 落点的模型。
pub const ALL: &[Mirror] = &[
    Mirror {
        model_id: SENSE_VOICE_MODEL_ID,
        repo: SENSE_VOICE_REPO,
        files: Some(SENSE_VOICE_FILES),
    },
    Mirror {
        model_id: KOKORO_MODEL_ID,
        repo: KOKORO_REPO,
        files: Some(KOKORO_FILES),
    },
    Mirror {
        model_id: LAYA_MODEL_ID,
        repo: LAYA_REPO,
        files: Some(LAYA_FILES),
    },
    Mirror {
        model_id: TURBOVLA_MODEL_ID,
        repo: TURBOVLA_REPO,
        files: Some(TURBOVLA_FILES),
    },
    Mirror {
        model_id: GPT_SOVITS_MODEL_ID,
        repo: GPT_SOVITS_REPO,
        files: Some(GPT_SOVITS_FILES),
    },
    Mirror {
        model_id: QWEN3_5_GGUF_MODEL_ID,
        repo: QWEN3_5_GGUF_REPO,
        files: Some(QWEN3_5_GGUF_FILES),
    },
    Mirror {
        model_id: HY_MT2_MODEL_ID,
        repo: HY_MT2_REPO,
        files: None,
    },
];

/// 按模型 id 找镜像；找不到就是普通 ModelScope 仓库（`repo` 直接用 id）。
pub fn mirror(model_id: &str) -> Option<&'static Mirror> {
    ALL.iter().find(|m| m.model_id == model_id)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_mirror_has_a_repo_and_matching_id() {
        for m in ALL {
            assert!(!m.repo.is_empty(), "{} 没有仓库", m.model_id);
            assert_eq!(mirror(m.model_id).map(|x| x.repo), Some(m.repo));
        }
    }

    /// 这几个是改过仓库名的（去掉了 `aha-` 前缀），别改回去。
    #[test]
    fn repos_have_no_aha_prefix() {
        for m in ALL {
            assert!(
                !m.repo.to_lowercase().contains("aha-"),
                "{} 的仓库名还带 aha 前缀：{}",
                m.model_id,
                m.repo
            );
        }
    }

    /// GGUF 镜像的两个文件必须都在清单里，且顺序固定（模型在前、mmproj 在后）。
    #[test]
    fn gguf_files_model_first() {
        assert_eq!(
            QWEN3_5_GGUF_FILES,
            &[QWEN3_5_GGUF_FILE, QWEN3_5_MMPROJ_FILE]
        );
    }
}

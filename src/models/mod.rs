pub mod all_minilm_l6_v2;
pub mod bigvgan;
pub mod campplus;
pub mod common;
pub mod deepseek_ocr;
pub mod feature_extractor;
pub mod fun_asr_nano;
pub mod glm_asr_nano;
pub mod glm_ocr;
#[cfg(feature = "gpt-sovits")]
pub mod gpt_sovits;
pub mod hunyuan_ocr;
pub mod index_tts;
#[cfg(feature = "sherpa-tts")]
pub mod kokoro;
#[cfg(feature = "laya")]
pub mod laya;
pub mod lfm2;
pub mod lfm2vl;
pub mod mask_gct;
pub mod minicpm4;
pub mod minicpm5;
pub mod moss_audio_tokenizer_nano;
pub mod moss_tts_nano;
pub mod paddleocr_vl;
pub mod qwen2;
pub mod qwen2_5vl;
pub mod qwen3;
pub mod qwen3_5;
pub mod qwen3_asr;
pub mod qwen3_embedding;
pub mod qwen3_reranker;
pub mod qwen3vl;
pub mod rmbg2_0;
#[cfg(feature = "sherpa-asr")]
pub mod sense_voice;
pub mod voxcpm;
pub mod voxcpm_refact;
pub mod w2v_bert_2_0;
#[cfg(feature = "vla")]
pub mod turbovla;
// pub mod sam3;
pub mod fire_red_vad;
pub mod gpt2;
pub mod llama;

use crate::{
    models::{
        all_minilm_l6_v2::AllMiniLML6V2Embedding,
        common::{embedding::TextEmbedding, model_mapping::WhichModel, reranker::TextRerank},
        minicpm5::generate::MiniCPM5GenerateModel,
        qwen3_embedding::Qwen3Embedding,
        qwen3_reranker::Qwen3Reranker,
    },
    params::chat::{ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse},
};
use anyhow::{Result, anyhow};
use candle_core::{DType, Device};
use futures_util::Stream;

use crate::models::{
    deepseek_ocr::generate::DeepseekOCRGenerateModel,
    fun_asr_nano::generate::FunAsrNanoGenerateModel,
    glm_asr_nano::generate::GlmAsrNanoGenerateModel, glm_ocr::generate::GlmOcrGenerateModel,
    hunyuan_ocr::generate::HunyuanOCRGenerateModel, lfm2::generate::Lfm2GenerateModel,
    lfm2vl::generate::Lfm2VLGenerateModel, minicpm4::generate::MiniCPM4GenerateModel,
    paddleocr_vl::generate::PaddleOCRVLGenerateModel, qwen2_5vl::generate::Qwen2_5VLGenerateModel,
    qwen3::generate::Qwen3GenerateModel, qwen3_5::generate::Qwen3_5GenerateModel,
    qwen3_asr::generate::Qwen3AsrGenerateModel, qwen3vl::generate::Qwen3VLGenerateModel,
    rmbg2_0::generate::RMBG2_0Model, voxcpm::generate::VoxCPMGenerate,
};

pub trait GenerateModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse>;
    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    >;
}

pub enum ModelInstance<'a> {
    AllMiniLML6V2(AllMiniLML6V2Embedding),
    MiniCPM4(MiniCPM4GenerateModel<'a>),
    MiniCPM5(MiniCPM5GenerateModel<'a>),
    Lfm2(Lfm2GenerateModel<'a>),
    Lfm2VL(Lfm2VLGenerateModel<'a>),
    Qwen2_5VL(Qwen2_5VLGenerateModel<'a>),
    Qwen3(Qwen3GenerateModel<'a>),
    Qwen3_5(Qwen3_5GenerateModel<'a>),
    Qwen3ASR(Qwen3AsrGenerateModel<'a>),
    Qwen3Embedding(Qwen3Embedding),
    Qwen3Reranker(Qwen3Reranker),
    Qwen3VL(Box<Qwen3VLGenerateModel<'a>>),
    DeepSeekOCR(DeepseekOCRGenerateModel),
    HunyuanOCR(HunyuanOCRGenerateModel<'a>),
    PaddleOCRVL(Box<PaddleOCRVLGenerateModel<'a>>),
    RMBG2_0(Box<RMBG2_0Model>),
    VoxCPM(Box<VoxCPMGenerate>),
    GlmASRNano(GlmAsrNanoGenerateModel<'a>),
    FunASRNano(FunAsrNanoGenerateModel),
    GlmOCR(GlmOcrGenerateModel),
    #[cfg(feature = "sherpa-asr")]
    SenseVoice(crate::models::sense_voice::SenseVoiceModel),
    #[cfg(feature = "sherpa-tts")]
    Kokoro(crate::models::kokoro::KokoroTtsModel),
    #[cfg(feature = "laya")]
    Laya(crate::models::laya::LayaRouteModel),
    #[cfg(feature = "vla")]
    Vla(crate::models::turbovla::TurboVlaModel),
    #[cfg(feature = "gpt-sovits")]
    GptSovits(crate::models::gpt_sovits::GptSovitsModel),
}

impl<'a> GenerateModel for ModelInstance<'a> {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        match self {
            ModelInstance::AllMiniLML6V2(_) => {
                Err(anyhow!("embedding model does not support chat completions"))
            }
            ModelInstance::MiniCPM4(model) => model.generate(mes),
            ModelInstance::MiniCPM5(model) => model.generate(mes),
            ModelInstance::Lfm2(model) => model.generate(mes),
            ModelInstance::Lfm2VL(model) => model.generate(mes),
            ModelInstance::Qwen2_5VL(model) => model.generate(mes),
            ModelInstance::Qwen3(model) => model.generate(mes),

            ModelInstance::Qwen3Embedding(_) => {
                Err(anyhow!("embedding model does not support chat completions"))
            }
            ModelInstance::Qwen3Reranker(_) => {
                Err(anyhow!("reranker model does not support chat completions"))
            }
            ModelInstance::Qwen3_5(model) => model.generate(mes),

            ModelInstance::Qwen3ASR(model) => model.generate(mes),
            ModelInstance::Qwen3VL(model) => model.generate(mes),
            ModelInstance::DeepSeekOCR(model) => model.generate(mes),
            ModelInstance::HunyuanOCR(model) => model.generate(mes),
            ModelInstance::PaddleOCRVL(model) => model.generate(mes),
            ModelInstance::RMBG2_0(model) => model.generate(mes),
            ModelInstance::VoxCPM(model) => model.generate(mes),
            ModelInstance::GlmASRNano(model) => model.generate(mes),
            ModelInstance::FunASRNano(model) => model.generate(mes),
            ModelInstance::GlmOCR(model) => model.generate(mes),
            #[cfg(feature = "sherpa-asr")]
            ModelInstance::SenseVoice(model) => model.generate(mes),
            #[cfg(feature = "sherpa-tts")]
            ModelInstance::Kokoro(model) => model.generate(mes),
            #[cfg(feature = "laya")]
            ModelInstance::Laya(_) => Err(anyhow!(
                "laya is a routing model; use POST /v1/decisions instead of chat completions"
            )),
            #[cfg(feature = "vla")]
            ModelInstance::Vla(_) => Err(anyhow!(
                "vla is an action model; use POST /v1/actions instead of chat completions"
            )),
            #[cfg(feature = "gpt-sovits")]
            ModelInstance::GptSovits(_) => Err(anyhow!(
                "gpt-sovits is a tts model; use POST /v1/tts instead of chat completions"
            )),
        }
    }

    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<
        Box<
            dyn Stream<Item = Result<ChatCompletionChunkResponse, anyhow::Error>>
                + Send
                + Unpin
                + '_,
        >,
    > {
        match self {
            ModelInstance::AllMiniLML6V2(_) => {
                Err(anyhow!("embedding model does not support chat completions"))
            }
            ModelInstance::MiniCPM4(model) => model.generate_stream(mes),
            ModelInstance::MiniCPM5(model) => model.generate_stream(mes),
            ModelInstance::Lfm2(model) => model.generate_stream(mes),
            ModelInstance::Lfm2VL(model) => model.generate_stream(mes),
            ModelInstance::Qwen2_5VL(model) => model.generate_stream(mes),
            ModelInstance::Qwen3(model) => model.generate_stream(mes),

            ModelInstance::Qwen3Embedding(_) => Err(anyhow!(
                "embedding model does not support streaming chat completions"
            )),
            ModelInstance::Qwen3Reranker(_) => Err(anyhow!(
                "reranker model does not support streaming chat completions"
            )),
            ModelInstance::Qwen3_5(model) => model.generate_stream(mes),
            ModelInstance::Qwen3ASR(model) => model.generate_stream(mes),
            ModelInstance::Qwen3VL(model) => model.generate_stream(mes),
            ModelInstance::DeepSeekOCR(model) => model.generate_stream(mes),
            ModelInstance::HunyuanOCR(model) => model.generate_stream(mes),
            ModelInstance::PaddleOCRVL(model) => model.generate_stream(mes),
            ModelInstance::RMBG2_0(model) => model.generate_stream(mes),
            ModelInstance::VoxCPM(model) => model.generate_stream(mes),
            ModelInstance::GlmASRNano(model) => model.generate_stream(mes),
            ModelInstance::FunASRNano(model) => model.generate_stream(mes),
            ModelInstance::GlmOCR(model) => model.generate_stream(mes),
            #[cfg(feature = "sherpa-asr")]
            ModelInstance::SenseVoice(model) => model.generate_stream(mes),
            #[cfg(feature = "sherpa-tts")]
            ModelInstance::Kokoro(model) => model.generate_stream(mes),
            #[cfg(feature = "laya")]
            ModelInstance::Laya(_) => Err(anyhow!(
                "laya is a routing model; use POST /v1/decisions instead of chat completions"
            )),
            #[cfg(feature = "vla")]
            ModelInstance::Vla(_) => Err(anyhow!(
                "vla is an action model; use POST /v1/actions instead of chat completions"
            )),
            #[cfg(feature = "gpt-sovits")]
            ModelInstance::GptSovits(_) => Err(anyhow!(
                "gpt-sovits is a tts model; use POST /v1/tts instead of streaming chat completions"
            )),
        }
    }
}

impl<'a> ModelInstance<'a> {
    /// 取 Laya 路由模型（只有加载了 laya 时才有）。`/v1/decisions` 端点用它。
    #[cfg(feature = "laya")]
    pub fn route_model(&self) -> Option<&crate::models::laya::LayaRouteModel> {
        match self {
            ModelInstance::Laya(model) => Some(model),
            _ => None,
        }
    }

    /// 取 TurboVLA 动作模型（只有加载了 vla 时才有）。`/v1/actions` 端点用它。
    #[cfg(feature = "vla")]
    pub fn vla_model(&self) -> Option<&crate::models::turbovla::TurboVlaModel> {
        match self {
            ModelInstance::Vla(model) => Some(model),
            _ => None,
        }
    }

    /// 取 SenseVoice（只有编了 sherpa-asr 时才有）。`/v1/asr` 端点用它。
    #[cfg(feature = "sherpa-asr")]
    pub fn asr_model(&self) -> Option<&crate::models::sense_voice::SenseVoiceModel> {
        match self {
            ModelInstance::SenseVoice(model) => Some(model),
            _ => None,
        }
    }

    /// 取 GPT-SoVITS（只有加载了 gpt-sovits 时才有）。`/v1/tts` 端点用它。
    #[cfg(feature = "gpt-sovits")]
    pub fn gpt_sovits_model(&self) -> Option<&crate::models::gpt_sovits::GptSovitsModel> {
        match self {
            ModelInstance::GptSovits(model) => Some(model),
            _ => None,
        }
    }

    /// 取 Kokoro（只有编了 sherpa-tts 时才有）。`/v1/tts` 端点用它。
    #[cfg(feature = "sherpa-tts")]
    pub fn tts_model(&self) -> Option<&crate::models::kokoro::KokoroTtsModel> {
        match self {
            ModelInstance::Kokoro(model) => Some(model),
            _ => None,
        }
    }
}

impl<'a> ModelInstance<'a> {
    pub fn embedding(&mut self, input: &[String]) -> Result<Vec<Vec<f32>>> {
        match self {
            ModelInstance::Qwen3Embedding(model) => model.embed_texts(input),
            ModelInstance::AllMiniLML6V2(model) => model.embed_texts(input),
            _ => Err(anyhow!("current model does not support embeddings")),
        }
    }
    pub fn rerank(&mut self, query: &str, documents: &[String]) -> Result<Vec<f32>> {
        match self {
            ModelInstance::Qwen3Reranker(model) => model.rerank(query, documents),
            _ => Err(anyhow!("current model does not support rerank")),
        }
    }
}

#[allow(unused)]
pub fn load_gguf_model<'a>(
    model_type: WhichModel,
    config_path: Option<&str>, // 有些gguf未包含模型其他配置，需额外指定
    gguf_path: &str,
    mmproj_path: Option<&str>,
    device: Option<&Device>,
) -> Result<ModelInstance<'a>> {
    let model = match model_type {
        WhichModel::Qwen3_5Gguf => {
            let model = Qwen3_5GenerateModel::init_from_gguf(gguf_path, mmproj_path, device)?;
            ModelInstance::Qwen3_5(model)
        }
        _ => {
            let model_id = model_type.as_string();
            return Err(anyhow!("model id {model_id} is not gguf model"));
        }
    };
    Ok(model)
}

/// sherpa-onnx 后端不看 `device` / `dtype`：它用 ONNX Runtime 自己的 provider，
/// 不接 candle 的 `Device`。所以 `--device cuda` 对 SenseVoice / Kokoro 是**静默失效**的。
///
/// 这里至少喊一声。不报错：默认就是 CPU，大多数调用方传的也是 CPU；
/// 但只要传了非 CPU 却没生效，那就得让用户知道。
#[cfg(any(feature = "sherpa-asr", feature = "sherpa-tts"))]
fn warn_sherpa_ignores_device(model: WhichModel, device: Option<&Device>, dtype: Option<DType>) {
    let non_cpu = matches!(device, Some(device) if !matches!(device, Device::Cpu));
    if non_cpu || dtype.is_some() {
        tracing::warn!(
            model = %model.as_string(),
            device = ?device,
            dtype = ?dtype,
            "sherpa-onnx 模型不走 candle 的 Device/精度：--device / --dtype 对它们不生效，\
             实际执行由 ONNX Runtime 自己选 provider"
        );
    }
}

pub fn load_model<'a>(
    model_type: WhichModel,
    path: &str,
    device: Option<&Device>,
    dtype: Option<DType>,
    threads: i32,
) -> Result<ModelInstance<'a>> {
    let model = match model_type {
        WhichModel::AllMiniLML6V2 => {
            let model = AllMiniLML6V2Embedding::init(path, device, dtype)?;
            ModelInstance::AllMiniLML6V2(model)
        }
        WhichModel::MiniCPM4_0_5B => {
            let model = MiniCPM4GenerateModel::init(path, device, dtype)?;
            ModelInstance::MiniCPM4(model)
        }
        WhichModel::MiniCPM5_1B => {
            let model = MiniCPM5GenerateModel::init(path, device, dtype)?;
            ModelInstance::MiniCPM5(model)
        }
        WhichModel::LFM2_1_2B | WhichModel::LFM2_5_1_2BInstruct => {
            let model = Lfm2GenerateModel::init(path, device, dtype)?;
            ModelInstance::Lfm2(model)
        }
        WhichModel::LFM2_5VL1_6B | WhichModel::LFM2VL1_6B | WhichModel::LFM2_5VL450M => {
            let model = Lfm2VLGenerateModel::init(path, device, dtype)?;
            ModelInstance::Lfm2VL(model)
        }
        WhichModel::Qwen2_5VL3B | WhichModel::Qwen2_5VL7B => {
            let model = Qwen2_5VLGenerateModel::init(path, device, dtype)?;
            ModelInstance::Qwen2_5VL(model)
        }
        WhichModel::Qwen3_0_6B | WhichModel::Qwen3_1_7B | WhichModel::Qwen3_4B => {
            let model = Qwen3GenerateModel::init(path, device, dtype)?;
            ModelInstance::Qwen3(model)
        }
        WhichModel::Qwen3_5_0_8B
        | WhichModel::Qwen3_5_2B
        | WhichModel::Qwen3_5_4B
        | WhichModel::Qwen3_5_9B => {
            let model = Qwen3_5GenerateModel::init(path, device, dtype)?;
            ModelInstance::Qwen3_5(model)
        }
        WhichModel::Qwen3ASR0_6B | WhichModel::Qwen3ASR1_7B => {
            let model = Qwen3AsrGenerateModel::init(path, device, dtype)?;
            ModelInstance::Qwen3ASR(model)
        }
        WhichModel::Qwen3Embedding0_6B
        | WhichModel::Qwen3Embedding4B
        | WhichModel::Qwen3Embedding8B => {
            let model = Qwen3Embedding::init(path, device, dtype)?;
            ModelInstance::Qwen3Embedding(model)
        }
        WhichModel::Qwen3Reranker0_6B
        | WhichModel::Qwen3Reranker4B
        | WhichModel::Qwen3Reranker8B => {
            let model = Qwen3Reranker::init(path, device, dtype)?;
            ModelInstance::Qwen3Reranker(model)
        }
        WhichModel::Qwen3VL2B
        | WhichModel::Qwen3VL4B
        | WhichModel::Qwen3VL8B
        | WhichModel::Qwen3VL32B => {
            let model = Qwen3VLGenerateModel::init(path, device, dtype)?;
            ModelInstance::Qwen3VL(Box::new(model))
        }
        WhichModel::DeepSeekOCR | WhichModel::DeepSeekOCR2 => {
            let model = DeepseekOCRGenerateModel::init(path, device, dtype)?;
            ModelInstance::DeepSeekOCR(model)
        }
        WhichModel::HunyuanOCR => {
            let model = HunyuanOCRGenerateModel::init(path, device, dtype)?;
            ModelInstance::HunyuanOCR(model)
        }
        WhichModel::PaddleOCRVL | WhichModel::PaddleOCRVL1_5 | WhichModel::PaddleOCRVL1_6 => {
            let model = PaddleOCRVLGenerateModel::init(path, device, dtype)?;
            ModelInstance::PaddleOCRVL(Box::new(model))
        }
        WhichModel::RMBG2_0 => {
            let model = RMBG2_0Model::init(path, device, dtype)?;
            ModelInstance::RMBG2_0(Box::new(model))
        }
        WhichModel::VoxCPM | WhichModel::VoxCPM1_5 | WhichModel::VoxCPM2 => {
            let model = VoxCPMGenerate::init(path, device, dtype)?;
            ModelInstance::VoxCPM(Box::new(model))
        }
        WhichModel::GlmASRNano2512 => {
            let model = GlmAsrNanoGenerateModel::init(path, device, dtype)?;
            ModelInstance::GlmASRNano(model)
        }
        WhichModel::FunASRNano2512 => {
            let model = FunAsrNanoGenerateModel::init(path, device, dtype)?;
            ModelInstance::FunASRNano(model)
        }
        WhichModel::GlmOCR => {
            let model = GlmOcrGenerateModel::init(path, device, dtype)?;
            ModelInstance::GlmOCR(model)
        }
        #[cfg(feature = "sherpa-asr")]
        WhichModel::SenseVoice => {
            warn_sherpa_ignores_device(model_type, device, dtype);
            let model = crate::models::sense_voice::SenseVoiceModel::init_from_dir(
                path,
                if threads > 0 { threads } else { 4 },
            )?;
            ModelInstance::SenseVoice(model)
        }
        #[cfg(feature = "sherpa-tts")]
        WhichModel::Kokoro => {
            warn_sherpa_ignores_device(model_type, device, dtype);
            let model = crate::models::kokoro::KokoroTtsModel::init(
                path,
                if threads > 0 { threads } else { 4 },
                crate::models::kokoro::DEFAULT_SID,
            )?;
            ModelInstance::Kokoro(model)
        }
        #[cfg(feature = "laya")]
        WhichModel::Laya => {
            let model = crate::models::laya::LayaRouteModel::load(
                std::path::Path::new(path),
                crate::models::laya::route::MIN_CONFIDENCE,
                15_000,
                0,
            )?;
            ModelInstance::Laya(model)
        }
        #[cfg(feature = "gpt-sovits")]
        WhichModel::GptSovits => {
            let model = crate::models::gpt_sovits::GptSovitsModel::init(path, device, dtype)?;
            ModelInstance::GptSovits(model)
        }
        #[cfg(feature = "vla")]
        WhichModel::TurboVla => {
            let dev = device.cloned().unwrap_or(Device::Cpu);
            let model = crate::models::turbovla::TurboVlaModel::load(
                path,
                &dev,
                dtype.unwrap_or(DType::F32),
            )?;
            ModelInstance::Vla(model)
        }
        _ => {
            let model_id = model_type.as_string();
            if model_id.to_lowercase().contains("gguf") || model_id.to_lowercase().contains("onnx")
            {
                return Err(anyhow!("model id {model_id} is not safetensor model"));
            } else {
                return Err(anyhow!("model id {model_id} not impl load_model function"));
            }
        }
    };
    Ok(model)
}

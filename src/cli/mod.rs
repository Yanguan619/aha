use crate::{
    cli::args::{
        CliArgs, DeleteArgs, DownloadArgs, ListArgs, MultiArgs, ProbeArgs, RunArgs, ServArgs,
        ServListArgs,
    },
    server::{
        api::{InitOptions, ModelSpec, init, init_multi},
        process::{ServiceStatus, find_aha_services},
        start_http_server,
    },
};
use aha::exec::*;
use aha::{
    models::common::model_mapping::WhichModel,
    utils::{
        bytes_to_human, dir_size, download_model, get_default_save_dir, get_default_weight_path,
        is_model_downloaded,
    },
};
use anyhow::anyhow;
use serde::Serialize;

pub mod args;

/// Model information for JSON output
#[derive(Serialize)]
struct ModelInfo {
    model_id: String,
    owner: String,
    #[serde(rename = "type")]
    model_type: String,
    downloaded: bool,
}

/// List all supported models
pub(crate) fn run_list(args: ListArgs) -> anyhow::Result<()> {
    let models = WhichModel::model_list();

    if args.json {
        // JSON output
        let model_infos: Vec<ModelInfo> = models
            .iter()
            .map(|model| ModelInfo {
                model_id: model.as_string(),
                owner: model.model_owner(),
                model_type: model.model_type().to_string(),
                downloaded: is_model_downloaded(*model),
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&model_infos)?);
    } else {
        // Table output (default)
        println!("Available models:");
        println!();
        println!(
            "{:<40} {:<20} {:<10} {:<10}",
            "Model ID", "Owner", "type", "Download"
        );
        println!("{}", "-".repeat(80));
        for model in models {
            let model_id = model.as_string();
            let owner = model.model_owner();
            let model_type = model.model_type();
            let download_status = if is_model_downloaded(model) {
                "  ✔"
            } else {
                ""
            };
            println!(
                "{:<40} {:<20} {:<10} {:<10}",
                model_id, owner, model_type, download_status
            );
        }
    }

    Ok(())
}

/// Run the 'cli' subcommand: download model (if needed) and start service
pub(crate) async fn run_cli(args: CliArgs) -> anyhow::Result<()> {
    let CliArgs {
        model,
        server_common,
        pet,
        save_dir,
        download_retries,
        path_common,
    } = args;
    let model_id = model.as_string();

    let (model_path, gguf, mmproj) = if model.is_gguf() {
        if path_common.gguf_path.is_none() {
            return Err(anyhow!("gguf model path is required"));
        }
        (
            "GGUF".to_string(),
            path_common.gguf_path,
            path_common.mmproj_path,
        )
    } else if model.is_onnx() && !model.is_pet_delegated_model() {
        // 桌宠三个模型走 sherpa/sys1，不进 onnx 门禁（SenseVoice 的 id 里就带 "onnx"）
        return Err(anyhow!("onnx model not support now"));
    } else {
        let model_path = match path_common.weight_path {
            Some(path) => path,
            None => {
                #[cfg(feature = "sherpa")]
                if model.is_sherpa_model() {
                    return Err(anyhow!(
                        "{model_id} is served by sherpa-onnx and is not hosted on ModelScope.\n\
                         Download the model (ONNX + tokens/voices) and pass --weight-path <dir>."
                    ));
                }
                #[cfg(feature = "laya")]
                if model.is_laya_model() {
                    return Err(anyhow!(
                        "{model_id} is served by sys1 and is not hosted on ModelScope.\n\
                         Download the model and pass --weight-path <dir>."
                    ));
                }
                #[cfg(feature = "gpt-sovits")]
                if model.is_gpt_sovits_model() {
                    return Err(anyhow!(
                        "{model_id} 是本地转换的 GPT-SoVITS 权重，不在 ModelScope 上。\n\
                         先用 scripts/gpt_sovits/convert_weights.py 转换，再传 --weight-path <dir>。"
                    ));
                }
                let save_dir = match save_dir {
                    Some(dir) => dir,
                    None => get_default_save_dir().expect("Failed to get home directory"),
                };
                let max_retries = download_retries.unwrap_or(3);
                download_model(&model_id, &save_dir, max_retries).await?;
                save_dir + "/" + &model_id
            }
        };
        (model_path, None, None)
    };

    init(
        model,
        model_path,
        gguf,
        mmproj,
        &InitOptions {
            device: pet.device,
            vla_device: None,
            min_confidence: pet.min_confidence,
            request_timeout_ms: pet.request_timeout_ms,
            threads: pet.threads,
            no_warmup: pet.no_warmup,
        },
    )?;
    // 预热：把首个请求的一次性开销挪到端口开之前。
    crate::server::api::warmup_all(!pet.no_warmup).await;
    start_http_server(
        server_common.address,
        server_common.port,
        server_common.allow_remote_shutdown,
        server_common.parent_pid,
    )
    .await?;

    Ok(())
}

/// Run the 'serv' subcommand: start service only (no download)
pub(crate) async fn run_serv(args: ServArgs) -> anyhow::Result<()> {
    let ServArgs {
        model,
        server_common,
        pet,
        path_common,
    } = args;

    let mut model_path;
    let weight_path = path_common.weight_path.clone();
    let (gguf, mmproj) = if model.is_gguf() {
        if path_common.gguf_path.is_none() {
            return Err(anyhow!("gguf model path is required"));
        }
        model_path = "GGUF".to_string();
        (path_common.gguf_path, path_common.mmproj_path)
    } else if model.is_onnx() && !model.is_pet_delegated_model() {
        // 桌宠三个模型走 sherpa/sys1，不进 onnx 门禁（SenseVoice 的 id 里就带 "onnx"）
        return Err(anyhow!("onnx model not support now"));
    } else {
        match weight_path.clone() {
            Some(path) => model_path = path,
            None => model_path = get_default_weight_path(model),
        }
        (None, None)
    };

    // 桌宠委托模式：SenseVoice / Kokoro 首次启动自己按文件列表下载（必填文件齐了就跳过）；
    // Laya 走 sys1 的 hub（--weight-path 优先，其次 HF 缓存，最后联网下）。
    #[cfg(feature = "sherpa")]
    if model.is_sherpa_model() {
        tracing::info!(dir = %model_path, model = %model.as_string(), "检查 sherpa 模型文件");
        aha::models::common::sherpa::ensure_downloaded(&model_path, model)?;
    }
    #[cfg(feature = "laya")]
    if model.is_laya_model() {
        let (path, source) =
            aha::models::laya::resolve(weight_path.as_deref(), &model.as_string()).await?;
        tracing::info!(dir = %path.display(), source, "Laya 模型目录就绪");
        model_path = path.display().to_string();
    }

    if !std::path::Path::new(&model_path).exists() {
        return Err(anyhow!(
            "serv subcommand will not download model, use `weight-path` to pass the model path"
        ));
    }

    init(
        model,
        model_path,
        gguf,
        mmproj,
        &InitOptions {
            device: pet.device,
            vla_device: None,
            min_confidence: pet.min_confidence,
            request_timeout_ms: pet.request_timeout_ms,
            threads: pet.threads,
            no_warmup: pet.no_warmup,
        },
    )?;
    // 预热：把首个请求的一次性开销挪到端口开之前。
    crate::server::api::warmup_all(!pet.no_warmup).await;
    start_http_server(
        server_common.address,
        server_common.port,
        server_common.allow_remote_shutdown,
        server_common.parent_pid,
    )
    .await?;

    Ok(())
}

/// Run the 'multi' subcommand: 一个进程服务多个模型（桌宠 asr/tts/laya）。
pub(crate) async fn run_multi(args: MultiArgs) -> anyhow::Result<()> {
    let MultiArgs {
        asr,
        tts,
        tts_model,
        laya,
        laya_path,
        vla,
        vla_device,
        server_common,
        pet,
    } = args;

    let mut specs: Vec<ModelSpec> = Vec::new();
    // tts 角色用哪个模型：默认 Kokoro（sherpa）；传 gpt-sovits-v2pro-zh 时换成
    // 本地转换的 GPT-SoVITS（--tts 指向转换后的目录，不走 sherpa、不下载）。
    #[cfg(feature = "sherpa")]
    let tts_model = tts_model.unwrap_or(WhichModel::Kokoro);
    #[cfg(not(feature = "sherpa"))]
    let tts_model =
        tts_model.ok_or_else(|| anyhow!("这个二进制没编 sherpa 特性，--tts 必须配 --tts-model"))?;

    #[cfg(feature = "sherpa")]
    {
        if let Some(dir) = asr.as_deref() {
            tracing::info!(dir = %dir, role = "asr", "检查 sherpa 模型文件");
            aha::models::common::sherpa::ensure_downloaded(dir, WhichModel::SenseVoice)?;
            specs.push(ModelSpec::new("asr", WhichModel::SenseVoice, dir));
        }
        if let Some(dir) = tts.as_deref()
            && tts_model.is_sherpa_model()
        {
            tracing::info!(dir = %dir, role = "tts", "检查 sherpa 模型文件");
            aha::models::common::sherpa::ensure_downloaded(dir, tts_model)?;
        }
        if let Some(dir) = tts.as_deref() {
            specs.push(ModelSpec::new("tts", tts_model, dir));
        }
    }
    #[cfg(not(feature = "sherpa"))]
    if let Some(dir) = tts.as_deref() {
        specs.push(ModelSpec::new("tts", tts_model, dir));
    }
    #[cfg(not(feature = "sherpa"))]
    if asr.is_some() {
        return Err(anyhow!("这个二进制没编 sherpa 特性，不能启用 asr 角色"));
    }

    #[cfg(feature = "laya")]
    if laya || laya_path.is_some() {
        let (path, source) =
            aha::models::laya::resolve(laya_path.as_deref(), &WhichModel::Laya.as_string()).await?;
        tracing::info!(dir = %path.display(), source, "Laya 模型目录就绪");
        specs.push(ModelSpec::new(
            "system1",
            WhichModel::Laya,
            path.display().to_string(),
        ));
    }
    #[cfg(not(feature = "laya"))]
    if laya || laya_path.is_some() {
        return Err(anyhow!("这个二进制没编 laya 特性，不能启用 laya 角色"));
    }

    #[cfg(feature = "vla")]
    if let Some(dir) = vla.as_deref() {
        specs.push(ModelSpec::new("vla", WhichModel::TurboVla, dir));
    }
    #[cfg(not(feature = "vla"))]
    if vla.is_some() {
        return Err(anyhow!("这个二进制没编 vla 特性，不能启用 vla 角色"));
    }

    if specs.is_empty() {
        return Err(anyhow!(
            "multi 至少要启用一个角色（--asr <dir> / --tts <dir> / --laya）"
        ));
    }

    init_multi(
        specs,
        &InitOptions {
            device: pet.device,
            vla_device,
            min_confidence: pet.min_confidence,
            request_timeout_ms: pet.request_timeout_ms,
            threads: pet.threads,
            no_warmup: pet.no_warmup,
        },
    )?;
    // 预热：把首个请求的一次性开销挪到端口开之前。
    crate::server::api::warmup_all(!pet.no_warmup).await;
    // 三条角色的路径不同（/v1/asr、/v1/tts、/v1/decisions），一个端口就够。
    start_http_server(
        server_common.address,
        server_common.port,
        server_common.allow_remote_shutdown,
        server_common.parent_pid,
    )
    .await?;
    Ok(())
}

/// Run the 'probe' subcommand: check the binary's compiled features against the
/// requested device, without loading any model. desktop-pet runs this before
/// starting the Laya bridge so a device mismatch shows up under the toggle.
///
/// `--selftest` additionally loads the model and runs the routing corpus.
pub(crate) async fn run_probe(args: ProbeArgs) -> anyhow::Result<()> {
    let ProbeArgs {
        model,
        device,
        selftest,
    } = args;
    #[cfg(feature = "laya")]
    if model.is_laya_model() {
        aha::models::laya::check_device(&device)?;
        if selftest {
            let (path, source) = aha::models::laya::resolve(None, &model.as_string()).await?;
            tracing::info!(dir = %path.display(), source, "Laya 模型目录就绪");
            let model = aha::models::laya::LayaRouteModel::load(
                &path,
                aha::models::laya::route::MIN_CONFIDENCE,
                15_000,
                0,
            )?;
            aha::models::laya::selftest(&model).await?;
            return Ok(());
        }
        println!("{}", aha::models::laya::compile_device());
        return Ok(());
    }
    let _ = selftest;
    // 非 laya：probe 只确认二进制能跑（桌面端真正拉起前用 --version 探一次）。
    println!("ok");
    Ok(())
}

/// Run the 'ps' subcommand: list running AHA services
pub(crate) fn run_ps(args: ServListArgs) -> anyhow::Result<()> {
    let services = find_aha_services()?;

    if services.is_empty() {
        println!("No aha services found running.");
        return Ok(());
    }

    if args.compact {
        // Compact format: one service per line
        for svc in services {
            println!("{}", svc.service_id);
        }
    } else {
        // Table format
        println!(
            "{:<20} {:<10} {:<20} {:<10} {:<15} {:<10}",
            "Service ID", "PID", "Model", "Port", "Address", "Status"
        );
        println!("{}", "-".repeat(85));

        for svc in services {
            let model = svc.model.as_deref().unwrap_or("N/A");
            let status = match svc.status {
                ServiceStatus::Running => "Running",
                ServiceStatus::Stopping => "Stopping",
                ServiceStatus::Unknown => "Unknown",
            };
            println!(
                "{:<20} {:<10} {:<20} {:<10} {:<15} {:<10}",
                svc.service_id, svc.pid, model, svc.port, svc.address, status,
            );
        }
    }

    Ok(())
}

/// Run the 'download' subcommand: download model only (no server)
pub(crate) async fn run_download(args: DownloadArgs) -> anyhow::Result<()> {
    let DownloadArgs {
        model,
        save_dir,
        download_retries,
    } = args;
    let model_id = model.as_string();

    let save_dir = match save_dir {
        Some(dir) => dir,
        None => get_default_save_dir().expect("Failed to get home directory"),
    };
    let max_retries = download_retries.unwrap_or(3);

    download_model(&model_id, &save_dir, max_retries).await?;

    Ok(())
}

/// Run the 'run' subcommand: direct model inference
pub(crate) fn run_run(args: RunArgs) -> anyhow::Result<()> {
    let RunArgs {
        model,
        input,
        output,
        pet,
        path_common,
    } = args;

    // Use default weight path if not specified
    let weight_path = match path_common.weight_path {
        Some(path) => path,
        None => get_default_weight_path(model),
    };
    match model {
        WhichModel::AllMiniLML6V2 => {
            all_minilm_l6_v2::AllMiniLML6V2Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::MiniCPM4_0_5B => {
            minicpm4::MiniCPM4Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::MiniCPM5_1B => {
            minicpm5::MiniCPM5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::LFM2_1_2B | WhichModel::LFM2_5_1_2BInstruct => {
            lfm2::Lfm2Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::LFM2_5VL1_6B | WhichModel::LFM2VL1_6B | WhichModel::LFM2_5VL450M => {
            lfm2vl::Lfm2VLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen2_5VL3B | WhichModel::Qwen2_5VL7B => {
            qwen2_5vl::Qwen2_5VLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_0_6B | WhichModel::Qwen3_1_7B | WhichModel::Qwen3_4B => {
            qwen3::Qwen3Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5_0_8B
        | WhichModel::Qwen3_5_2B
        | WhichModel::Qwen3_5_4B
        | WhichModel::Qwen3_5_9B => {
            qwen3_5::Qwen3_5Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3_5Gguf => {
            qwen3_5::Qwen3_5Exec::run_gguf(
                &input,
                output.as_deref(),
                path_common.gguf_path,
                path_common.mmproj_path,
            )?;
        }
        WhichModel::Qwen3ASR0_6B | WhichModel::Qwen3ASR1_7B => {
            qwen3_asr::Qwen3ASRExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3Embedding0_6B
        | WhichModel::Qwen3Embedding4B
        | WhichModel::Qwen3Embedding8B => {
            qwen3_embedding::Qwen3EmbeddingExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3Reranker0_6B
        | WhichModel::Qwen3Reranker4B
        | WhichModel::Qwen3Reranker8B => {
            qwen3_reranker::Qwen3RerankerExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::Qwen3VL2B
        | WhichModel::Qwen3VL4B
        | WhichModel::Qwen3VL8B
        | WhichModel::Qwen3VL32B => {
            qwen3vl::Qwen3VLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::DeepSeekOCR | WhichModel::DeepSeekOCR2 => {
            deepseek_ocr::DeepSeekORExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::HunyuanOCR => {
            hunyuan_ocr::HunyuanORExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::PaddleOCRVL | WhichModel::PaddleOCRVL1_5 | WhichModel::PaddleOCRVL1_6 => {
            paddleocr_vl::PaddleOVLExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::RMBG2_0 => {
            rmbg2_0::RMBG2_0Exec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::VoxCPM | WhichModel::VoxCPM1_5 | WhichModel::VoxCPM2 => {
            voxcpm::VoxCPMExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::GlmASRNano2512 => {
            glm_asr_nano::GlmASRNanoExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::FunASRNano2512 => {
            fun_asr_nano::FunASRNanoExec::run(&input, output.as_deref(), &weight_path)?;
        }
        WhichModel::GlmOCR => {
            glm_ocr::GlmOcrExec::run(&input, output.as_deref(), &weight_path)?;
        }
        #[cfg(feature = "sherpa")]
        WhichModel::SenseVoice => {
            sense_voice::SenseVoiceExec::run(&input, output.as_deref(), &weight_path)?;
        }
        #[cfg(feature = "sherpa")]
        WhichModel::Kokoro => {
            kokoro::KokoroExec::run(&input, output.as_deref(), &weight_path)?;
        }
        #[cfg(feature = "gpt-sovits")]
        WhichModel::GptSovits => {
            gpt_sovits::GptSovitsExec::run_with_device(
                &input,
                output.as_deref(),
                &weight_path,
                &pet.device,
            )?;
        }
        #[cfg(feature = "laya")]
        WhichModel::Laya => {
            return Err(anyhow!(
                "laya 没有 `run` 子命令；启动服务请用 `aha serv --model convaiinnovations/laya-multilingual --weight-path <dir>`"
            ));
        }
        #[cfg(feature = "vla")]
        WhichModel::TurboVla => {
            return Err(anyhow!(
                "TurboVLA 没有 `run` 子命令；启动服务请用 `aha multi --vla <dir>` 或 `aha serv --model OpenBMB/TurboVLA --weight-path <dir>`"
            ));
        }
    }

    Ok(())
}

/// Run the 'delete' subcommand: delete model from default location
pub(crate) fn run_delete(args: DeleteArgs) -> anyhow::Result<()> {
    let DeleteArgs { model } = args;
    let model_id = model.as_string();
    let save_dir = get_default_save_dir().expect("Failed to get home directory");
    let model_path = format!("{}/{}", save_dir, model_id);

    let path = std::path::Path::new(&model_path);

    if !path.exists() {
        println!("Model not found: {} does not exist", model_path);
        return Ok(());
    }

    // Show model info
    println!("Model ID: {}", model_id);
    println!("Location: {}", model_path);

    // Calculate size if possible
    if let Ok(metadata) = std::fs::metadata(path)
        && metadata.is_dir()
        && let Ok(total_size) = dir_size(path)
    {
        println!("Size: {}", bytes_to_human(total_size));
    }

    // Confirm deletion
    print!("Are you sure you want to delete this model? (y/N): ");
    use std::io::Write;
    std::io::stdout().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;

    let input = input.trim().to_lowercase();
    if input != "y" && input != "yes" {
        println!("Deletion cancelled.");
        return Ok(());
    }

    // Delete the directory
    std::fs::remove_dir_all(path)?;

    println!("Model deleted successfully: {}", model_path);

    Ok(())
}

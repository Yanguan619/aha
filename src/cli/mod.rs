use crate::{
    cli::args::{
        CliArgs, DeleteArgs, DownloadArgs, ListArgs, MultiArgs, NpuArgs, ProbeArgs, QnnArgs,
        RunArgs, ServArgs, ServListArgs,
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

/// 把 `--provider` / `--device npu` 落实到进程级 provider 上。
///
/// 三件事：
/// 1. 校验 `--provider`（clap 只收字符串，非法值要到这儿才报）；
/// 2. `--device npu` 没配 `--provider` 时，按"要 NPU"处理（探测到 QNN 就用 QNN）；
/// 3. 打一行日志说清楚**到底**用的是什么——sherpa 输出的 fallback 日志很容易漏看。
///
/// 只影响 ONNX Runtime 的模型（SenseVoice / Kokoro）；candle 的模型
/// 由 `load_model` 那边打"provider 不生效"的警告。
pub(crate) fn init_inference_backend(device: &str, provider: Option<&str>) -> anyhow::Result<()> {
    use aha::utils::npu;

    let explicit = match provider {
        Some(spec) => npu::parse_provider(spec)?,
        // `--device npu` 是 provider 的快捷方式；显式 `--provider` 优先于它。
        None if npu::device_implies_npu(device) => Some("auto-npu".to_string()),
        None => None,
    };
    if let Some(provider) = explicit.as_deref() {
        npu::set_active_provider(Some(provider));
    }
    let wanted_npu = explicit.as_deref().is_some_and(npu::is_npu_provider);

    let Some(provider) = npu::effective_provider() else {
        if wanted_npu {
            // 只可能是 `--device npu` / `--provider npu`：显式要 NPU 却没探到可用
            // 的（QNN 运行时不在，或链接的 ORT 没编 QNN EP），别静默降级。
            let report = npu::probe();
            let reason = report
                .notes
                .first()
                .map(String::as_str)
                .unwrap_or("没找到 QNN / NNAPI 运行时（跑 `aha npu` 看缺什么）");
            // 默认不装 tracing subscriber，所以用 eprintln：用户明确要了 NPU，
            // 静默降级成 CPU 是最坏的结果。
            eprintln!(
                "注意：要了 NPU，但没找到可用的推理后端。\n  \
                 原因：{reason}\n  \
                 已按 CPU 继续；跑 `aha npu` 看这台机器缺什么。"
            );
        }
        return Ok(());
    };
    if !npu::is_npu_provider(&provider) {
        tracing::info!(%provider, "ONNX Runtime provider（只影响 SenseVoice / Kokoro）");
        return Ok(());
    }

    // plugin EP：真的跑在什么后端由配置文件的 backend_type 决定，aha 这边探不出来，
    // 只能确认配置文件在不在 —— 少写这个文件是最常见的失败原因。
    if provider.eq_ignore_ascii_case("plugin") {
        eprintln!(
            "注意：--provider plugin 没带配置文件，sherpa 会因为没有 library= 而退回 CPU。\n  \
             写成 --provider plugin:/path/to/qnn.conf"
        );
        return Ok(());
    }
    if let Some(config_path) = npu::plugin_config(&provider) {
        if std::path::Path::new(config_path).is_file() {
            tracing::info!(%provider, "ONNX Runtime 走 plugin EP（配置文件 {config_path}）");
        } else {
            eprintln!(
                "注意：--provider {provider} 的配置文件不存在：{config_path}\n  \
                 会话会退回 CPU；跑 `aha npu` 看看这台机器有哪些 QNN 库可用。"
            );
        }
        return Ok(());
    }

    let report = npu::probe();
    match report.recommended() {
        Some(found) if found == provider => tracing::info!(
            %provider,
            htp_arch = ?report.htp_arch,
            "NPU provider 已启用（ONNX Runtime 模型）；若日志出现 'Failed to create' / 'Fallback to cpu'，\
             说明链接的 libonnxruntime 没编进这个 EP"
        ),
        Some(found) => {
            eprintln!("注意：环境里只有 {found} 运行时，{provider} 大概率不可用，会话会退回 CPU。")
        }
        None => {
            let reason = report
                .notes
                .first()
                .map(String::as_str)
                .unwrap_or("没有可用 NPU 运行时");
            eprintln!(
                "注意：--provider {provider} 在这台机器上大概率无效，会话会退回 CPU。\n  \
                 原因：{reason}"
            );
        }
    }
    Ok(())
}

/// 装了 `RUST_LOG` 才初始化日志：`RUST_LOG=aha::server::realtime=debug,info aha multi …`。
///
/// 不设环境变量时完全不装 subscriber，行为与以前一致（tracing 宏是 no-op）。
pub(crate) fn init_tracing() {
    if std::env::var_os("RUST_LOG").is_none() {
        return;
    }
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(std::io::stderr)
        .try_init();
}

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

/// 桌宠委托与本地转换的模型（SenseVoice / Kokoro / Laya / TurboVLA / GPT-SoVITS）用各自的
/// ModelScope 镜像下载，落盘位置统一是 `save_dir/<model_id>`（跟 `aha download` 一致）。
///
/// - `Some(path)`：权重已就位，直接用这个目录；
/// - `None`：不是这些模型，交给 [`download_model`] 按 ModelScope 仓库 id 全量下。
///
/// 这几条路径内部固定重试 3 次（跟 `--download-retries` 的默认值一致）。
#[allow(unused_variables)] // 一个桌宠特性都没编时参数用不上
async fn pet_model_or_download(
    model: WhichModel,
    model_id: &str,
    save_dir: &str,
) -> anyhow::Result<Option<String>> {
    if model.is_gguf() {
        let dir = format!("{save_dir}/{model_id}");
        tracing::info!(dir = %dir, "检查 GGUF 权重（ModelScope 镜像）");
        aha::models::download::ensure_gguf(model_id, std::path::Path::new(&dir), 3).await?;
        return Ok(Some(dir));
    }
    #[cfg(any(feature = "sherpa-asr", feature = "sherpa-tts"))]
    if model.is_sherpa_model() {
        let dir = format!("{save_dir}/{model_id}");
        tracing::info!(dir = %dir, "检查 sherpa 模型文件（ModelScope 镜像）");
        aha::models::common::sherpa::ensure_downloaded(&dir, model).await?;
        return Ok(Some(dir));
    }
    #[cfg(feature = "laya")]
    if model.is_laya_model() {
        let dir = format!("{save_dir}/{model_id}");
        let (path, source) = aha::models::laya::resolve(Some(&dir), model_id).await?;
        tracing::info!(dir = %path.display(), source, "Laya 模型目录就绪");
        return Ok(Some(path.display().to_string()));
    }
    #[cfg(feature = "vla")]
    if model.is_vla_model() {
        let dir = format!("{save_dir}/{model_id}");
        tracing::info!(dir = %dir, "检查 TurboVLA 权重（ModelScope 镜像）");
        aha::models::download::ensure_files(model_id, std::path::Path::new(&dir), 3).await?;
        return Ok(Some(dir));
    }
    #[cfg(feature = "gpt-sovits")]
    if model.is_gpt_sovits_model() {
        let dir = format!("{save_dir}/{model_id}");
        tracing::info!(dir = %dir, "检查 GPT-SoVITS 权重（ModelScope 镜像）");
        aha::models::download::ensure_files(model_id, std::path::Path::new(&dir), 3).await?;
        return Ok(Some(dir));
    }
    Ok(None)
}

/// GGUF 的权重路径：给了 `--gguf-path` 就用；没给就从 ModelScope 镜像下到
/// `save_dir/<model id>`（qwen3.5-gguf 或 hy-mt2-1.8b-1.25bit-gguf）。
async fn gguf_or_download(
    model: WhichModel,
    save_dir: Option<String>,
    gguf_path: Option<String>,
    mmproj_path: Option<String>,
) -> anyhow::Result<(String, Option<String>)> {
    if let Some(path) = gguf_path {
        return Ok((path, mmproj_path));
    }
    let save_dir = match save_dir {
        Some(dir) => dir,
        None => get_default_save_dir().expect("Failed to get home directory"),
    };
    let dir = format!("{save_dir}/{}", model.as_string());
    let weights =
        aha::models::download::ensure_gguf(&model.as_string(), std::path::Path::new(&dir), 3)
            .await?;
    Ok((
        weights.model.display().to_string(),
        weights.mmproj.map(|path| path.display().to_string()),
    ))
}

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
    init_inference_backend(&pet.device, pet.provider.as_deref())?;

    let (model_path, gguf, mmproj) = if model.is_gguf() {
        let (gguf, mmproj) = gguf_or_download(
            model,
            save_dir.clone(),
            path_common.gguf_path,
            path_common.mmproj_path,
        )
        .await?;
        ("GGUF".to_string(), Some(gguf), mmproj)
    } else if model.is_onnx() && !model.is_pet_delegated_model() {
        // 桌宠模型走 sherpa/sys1，不进 onnx 门禁（SenseVoice 的 id 里就带 "onnx"）
        return Err(anyhow!("onnx model not support now"));
    } else {
        let model_path = match path_common.weight_path {
            Some(path) => path,
            None => {
                let save_dir = match save_dir {
                    Some(dir) => dir,
                    None => get_default_save_dir().expect("Failed to get home directory"),
                };
                let max_retries = download_retries.unwrap_or(3);
                match pet_model_or_download(model, &model_id, &save_dir).await? {
                    Some(path) => path,
                    None => {
                        download_model(&model_id, &save_dir, max_retries).await?;
                        save_dir + "/" + &model_id
                    }
                }
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
            ctx_size: pet.ctx_size,
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

    init_inference_backend(&pet.device, pet.provider.as_deref())?;

    let mut model_path;
    let weight_path = path_common.weight_path.clone();
    let (gguf, mmproj) = if model.is_gguf() {
        model_path = "GGUF".to_string();
        let (gguf, mmproj) =
            gguf_or_download(model, None, path_common.gguf_path, path_common.mmproj_path).await?;
        (Some(gguf), mmproj)
    } else if model.is_onnx() && !model.is_pet_delegated_model() {
        // 桌宠模型走 sherpa/sys1，不进 onnx 门禁（SenseVoice 的 id 里就带 "onnx"）
        return Err(anyhow!("onnx model not support now"));
    } else {
        match weight_path.clone() {
            Some(path) => model_path = path,
            None => model_path = get_default_weight_path(model),
        }
        (None, None)
    };

    // 桌宠委托模式：SenseVoice / Kokoro 首次启动自己从 ModelScope 镜像按文件列表下载
    // （必填文件齐了就跳过）；Laya 先找 --weight-path / HF 缓存 / ~/.aha，
    // 都没有再从 ModelScope 镜像下。
    #[cfg(any(feature = "sherpa-asr", feature = "sherpa-tts"))]
    if model.is_sherpa_model() {
        tracing::info!(dir = %model_path, model = %model.as_string(), "检查 sherpa 模型文件");
        aha::models::common::sherpa::ensure_downloaded(&model_path, model).await?;
    }
    #[cfg(feature = "laya")]
    if model.is_laya_model() {
        let (path, source) =
            aha::models::laya::resolve(weight_path.as_deref(), &model.as_string()).await?;
        tracing::info!(dir = %path.display(), source, "Laya 模型目录就绪");
        model_path = path.display().to_string();
    }

    if !model.is_gguf() && !std::path::Path::new(&model_path).exists() {
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
            ctx_size: pet.ctx_size,
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

/// Run the 'multi' subcommand: 一个进程服务多个模型（桌宠 asr/tts/laya，可再加 llm）。
pub(crate) async fn run_multi(args: MultiArgs) -> anyhow::Result<()> {
    let MultiArgs {
        asr,
        tts,
        tts_model,
        llm,
        llm_model,
        laya,
        laya_path,
        vla,
        vla_device,
        server_common,
        pet,
    } = args;

    let mut specs: Vec<ModelSpec> = Vec::new();

    init_inference_backend(&pet.device, pet.provider.as_deref())?;

    // asr 角色：只有 sherpa-asr（SenseVoice）能提供。
    #[cfg(feature = "sherpa-asr")]
    if let Some(dir) = asr.as_deref() {
        tracing::info!(dir = %dir, role = "asr", "检查 sherpa 模型文件");
        aha::models::common::sherpa::ensure_downloaded(dir, WhichModel::SenseVoice).await?;
        specs.push(ModelSpec::new("asr", WhichModel::SenseVoice, dir));
    }
    #[cfg(not(feature = "sherpa-asr"))]
    if asr.is_some() {
        return Err(anyhow!("这个二进制没编 sherpa-asr 特性，不能启用 asr 角色"));
    }

    // tts 角色用哪个模型：默认 Kokoro（sherpa-tts）；传 gpt-sovits-v2pro-zh 时换成
    // 本地转换的 GPT-SoVITS（--tts 指向转换后的目录，不走 sherpa、不下载）。
    #[cfg(feature = "sherpa-tts")]
    let tts_model = tts_model.unwrap_or(WhichModel::Kokoro);
    #[cfg(not(feature = "sherpa-tts"))]
    let tts_model = tts_model
        .ok_or_else(|| anyhow!("这个二进制没编 sherpa-tts 特性，--tts 必须配 --tts-model"))?;

    if let Some(dir) = tts.as_deref() {
        #[cfg(any(feature = "sherpa-asr", feature = "sherpa-tts"))]
        if tts_model.is_sherpa_model() {
            tracing::info!(dir = %dir, role = "tts", "检查 sherpa 模型文件");
            aha::models::common::sherpa::ensure_downloaded(dir, tts_model).await?;
        }
        specs.push(ModelSpec::new("tts", tts_model, dir));
    }

    // llm 角色：真正会「回话」的对话模型——/realtime 的 ASR→LLM→TTS 里中间那段。
    // 显式 key 用 "llm"，客户端连 WebSocket 时带 `?model=llm` 就行。
    if let Some(dir) = llm.as_deref() {
        let llm_model = llm_model.unwrap_or(WhichModel::Qwen3_0_6B);
        tracing::info!(dir = %dir, role = "llm", model = %llm_model.as_string(), "登记对话模型");
        specs.push(ModelSpec::new("llm", llm_model, dir));
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
            ctx_size: pet.ctx_size,
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

/// Run the 'npu' subcommand: 探测 NPU 运行时（QNN / NNAPI）并给出 `--provider` 建议。
///
/// 不加载任何模型、不联网，手机上也能秒回。`--json` 给桌面端 / Android 壳读，
/// `-r` 只输出 provider 名字（探测不到就是 `cpu`）。
pub(crate) fn run_npu(args: NpuArgs) -> anyhow::Result<()> {
    let NpuArgs { json, recommended } = args;
    let report = aha::utils::npu::probe();

    if recommended {
        println!("{}", report.recommended().unwrap_or("cpu"));
        return Ok(());
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&report.to_json())?);
        return Ok(());
    }
    print!("{}", report.render());
    Ok(())
}

/// Run the 'qnn' subcommand: 直接检查/执行一个 QAIRT context binary。
///
/// 图名、张量名、类型、维度全部从 bin 里读，所以任何用
/// `scripts/qnn_convert.py` 转出来的模型都能用同一条命令跑。
/// `--input 名字=文件` 给指定张量喂数据，没给的补零；`--out-dir` 把每个输出
/// 张量按名字写成 `.bin`（float32 的话可以直接当 numpy 读）。
pub(crate) fn run_qnn(args: QnnArgs) -> anyhow::Result<()> {
    use aha::models::common::qnn::{self, Runtime};
    use std::time::Instant;

    let QnnArgs {
        context,
        model,
        backend,
        system,
        info,
        graph,
        inputs,
        out_dir,
    } = args;

    let runtime = match (context, model) {
        (Some(context), None) => {
            let context_path = std::path::PathBuf::from(&context);
            if !context_path.is_file() {
                anyhow::bail!("context binary 不存在：{context}");
            }
            Runtime::open(&context_path, &backend, &system)?
        }
        (None, Some(model)) => {
            let model_path = std::path::PathBuf::from(&model);
            if !model_path.is_file() {
                anyhow::bail!("模型库不存在：{model}");
            }
            Runtime::open_model_lib(&model_path, &backend)?
        }
        _ => anyhow::bail!("二选一：--context <context binary> 或 --model <libxxx.so>"),
    };
    let graphs = runtime.graphs();
    if graphs.is_empty() {
        anyhow::bail!("这个 QNN 会话里没有图");
    }

    for (index, item) in graphs.iter().enumerate() {
        let marker = if index == graph { "*" } else { " " };
        println!("{marker} graph[{index}] {}", item.name);
        for (role, tensors) in [("in", &item.inputs), ("out", &item.outputs)] {
            for tensor in tensors {
                let bytes = runtime.bytes_per_element(tensor.data_type) * tensor.elements as usize;
                println!(
                    "    {role} {:<24} type={:<3} dims={:?} bytes={bytes}",
                    tensor.name, tensor.data_type, tensor.dims
                );
            }
        }
    }
    if info {
        return Ok(());
    }

    let target = graphs
        .get(graph)
        .ok_or_else(|| anyhow::anyhow!("没有 graph[{graph}]（共 {} 张图）", graphs.len()))?;

    // 输入缓冲区：默认全零，`--input 名字=文件` 覆盖。
    let mut input_buffers: Vec<Vec<u8>> = target
        .inputs
        .iter()
        .map(|tensor| {
            vec![0u8; runtime.bytes_per_element(tensor.data_type) * tensor.elements as usize]
        })
        .collect();
    for spec in &inputs {
        let (name, path) = spec
            .split_once('=')
            .ok_or_else(|| anyhow::anyhow!("--input 要写成 名字=文件，实际是 {spec:?}"))?;
        let slot = target
            .inputs
            .iter()
            .position(|tensor| tensor.name == name)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "图里没有输入张量 {name:?}（可选：{}）",
                    target
                        .inputs
                        .iter()
                        .map(|t| t.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            })?;
        let data = std::fs::read(path)?;
        let expected = input_buffers[slot].len();
        if data.len() != expected {
            anyhow::bail!(
                "{path} 是 {} 字节，张量 {name:?} 需要 {expected} 字节",
                data.len()
            );
        }
        input_buffers[slot] = data;
    }

    let mut output_buffers: Vec<Vec<u8>> = target
        .outputs
        .iter()
        .map(|tensor| {
            vec![0u8; runtime.bytes_per_element(tensor.data_type) * tensor.elements as usize]
        })
        .collect();

    let started = Instant::now();
    runtime.run(graph, &mut input_buffers, &mut output_buffers)?;
    println!(
        "execute graph[{graph}] elapsed {:.1} ms",
        started.elapsed().as_secs_f64() * 1e3
    );

    match out_dir {
        Some(dir) => {
            std::fs::create_dir_all(&dir)?;
            for (tensor, data) in target.outputs.iter().zip(&output_buffers) {
                let path = std::path::Path::new(&dir).join(format!("{}.bin", tensor.name));
                std::fs::write(&path, data)?;
                println!("wrote {} ({} bytes)", path.display(), data.len());
            }
        }
        None => {
            for (tensor, data) in target.outputs.iter().zip(&output_buffers) {
                let sum: f64 = data.iter().map(|byte| *byte as f64).sum();
                println!(
                    "out {:<24} {} bytes（--out-dir 可落盘；字节和 {sum:.0}）",
                    tensor.name,
                    data.len()
                );
            }
        }
    }

    let _ = qnn::shim_path();
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

    // 桌宠模型（sherpa/laya/turbovla/gpt-sovits）走自己的 ModelScope 镜像，
    // GGUF 镜像只含两个文件、单独下载，其它模型整仓库全量下
    if model.is_gguf() {
        let (gguf, mmproj) = gguf_or_download(model, Some(save_dir.clone()), None, None).await?;
        println!("GGUF: {gguf}");
        if let Some(path) = mmproj {
            println!("mmproj: {path}");
        }
    } else if pet_model_or_download(model, &model_id, &save_dir)
        .await?
        .is_none()
    {
        download_model(&model_id, &save_dir, max_retries).await?;
    }

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

    init_inference_backend(&pet.device, pet.provider.as_deref())?;

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
            let (gguf_path, mmproj_path) = match path_common.gguf_path {
                Some(path) => (path, path_common.mmproj_path.clone()),
                None => {
                    let dir = format!(
                        "{}/{}",
                        get_default_save_dir().expect("Failed to get home directory"),
                        WhichModel::Qwen3_5Gguf.as_string()
                    );
                    let (gguf, mmproj) = aha::models::download::ensure_gguf_blocking(
                        &WhichModel::Qwen3_5Gguf.as_string(),
                        &dir,
                    )?;
                    (gguf, mmproj)
                }
            };
            qwen3_5::Qwen3_5Exec::run_gguf(
                &input,
                output.as_deref(),
                Some(gguf_path),
                mmproj_path,
            )?;
        }
        WhichModel::HyMT2Gguf => {
            let (gguf_path, _mmproj) = match path_common.gguf_path {
                Some(path) => (path, path_common.mmproj_path.clone()),
                None => {
                    let dir = format!(
                        "{}/{}",
                        get_default_save_dir().expect("Failed to get home directory"),
                        WhichModel::HyMT2Gguf.as_string()
                    );
                    aha::models::download::ensure_gguf_blocking(&model.as_string(), &dir)?
                }
            };
            #[cfg(feature = "llama")]
            {
                llama_cpp::LlamaCppExec::run_gguf(
                    &input,
                    output.as_deref(),
                    &gguf_path,
                    pet.ctx_size,
                    pet.threads,
                )?;
            }
            #[cfg(not(feature = "llama"))]
            {
                let _ = (&input, output, &gguf_path, pet.ctx_size, pet.threads);
                return Err(anyhow!(
                    "Hy-MT2 的 STQ 量化只能走 llama.cpp 后端：用 `cargo build --features llama` 编译，\
                     并先跑 native/scripts/build-llama.sh <abi> 生成静态库"
                ));
            }
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
        #[cfg(feature = "sherpa-asr")]
        WhichModel::SenseVoice => {
            sense_voice::SenseVoiceExec::run(&input, output.as_deref(), &weight_path)?;
        }
        #[cfg(feature = "sherpa-tts")]
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

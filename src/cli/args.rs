use aha::models::common::model_mapping::WhichModel;
use clap::{Args, Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(name = "aha")]
#[command(version, about, long_about = None)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
    /// Download model and start service (default)
    Cli(CliArgs),
    /// Start service only (--weight-path is optional, defaults to ~/.aha/{model_id})
    Serv(ServArgs),
    /// 一个进程同时服务多个模型（桌宠的 asr/tts/laya 三件套，单端口）
    Multi(MultiArgs),
    /// 校验模型与二进制编译特征是否对得上（桌宠拉起 Laya 前先问一次）
    Probe(ProbeArgs),
    /// List all running aha services
    Ps(ServListArgs),
    /// Delete a downloaded model from the default location (~/.aha/{model_id})
    Delete(DeleteArgs),
    /// Download model only
    Download(DownloadArgs),
    /// Run model inference directly
    Run(RunArgs),
    /// List all supported models
    List(ListArgs),
}

/// Common/shared arguments for server operations
#[derive(Args, Debug)]
pub(crate) struct ServerCommonArgs {
    /// Service listen address
    #[arg(short, long, default_value = "127.0.0.1")]
    pub address: String,

    /// Service listen port
    #[arg(short, long, default_value_t = 10100)]
    pub port: u16,

    /// Allow remote shutdown requests (default: local only, use with caution)
    #[arg(long)]
    pub allow_remote_shutdown: bool,

    /// 桌宠委托模式：父进程 pid；它一死这个服务也退出（防孤儿占端口/显存）。
    /// 0 = 不启用（默认）。
    #[arg(long, default_value_t = 0)]
    pub parent_pid: u32,
}

/// 桌宠三个模型的额外参数（只对 SenseVoice / Kokoro / Laya 生效，其余模型忽略）。
#[derive(Args, Debug, Default)]
pub(crate) struct PetServeArgs {
    /// Laya 推理设备。**编译期决定**：cpu 用 `--features laya` 编，cuda 用
    /// `--features laya-cuda`；对不上直接报错退出（不静默降级）。
    #[arg(long, default_value = "cpu")]
    pub device: String,
    /// Laya 低于这个置信度的 vla 判定回退 llm（默认 0.8，和 route.py 一致）。
    #[arg(long, default_value_t = 0.8)]
    pub min_confidence: f32,
    /// Laya 单次推理超时（毫秒）；0 用默认 15000。
    #[arg(long, default_value_t = 15_000)]
    pub request_timeout_ms: u64,
    /// 推理线程数（sherpa-onnx 的 intra-op 线程；0 = 按 CPU 自动，取逻辑核数）。
    #[arg(long, default_value_t = 0)]
    pub threads: i32,
}

#[derive(Args, Debug)]
pub(crate) struct PathCommonArgs {
    /// Local model weight path (skip download if provided)
    #[arg(long)]
    pub weight_path: Option<String>,
    /// Local GGUF model weight path (required for loading models with GGUF).
    #[arg(long)]
    pub gguf_path: Option<String>,

    /// Local path for mmproj GGUF model weights (required for loading with multimodel GGUF)
    #[arg(long)]
    pub mmproj_path: Option<String>,

    /// Local path for onnx model weights (required for loading with onnx)
    #[arg(long)]
    pub onnx_path: Option<String>,

    /// config path for onnx/gguf model need extra config file
    #[arg(long)]
    pub config_path: Option<String>,
}

/// Arguments for the 'cli' subcommand (download + serve)
#[derive(Args, Debug)]
pub(crate) struct CliArgs {
    /// Model type (required)
    #[arg(short, long)]
    pub model: WhichModel,

    #[command(flatten)]
    pub server_common: ServerCommonArgs,

    /// 桌宠三件套（SenseVoice / Kokoro / Laya）的额外参数
    #[command(flatten)]
    pub pet: PetServeArgs,

    /// Model download save directory
    #[arg(long)]
    pub save_dir: Option<String>,

    /// Download retry count
    #[arg(long)]
    pub download_retries: Option<u32>,

    #[command(flatten)]
    pub path_common: PathCommonArgs,
}

/// Arguments for the 'serv start' subcommand
#[derive(Args, Debug)]
pub(crate) struct ServArgs {
    /// Model type (required)
    #[arg(short, long)]
    pub model: WhichModel,

    #[command(flatten)]
    pub server_common: ServerCommonArgs,

    /// 桌宠三件套（SenseVoice / Kokoro / Laya）的额外参数
    #[command(flatten)]
    pub pet: PetServeArgs,

    #[command(flatten)]
    pub path_common: PathCommonArgs,
}

/// Arguments for the 'probe' subcommand
#[derive(Args, Debug)]
pub(crate) struct ProbeArgs {
    /// Model type (required)
    #[arg(short, long)]
    pub model: WhichModel,

    /// 请求的推理设备（只对 Laya 有意义）
    #[arg(long, default_value = "cpu")]
    pub device: String,

    /// 跑一遍判定语料再退出（只对 Laya 有意义；会加载模型）
    #[arg(long)]
    pub selftest: bool,
}

/// Arguments for the 'serv list' subcommand
#[derive(Args, Debug)]
pub(crate) struct ServListArgs {
    /// Compact output format
    #[arg(short, long)]
    pub compact: bool,
}

/// Arguments for the 'download' subcommand (download only)
#[derive(Args, Debug)]
pub(crate) struct DownloadArgs {
    /// Model type (required)
    #[arg(short, long)]
    pub model: WhichModel,

    /// Model download save directory
    #[arg(short, long)]
    pub save_dir: Option<String>,

    /// Download retry count
    #[arg(long)]
    pub download_retries: Option<u32>,
}

/// Arguments for the 'run' subcommand (direct inference)
#[derive(Args, Debug)]
pub(crate) struct RunArgs {
    /// Model type (required)
    #[arg(short, long)]
    pub model: WhichModel,

    /// Input text or file path
    #[arg(short, long, num_args = 1..=2, value_delimiter = ' ')]
    pub input: Vec<String>,

    /// Output file path (optional)
    #[arg(short, long)]
    pub output: Option<String>,

    #[command(flatten)]
    pub path_common: PathCommonArgs,
}

/// Arguments for the 'delete' subcommand (delete model from default location)
#[derive(Args, Debug)]
pub(crate) struct DeleteArgs {
    /// Model type (required)
    #[arg(short, long)]
    pub model: WhichModel,
}

/// Arguments for the 'list' subcommand (list all supported models)
#[derive(Args, Debug)]
pub(crate) struct ListArgs {
    /// Output models in JSON format (includes name, model_id, and type fields)
    #[arg(short, long)]
    pub json: bool,
}

/// Arguments for the 'multi' subcommand: 一个进程同时服务多个模型（桌宠的 asr/tts/laya）。
///
/// 三个角色共用一个端口、一套 DLL，各自独立加载、独立加锁（不同模型可并发）。
/// 没启用的角色对应端点直接返回 400；OpenAI 端点按请求里的 `model` 字段分发。
#[derive(Args, Debug)]
pub(crate) struct MultiArgs {
    /// SenseVoice 模型目录（启用 ASR 角色）；首次会自动下载
    #[arg(long)]
    pub asr: Option<String>,
    /// Kokoro 模型目录（启用 TTS 角色）；首次会自动下载
    #[arg(long)]
    pub tts: Option<String>,
    /// 启用 Laya 决策角色（模型优先用 HF 缓存，其次下载）
    #[arg(long)]
    pub laya: bool,
    /// Laya 模型目录（不给则先找 HF 缓存，再下载）
    #[arg(long)]
    pub laya_path: Option<String>,

    #[command(flatten)]
    pub server_common: ServerCommonArgs,

    #[command(flatten)]
    pub pet: PetServeArgs,
}

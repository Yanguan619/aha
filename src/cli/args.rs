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
    /// 探测手机/盒子上的 NPU 运行时（QNN / NNAPI），并给出 --provider 建议（仅 Android 构建）
    #[cfg(target_os = "android")]
    Npu(NpuArgs),
    /// 直接检查/执行一个 QAIRT context binary（模型无关：图与张量都从 bin 里读）
    #[cfg(feature = "qnn")]
    Qnn(QnnArgs),
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
    /// 推理设备：`cpu` / `cuda` / `npu` / `auto`（默认 auto：编译进了 cuda 就用 GPU）。
    /// Laya 不看这个值（它由 `--features laya-cuda` 编译期决定），GPT-SoVITS 看。
    ///
    /// `npu` 只对 ONNX Runtime 的模型有意义（SenseVoice / Kokoro）：candle 没有
    /// 移动端 NPU 后端，LLM/VL 用 `npu` 等于 CPU。等价于 `--provider <探测到的 NPU>`。
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// ONNX Runtime 的 execution provider（只对 SenseVoice / Kokoro 这类 sherpa 模型生效）。
    ///
    /// 取值：`cpu` / `cuda` / `coreml` / `trt` / `rknn` / `qnn` / `nnapi` / `openvino`，
    /// `auto`（默认）= 看 `AHA_PROVIDER` 环境变量，都没有就用 CPU。
    /// 手机上用 `qnn`（`--device npu` 是它的快捷方式）；`nnapi` 只在 Android 14 及更早有意义。
    /// 预编译的 `libonnxruntime.so` 没编进对应 EP 时它会打印 fallback 再退回 CPU，
    /// 想知道环境里到底有没有 QNN 运行时，先跑 `aha npu`。
    ///
    /// `plugin:<conf>` 走 ONNX Runtime 的 plugin EP（Qualcomm 的 onnxruntime-qnn 就是
    /// 这种形式，不需要自编 ORT）：配置文件是 `key=value` 行，`library=` 指向 plugin
    /// 的 .so，其余键原样交给 EP（如 `backend_type=htp`、`backend_path=libQnnHtp.so`、
    /// `htp_arch=81`）。需要链接进 ORT ≥ 1.23 且带 plugin EP 支持。
    #[arg(long)]
    pub provider: Option<String>,
    /// Laya 低于这个置信度的 vla 判定回退 llm（默认 0.8，和 route.py 一致）。
    #[arg(long, default_value_t = 0.8)]
    pub min_confidence: f32,
    /// Laya 单次推理超时（毫秒）；0 用默认 15000。
    #[arg(long, default_value_t = 15_000)]
    pub request_timeout_ms: u64,
    /// 推理线程数（sherpa-onnx 的 intra-op 线程；0 = 按 CPU 自动，取逻辑核数）。
    #[arg(long, default_value_t = 0)]
    pub threads: i32,
    /// 上下文长度（GGUF/llama.cpp 后端用；默认 4096，Hy-MT2 这类短文本翻译够用）。
    #[arg(long, default_value_t = 4096)]
    pub ctx_size: u32,
    /// 关掉启动预热。
    ///
    /// 预热会在端口开之前空跑一遍每个角色（ASR / TTS / 决策 / VLA），
    /// 把 ONNX Runtime 选 provider、candle 搬权重、CUDA 加载 kernel 这些
    /// **一次性**开销提前。这样用户第一次说话就不会"特别慢"。
    ///
    /// 代价是启动慢几秒。调试或模型不在本机时可以用这个开关跳过。
    #[arg(long)]
    pub no_warmup: bool,
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

/// Arguments for the 'npu' subcommand
#[cfg(target_os = "android")]
#[derive(Args, Debug)]
pub(crate) struct NpuArgs {
    /// 输出 JSON（给桌面端 / Android 壳读）
    #[arg(long)]
    pub json: bool,

    /// 只打印建议的 provider（探测不到就打印 cpu），方便脚本里 `$(aha npu -r)`
    #[arg(long, short)]
    pub recommended: bool,
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

    /// 推理设备（`--device cpu/cuda/auto`）；和 serv 用同一个参数
    #[command(flatten)]
    pub pet: PetServeArgs,

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

/// Arguments for the 'qnn' subcommand: 直接看/跑一个 QAIRT context binary。
///
/// 图名、张量名、类型、维度全部从 context binary 里读，所以对任何用
/// `scripts/qnn_convert.py` 转出来的模型都成立。
#[cfg(feature = "qnn")]
#[derive(Args, Debug)]
pub(crate) struct QnnArgs {
    /// context binary 路径（`--stage` 产物里的 `vitsS_ctx.bin` 之类）
    #[arg(long, conflicts_with = "model")]
    pub context: Option<String>,
    /// 模型库路径（`lib<name>.so`），权重是同目录的 `<name>.bin`。
    /// 每个进程都要 compose + finalize 一次图（SoVITS 实测数分钟），运行时请优先用
    /// `--context` 的预编译 context binary
    #[arg(long, conflicts_with = "context")]
    pub model: Option<String>,
    /// HTP 后端库；Android 上通常是绝对路径
    #[arg(long, default_value = "libQnnHtp.so")]
    pub backend: String,
    /// QNN system 库（读 context binary 元数据用）
    #[arg(long, default_value = "libQnnSystem.so")]
    pub system: String,
    /// 只打印图和张量，不执行
    #[arg(long)]
    pub info: bool,
    /// 跑第几张图（默认 0）
    #[arg(long, default_value_t = 0)]
    pub graph: usize,
    /// 输入：`名字=文件`，按张量名匹配；可重复。不给就用全零缓冲区跑通链路
    #[arg(long = "input", value_name = "NAME=PATH")]
    pub inputs: Vec<String>,
    /// 把输出写到这个目录（每个张量一个 `.bin`）
    #[arg(long)]
    pub out_dir: Option<String>,
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
    /// TTS 用哪个模型：默认 Kokoro（sherpa）；gpt-sovits-v2pro-zh 时 `--tts` 指向
    /// 本地转换的 GPT-SoVITS 目录（不下载、不走 sherpa）
    #[arg(long)]
    pub tts_model: Option<WhichModel>,
    /// 对话模型目录（启用 LLM 角色）；/realtime 这类端到端语音接口需要它
    #[arg(long)]
    pub llm: Option<String>,
    /// LLM 用哪个模型（不给则按 Qwen3-0.6B 处理）
    #[arg(long)]
    pub llm_model: Option<WhichModel>,
    /// 启用 Laya 决策角色（模型优先用 HF 缓存，其次下载）
    #[arg(long)]
    pub laya: bool,
    /// Laya 模型目录（不给则先找 HF 缓存，再下载）
    #[arg(long)]
    pub laya_path: Option<String>,
    /// TurboVLA 模型目录（启用 VLA 角色；先用 tools/vla/convert_turbovla.py 转换）
    #[arg(long)]
    pub vla: Option<String>,
    /// VLA 单独的推理设备（cpu / cuda）；不给则用 `--device`。
    /// laya 和 vla 想各走各的设备时用它（桌宠默认 laya=cpu、vla=cuda）。
    #[arg(long)]
    pub vla_device: Option<String>,

    #[command(flatten)]
    pub server_common: ServerCommonArgs,

    #[command(flatten)]
    pub pet: PetServeArgs,
}

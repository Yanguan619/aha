//! 移动端 NPU（QNN / NNAPI）能力探测 + ONNX Runtime execution provider 注册表。
//!
//! 为什么需要这一层：
//!
//! - **candle 没有移动端 NPU 后端**（没有 QNN / NNAPI / Metal ANE 的 kernel）。
//!   所以 LLM / VL 这些走 candle 的模型，NPU 是**不接**的，只能 CPU（或桌面 CUDA）。
//! - 真正能上 NPU 的是 **ONNX Runtime 的模型**：目前在 aha 里就是 sherpa 的
//!   SenseVoice（ASR）和 Kokoro（TTS）。它们的执行设备由 ORT 的
//!   *execution provider* 决定，对应 sherpa-onnx 的 `provider` 字段。
//! - provider 到底能不能生效，取决于**两件事**：
//!   1. 运行时库在不在（`libQnnHtp.so` / `libQnnSystem.so`；NNAPI 则是
//!      `libneuralnetworks.so` + 系统里的 neuralnetworks 服务）；
//!   2. 链接进来的 `libonnxruntime.so` 有没有编进对应的 EP。
//!      预编译包经常只有 CPU EP —— 那时候 ORT 会打印一句
//!      `Failed to create ... execution provider` 然后**静默退回 CPU**。
//!
//! 第 2 条是可以静态查的：ORT 为每个编进去的 EP 导出一个
//! `OrtSessionOptionsAppendExecutionProvider_<EP>` 符号，扫一遍
//! `libonnxruntime.so` 的字节就能把 EP 列表抠出来（[`scan_ort_eps`]）。
//! 实测（sherpa-onnx 2.0.1 的 Android 预编译包）只有 **CPU + NNAPI**，
//! 没有 QNN —— 所以在那套 lib 上 `--provider qnn` 必然退回 CPU。
//!
//! 典型用法（Android）：
//!
//! ```text
//! aha npu                  # 人看
//! aha npu --json           # 上层（desktop-pet / AhaNative.kt）读
//! aha multi --device npu   # 探测到 QNN 就用 QNN，否则老实 CPU
//! ```
//!
//! 注意：Android 17（API 37）起系统里的 neuralnetworks **服务**已经被移除
//! （`dumpsys neuralnetworks` 为空、`service list | grep neural` 无结果），
//! `libneuralnetworks.so` 还在但已经没有后端可用。所以 Android 上现实的选择只有
//! **QNN**；NNAPI 只对老机器 / 老 ROM 有意义。

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// sherpa-onnx / ONNX Runtime 认得的 provider 名字。
///
/// 和 sherpa-onnx 的 `provider` 字段一致；`auto` 是 aha 自己的伪值
/// （= 先环境变量、再探测、最后 CPU）。
pub const PROVIDERS: &[&str] = &[
    "cpu", "cuda", "coreml", "trt", "rknn", "qnn", "nnapi", "openvino", "plugin",
];

/// 真正跑在 NPU 上的 provider（`coreml` 走 ANE 但也算加速器，这里单独归 GPU 侧不列）。
pub const NPU_PROVIDERS: &[&str] = &["qnn", "nnapi", "rknn"];

/// 进程级生效的 provider（`--provider` 设进来的）。没设 = `None`。
///
/// 用 `OnceLock` 而不是 `Mutex<Option<..>>`：provider 只在启动时定一次，
/// 之后所有模型加载都读同一份；线程安全、无锁。
static ACTIVE: OnceLock<String> = OnceLock::new();

/// 规范化并校验 `--provider`。
///
/// - `""` / `auto` → `Ok(None)`（= 不指定，交给 [`effective_provider`] 走探测）
/// - 已知名字 → `Ok(Some(小写))`
/// - 其它 → 报错并把可选值列出来（clap 的 value_parser 不如这里好加别名）
pub fn parse_provider(spec: &str) -> anyhow::Result<Option<String>> {
    let spec = spec.trim();
    // `plugin:<conf>` 的路径区分大小写，所以先拆开再只对名字小写化。
    let (name, config) = match spec.split_once(':') {
        Some((name, rest)) => (name.trim(), Some(rest.trim())),
        None => (spec, None),
    };
    let lower = name.to_ascii_lowercase();
    match (lower.as_str(), config) {
        ("", _) | ("auto", _) => Ok(None),
        // 手滑写法：ORT 那边叫 qnn，有人会写 htp / npu。
        ("npu", _) => Ok(Some("auto-npu".to_string())),
        ("plugin", None) => Ok(Some("plugin".to_string())),
        ("plugin", Some(path)) if !path.is_empty() => Ok(Some(format!("plugin:{path}"))),
        ("plugin", Some(_)) => anyhow::bail!(
            "--provider plugin: 后面要跟配置文件路径，\
             例如 --provider plugin:/data/local/tmp/qnn.conf"
        ),
        (other, None) if PROVIDERS.contains(&other) => Ok(Some(other.to_string())),
        (other, Some(_)) => anyhow::bail!(
            "只有 plugin 支持 <provider>:<config> 的写法，{other:?} 只接受名字（收到 {spec:?}）"
        ),
        (other, None) => anyhow::bail!(
            "未知的 ONNX Runtime provider {other:?}（可选 {}；留空/auto = 自动探测）",
            PROVIDERS.join(" / ")
        ),
    }
}

/// `plugin:<conf>` 里的配置文件路径（sherpa 会自己 split，这里只用来做提示/校验）。
pub fn plugin_config(provider: &str) -> Option<&str> {
    let (name, rest) = provider.trim().split_once(':')?;
    name.trim()
        .eq_ignore_ascii_case("plugin")
        .then(|| rest.trim())
        .filter(|rest| !rest.is_empty())
}

/// 记下本次进程用的 provider。重复设置会被忽略（第一次生效）。
pub fn set_active_provider(provider: Option<&str>) {
    if let Some(provider) = provider.filter(|p| !p.is_empty()) {
        // `plugin:<conf>` 的配置路径分大小写（Android 上 /data/local/tmp/qnn.conf
        // 和 /data/local/TMP/qnn.conf 是两个文件），只有名字部分能小写化。
        let normalized = match provider.split_once(':') {
            Some((name, rest)) => format!("{}:{rest}", name.trim().to_ascii_lowercase()),
            None => provider.to_ascii_lowercase(),
        };
        let _ = ACTIVE.set(normalized);
    }
}

/// `--provider` 显式指定的值（不含环境变量、不含自动探测）。
pub fn active_provider() -> Option<&'static str> {
    ACTIVE.get().map(String::as_str)
}

/// 实际要交给 ORT 的 provider：`--provider` > `AHA_PROVIDER` > 不指定（CPU）。
///
/// **不做隐式自动切换**：Android 上 `--provider qnn` 而链接的 ORT 没有 QNN EP 时，
/// ORT 会打印 "Failed to create ... provider" 再退回 CPU；如果默认就自动切过去，
/// 原本好好的 CPU ASR 会平白多一段报错日志。所以自动探测只用于**建议**
/// （[`probe`] / `aha npu`），要上 NPU 得显式 `--device npu` 或 `--provider qnn`。
///
/// 例外是 `--device npu` 这种"我就要 NPU"的显式意图（内部记成 `auto-npu`）：
/// 这时按探测结果选，探测不到就按 qnn 试——让日志说明为什么没生效，
/// 比静默 CPU 好排查。
pub fn effective_provider() -> Option<String> {
    if let Some(active) = active_provider() {
        if active == "auto-npu" {
            // auto-npu 是"我就要 NPU"的意图，但只按探测结果走：探测说这套
            // ORT 没编 QNN EP 时硬试只会多一行 "Failed to create" 的报错，
            // 而且慢（QNN EP 初始化失败前会先试加载 HTP）。
            return probe().recommended().map(str::to_string);
        }
        return Some(active.to_string());
    }
    if let Ok(env) = std::env::var("AHA_PROVIDER") {
        let env = env.trim();
        if !env.is_empty() && !env.eq_ignore_ascii_case("auto") {
            // plugin 的配置路径同样不能小写化。
            return Some(match env.split_once(':') {
                Some((name, rest)) => format!("{}:{rest}", name.trim().to_ascii_lowercase()),
                None => env.to_ascii_lowercase(),
            });
        }
    }
    None
}

/// 这个 provider 是不是 NPU（决定 candle 模型要不要警告"你选了个它不支持的设备"）。
///
/// `plugin` 是"用户自己挂的 EP 库"，跑在哪由配置文件里的 `backend_type` 决定；
/// 对 aha 来说按 NPU 意图处理，免得被"这台机器没有 NPU 运行时"的提示劝退。
pub fn is_npu_provider(provider: &str) -> bool {
    let provider = provider.trim();
    let name = provider
        .split_once(':')
        .map_or(provider, |(name, _)| name.trim())
        .to_ascii_lowercase();
    name == "auto-npu" || name == "plugin" || NPU_PROVIDERS.contains(&name.as_str())
}

/// `--device` 里的 `npu` 是个**组合快捷键**：candle 侧保持 CPU，ORT 侧尽量上 NPU。
pub fn device_implies_npu(device: &str) -> bool {
    matches!(
        device.trim().to_ascii_lowercase().as_str(),
        "npu" | "npu-auto"
    )
}

// ---------------------------------------------------------------------------
// 探测
// ---------------------------------------------------------------------------

/// 一个运行时库的命中记录。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LibHit {
    pub name: String,
    pub path: PathBuf,
}

impl LibHit {
    pub fn dir(&self) -> &Path {
        self.path.parent().unwrap_or(Path::new(""))
    }
}

/// 链接进来的 ONNX Runtime（`libonnxruntime.so` / `onnxruntime.dll`）。
///
/// `eps` 是从动态符号里扫出来的：ORT 为每个编进去的 EP 导出一个
/// `OrtSessionOptionsAppendExecutionProvider_<EP>` 符号，扫这个前缀就能
/// **静态**判断"这个 .so 到底能不能跑 QNN"，不用起进程试。
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrtLib {
    pub path: PathBuf,
    pub eps: Vec<String>,
}

impl OrtLib {
    pub fn supports(&self, ep: &str) -> bool {
        self.eps.iter().any(|e| e.eq_ignore_ascii_case(ep))
    }
}

/// 宿主侧 QAIRT SDK 里和转换链有关的那几个目录。
///
/// 和 `NpuReport` 里那堆设备侧 `.so` 是两件事：这些是**编译/转换**用的，
/// 只有想走"任意 ONNX → HTP 模型"（`scripts/qnn_convert.py`）时才需要。
#[derive(Debug, Clone, Default)]
pub struct QnnSdk {
    /// `QNN_SDK_ROOT` 指到的目录。
    pub root: Option<PathBuf>,
    /// `include/QNN`（编译模型库要 `QnnInterface.h`）。
    pub include: Option<PathBuf>,
    /// `share/QNN/converter/jni`（`QnnModel.cpp` 等包装层源码）。
    pub jni: Option<PathBuf>,
    /// `bin/<host>/qnn-onnx-converter`。
    pub converter: Option<PathBuf>,
    /// `lib/aarch64-android`（推给手机的那套运行库）。
    pub android_runtime: Option<PathBuf>,
}

impl QnnSdk {
    /// 转换链要的东西是不是齐了。
    pub fn is_complete(&self) -> bool {
        self.include.is_some() && self.converter.is_some() && self.android_runtime.is_some()
    }

    /// 缺哪一块，给人看的话（只算转换本身必需的三样）。
    pub fn missing(&self) -> Vec<&'static str> {
        let mut out = Vec::new();
        if self.include.is_none() {
            out.push("include/QNN");
        }
        if self.converter.is_none() {
            out.push("bin/<host>/qnn-onnx-converter");
        }
        if self.android_runtime.is_none() {
            out.push("lib/aarch64-android");
        }
        out
    }

    /// 编模型库（`--build-so`）才需要的 jni 包装层。
    pub fn has_jni(&self) -> bool {
        self.jni.is_some()
    }
}

/// 环境里实际摸到了什么。
#[derive(Debug, Clone, Default)]
pub struct NpuReport {
    pub os: String,
    pub arch: String,
    pub dirs: Vec<PathBuf>,
    /// `libQnnHtp.so`——QNN 的 HTP（Hexagon）后端本体。
    pub qnn_htp: Option<LibHit>,
    /// `libQnnSystem.so`——QNN 的模型/图管理，几乎总要一起带。
    pub qnn_system: Option<LibHit>,
    /// `libQnnHtpV{arch}Stub.so` 里的 arch，如 81。
    pub htp_arch: Option<u32>,
    /// `libneuralnetworks.so`——NNAPI 运行时（服务可能已死，见模块注释）。
    pub nnapi: Option<LibHit>,
    /// `libsnpe.so`——老的高通 SNPE（已被 QNN 取代）。
    pub snpe: Option<LibHit>,
    /// 找到的 ONNX Runtime 本体 + 它编进去的 EP。
    pub ort: Option<OrtLib>,
    /// 本次构建有没有链接 ONNX Runtime（`sherpa-*` / `onnx-runtime` 特性）。
    ///
    /// 没链接时 NPU provider 无处可去：探测到 QNN 也没用，
    /// 报告不该给出"建议 --provider qnn"这种误导。
    pub ort_linked: bool,
    /// 其它认出来的相关库（`libcdsprpc.so` / `libQnnTfliteDelegate.so` …），只做展示。
    pub others: Vec<LibHit>,
    /// 宿主侧 QAIRT SDK（转换链用，和上面的设备侧运行库无关）。
    pub sdk: Option<QnnSdk>,
    pub notes: Vec<String>,
}

impl NpuReport {
    /// 建议的 provider。
    ///
    /// 两个条件都要满足才建议 QNN：运行时库齐了，**且** ORT 里确实有 QNN EP
    /// （`ort` 为 `None` = 没找到 ORT 本体，无从判断，按"有戏"处理并给提示）。
    pub fn recommended(&self) -> Option<&'static str> {
        // 这个二进制根本没链接 ONNX Runtime：provider 交给谁都没意义。
        if !self.ort_linked {
            return None;
        }
        let qnn_runtime = self.qnn_htp.is_some() && self.qnn_system.is_some();
        let qnn_ep = self.ort.as_ref().map(|ort| ort.supports("QNN"));
        if qnn_runtime && qnn_ep != Some(false) {
            return Some("qnn");
        }
        let nnapi_ep = self.ort.as_ref().map(|ort| ort.supports("Nnapi"));
        if self.nnapi.is_some() && nnapi_ep != Some(false) {
            return Some("nnapi");
        }
        None
    }

    /// 人看的报告。
    pub fn render(&self) -> String {
        let mut out = String::new();
        out.push_str("aha npu —— 推理后端能力探测\n");
        out.push_str(&format!("  平台        {}/{}\n", self.os, self.arch));
        let dirs: Vec<String> = self.dirs.iter().map(|d| d.display().to_string()).collect();
        out.push_str(&format!("  搜索目录    {}\n", dirs.join(", ")));

        match (&self.qnn_htp, &self.qnn_system) {
            (Some(htp), Some(system)) => {
                let arch = self
                    .htp_arch
                    .map(|a| format!("（HTP v{a}）"))
                    .unwrap_or_default();
                out.push_str(&format!(
                    "  QNN         {}{arch} @ {}\n",
                    htp.name,
                    htp.dir().display()
                ));
                out.push_str(&format!(
                    "              {} @ {}\n",
                    system.name,
                    system.dir().display()
                ));
            }
            (Some(htp), None) => out.push_str(&format!(
                "  QNN         只有 {}，缺 libQnnSystem.so —— 大概率加载不了\n",
                htp.path.display()
            )),
            _ => out.push_str("  QNN         未找到（libQnnHtp.so / libQnnSystem.so）\n"),
        }

        if let Some(nnapi) = &self.nnapi {
            out.push_str(&format!(
                "  NNAPI       {} @ {}\n",
                nnapi.name,
                nnapi.dir().display()
            ));
        } else {
            out.push_str("  NNAPI       未找到\n");
        }
        match &self.ort {
            Some(ort) => {
                let eps = if ort.eps.is_empty() {
                    "没扫到 EP 符号".to_string()
                } else {
                    ort.eps.join(", ")
                };
                out.push_str(&format!(
                    "  ONNX RT     {}（EP: {eps}）\n",
                    ort.path.display()
                ));
            }
            None if !self.ort_linked => {
                out.push_str("  ONNX RT     未链接（本构建没编 sherpa / onnx-runtime 特性）\n")
            }
            None => out.push_str("  ONNX RT     未找到（不在搜索目录里）\n"),
        }
        if let Some(snpe) = &self.snpe {
            out.push_str(&format!(
                "  SNPE        {}（已被 QNN 取代，不接）\n",
                snpe.path.display()
            ));
        }
        for lib in &self.others {
            out.push_str(&format!("  其它        {}\n", lib.path.display()));
        }

        // 宿主侧转换链：和设备上的运行库是两回事，分开报。
        match &self.sdk {
            Some(sdk) if sdk.is_complete() => {
                out.push_str(&format!(
                    "  QAIRT SDK   就绪 @ {}（可把任意静态 ONNX 转成 HTP 模型）\n",
                    sdk.root
                        .as_ref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default()
                ));
                if !sdk.has_jni() {
                    out.push_str(
                        "              缺 share/QNN/converter/jni —— 只能出 .cpp/.bin，编不了模型库\n",
                    );
                }
            }
            Some(sdk) => out.push_str(&format!(
                "  QAIRT SDK   不完整（缺 {}）\n",
                sdk.missing().join(" / ")
            )),
            None => out.push_str(
                "  QAIRT SDK   未找到（设 QNN_SDK_ROOT 才能把任意 ONNX 转成 HTP 模型）\n",
            ),
        }

        match self.recommended() {
            Some(provider) => out.push_str(&format!(
                "  建议        --provider {provider}（SenseVoice / Kokoro 走 ONNX Runtime）\n"
            )),
            None if !self.ort_linked => out.push_str(
                "  建议        本构建没有 ONNX Runtime，NPU 无处可用；\
                 要跑 SenseVoice / Kokoro 请用 --features sherpa 构建\n",
            ),
            None => out.push_str(
                "  建议        没有可用 NPU 运行时 —— 保持 CPU；candle 模型本来也没有 NPU 后端\n",
            ),
        }
        for note in &self.notes {
            out.push_str(&format!("  注意        {note}\n"));
        }
        out
    }

    /// 机器读的报告（`aha npu --json`）。
    pub fn to_json(&self) -> serde_json::Value {
        use serde_json::json;
        let libs = |hit: &Option<LibHit>| {
            hit.as_ref()
                .map(|h| json!({ "name": h.name, "path": h.path.display().to_string() }))
        };
        json!({
            "os": self.os,
            "arch": self.arch,
            "search_dirs": self.dirs.iter().map(|d| d.display().to_string()).collect::<Vec<_>>(),
            "qnn": {
                "htp": libs(&self.qnn_htp),
                "system": libs(&self.qnn_system),
                "htp_arch": self.htp_arch,
            },
            "nnapi": libs(&self.nnapi),
            "snpe": libs(&self.snpe),
            "ort": self.ort.as_ref().map(|ort| json!({
                "path": ort.path.display().to_string(),
                "eps": ort.eps,
            })),
            "others": self.others.iter().map(|h| json!({ "name": h.name, "path": h.path.display().to_string() })).collect::<Vec<_>>(),
            "recommended_provider": self.recommended(),
            "ort_linked": self.ort_linked,
            "qnn_sdk": self.sdk.as_ref().map(|sdk| json!({
                "root": sdk.root.as_ref().map(|p| p.display().to_string()),
                "include": sdk.include.as_ref().map(|p| p.display().to_string()),
                "converter": sdk.converter.as_ref().map(|p| p.display().to_string()),
                "jni": sdk.jni.as_ref().map(|p| p.display().to_string()),
                "android_runtime": sdk.android_runtime.as_ref().map(|p| p.display().to_string()),
                "complete": sdk.is_complete(),
                "missing": sdk.missing(),
            })),
            "notes": self.notes,
        })
    }
}

/// 在 `base` 下面找 `…/<tail>/<marker>`，允许中间夹 0~4 层通配目录。
///
/// 为什么需要它：QAIRT 的 zip 常被**拆成几份**解包（bin+lib 一份、`share/` 一份、
/// `include/` 一份），`QNN_SDK_ROOT` 往往只指向其中一份，另两份可能落在同级的
/// 兄弟目录里（`lx/` 和 `x/`）。扫描按"目录预算"封顶，避免在 `E:\weights`
/// 这种放着几十个大模型的目录上呆住。
#[cfg(target_os = "android")]
fn find_under(base: &Path, tail: &[&str], marker: &str) -> Option<PathBuf> {
    /// 最多翻这么多个目录就放弃。
    const DIR_BUDGET: usize = 2000;
    /// 往下最多钻几层（`x/qairt/2.40.0/share/…` 这种布局要 4 层）。
    const MAX_DEPTH: usize = 4;

    fn probe(dir: &Path, tail: &[&str], marker: &str) -> Option<PathBuf> {
        let mut target = dir.to_path_buf();
        for seg in tail {
            target.push(seg);
        }
        target.join(marker).is_file().then_some(target)
    }

    fn walk(
        dir: &Path,
        depth: usize,
        tail: &[&str],
        marker: &str,
        found: &mut Option<PathBuf>,
        budget: &mut usize,
    ) {
        if found.is_some() || *budget == 0 {
            return;
        }
        *budget -= 1;
        if let Some(hit) = probe(dir, tail, marker) {
            *found = Some(hit);
            return;
        }
        if depth >= MAX_DEPTH {
            return;
        }
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        let mut subs: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        subs.sort();
        for sub in subs {
            walk(&sub, depth + 1, tail, marker, found, budget);
            if found.is_some() || *budget == 0 {
                return;
            }
        }
    }

    let mut found = None;
    let mut budget = DIR_BUDGET;
    walk(base, 1, tail, marker, &mut found, &mut budget);
    found
}

/// 从 `QNN_SDK_ROOT` 出发（含往上几层）摸一遍转换链要的目录。
#[cfg(target_os = "android")]
pub fn probe_qnn_sdk() -> Option<QnnSdk> {
    let root_env = std::env::var("QNN_SDK_ROOT")
        .ok()
        .filter(|s| !s.is_empty())?;
    let root = PathBuf::from(root_env);
    let mut sdk = QnnSdk {
        root: Some(root.clone()),
        ..Default::default()
    };

    let mut bases: Vec<PathBuf> = vec![root.clone()];
    let mut cursor = root.clone();
    for _ in 0..4 {
        let Some(parent) = cursor.parent() else { break };
        cursor = parent.to_path_buf();
        bases.push(cursor.clone());
    }

    for base in &bases {
        if sdk.include.is_none() {
            sdk.include = find_under(base, &["include", "QNN"], "QnnInterface.h");
        }
        if sdk.jni.is_none() {
            sdk.jni = find_under(base, &["share", "QNN", "converter", "jni"], "QnnModel.cpp");
        }
        if sdk.android_runtime.is_none() {
            sdk.android_runtime = find_under(base, &["lib", "aarch64-android"], "libQnnHtp.so");
        }
        if sdk.converter.is_none() {
            for platform in ["x86_64-windows-msvc", "x86_64-linux-clang"] {
                // `find_under` 返回的是"装着 marker 的目录"，这里要的是脚本本身。
                if let Some(dir) = find_under(base, &["bin", platform], "qnn-onnx-converter") {
                    sdk.converter = Some(dir.join("qnn-onnx-converter"));
                    break;
                }
            }
        }
        // jni 也找到了才算摸透：它常在另一份解包（`x/` vs `lx/`）里，
        // 所以不能一凑齐三件必需目录就收工。
        if sdk.is_complete() && sdk.has_jni() {
            break;
        }
    }
    Some(sdk)
}

/// 要扫的目录，按优先级排（先找到的先用）。
///
/// `AHA_NPU_LIB_DIR` 优先：APK 里 native 库的绝对路径由 Java 侧传进来，
/// 桌面/调试时也能手动指（多个用 `;` 或 `:` 分隔）。
#[cfg(target_os = "android")]
pub fn probe_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(explicit) = std::env::var("AHA_NPU_LIB_DIR") {
        for part in explicit
            .split([';', ':'])
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            dirs.push(PathBuf::from(part));
        }
    }
    // 可执行文件旁边（Windows 解压目录 / Android 上 dalvik 解出来的 nativeLibraryDir）
    if let Ok(exe) = std::env::current_exe()
        && let Some(parent) = exe.parent()
    {
        dirs.push(parent.to_path_buf());
    }
    if cfg!(target_os = "android") || cfg!(target_os = "linux") {
        for dir in [
            "/system_ext/lib64/litert",
            "/vendor/lib64/hw",
            "/vendor/lib64",
            "/odm/lib64/hw",
            "/odm/lib64",
            "/system/lib64",
        ] {
            dirs.push(PathBuf::from(dir));
        }
    }
    dirs.dedup();
    dirs
}

/// 扫一遍环境，看有没有 QNN / NNAPI 运行时。
///
/// 结果进程内缓存：探测要读一遍 `libonnxruntime.so`（20MB+），而 provider 在
/// 一次启动里是固定的，没必要每个模型加载都重扫。
#[cfg(target_os = "android")]
pub fn probe() -> NpuReport {
    static REPORT: OnceLock<NpuReport> = OnceLock::new();
    REPORT.get_or_init(probe_uncached).clone()
}

/// 桌面没有"NPU 运行时"这回事：给一个空报告，调用方统一按"没有 NPU"处理。
///
/// 桌面仍然支持 ONNX Runtime 的非移动端 provider（cuda / coreml / trt / openvino），
/// 那部分注册表不受影响。
#[cfg(not(target_os = "android"))]
pub fn probe() -> NpuReport {
    NpuReport {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        ort_linked: cfg!(any(
            feature = "sherpa-asr",
            feature = "sherpa-tts",
            feature = "onnx-runtime"
        )),
        notes: vec!["NPU 运行时探测只在 Android 构建里编（桌面没有 `aha npu`）".to_string()],
        ..Default::default()
    }
}

#[cfg(target_os = "android")]
fn probe_uncached() -> NpuReport {
    let dirs = probe_dirs();
    let mut report = NpuReport {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        dirs: dirs.clone(),
        ort_linked: cfg!(any(
            feature = "sherpa-asr",
            feature = "sherpa-tts",
            feature = "onnx-runtime"
        )),
        sdk: probe_qnn_sdk(),
        ..Default::default()
    };

    let mut ort_path: Option<PathBuf> = None;

    for dir in &dirs {
        let Ok(entries) = std::fs::read_dir(dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if ort_path.is_none() && is_ort_lib(&name) {
                ort_path = Some(entry.path());
            }
            let Some(kind) = classify(&name) else {
                continue;
            };
            let hit = LibHit {
                name: name.clone(),
                path: entry.path(),
            };
            match kind {
                LibKind::QnnHtp if report.qnn_htp.is_none() => report.qnn_htp = Some(hit),
                LibKind::QnnSystem if report.qnn_system.is_none() => report.qnn_system = Some(hit),
                LibKind::Nnapi if report.nnapi.is_none() => report.nnapi = Some(hit),
                LibKind::Snpe if report.snpe.is_none() => report.snpe = Some(hit),
                // 同一个库经常在 /system_ext 和 /odm 各有一份，按名字去重。
                LibKind::Other if !report.others.iter().any(|seen| seen.name == hit.name) => {
                    report.others.push(hit);
                }
                _ => {}
            }
            if let Some(arch) = htp_arch(&name) {
                report.htp_arch = Some(report.htp_arch.map_or(arch, |old| old.max(arch)));
            }
        }
    }

    // ORT 本体最后扫：要把整个 .so 读一遍（20MB+），但一个进程只做一次。
    report.ort = ort_path.map(|path| {
        let eps = scan_ort_eps(&path);
        OrtLib { path, eps }
    });

    // notes 第一条会被 CLI 拿去当日志里的原因，所以最"根因"的放最前面。
    if !report.ort_linked {
        report.notes.push(
            "这个构建没有链接 ONNX Runtime（缺 sherpa-asr / sherpa-tts / onnx-runtime 特性）：\
             NPU provider 只对 SenseVoice / Kokoro 有效，本构建上无处生效"
                .to_string(),
        );
    }
    if report.qnn_htp.is_some() && report.qnn_system.is_none() {
        report.notes.push(
            "有 libQnnHtp.so 但缺 libQnnSystem.so：QNN 初始化会失败，别指望 --provider qnn"
                .to_string(),
        );
    }
    if report.ort_linked && report.qnn_htp.is_some() && report.qnn_system.is_some() {
        match &report.ort {
            Some(ort) if !ort.supports("QNN") => report.notes.push(format!(
                "QNN 运行时库齐了，但这套 ONNX Runtime 没编 QNN EP（只扫到 {}）：\
                 --provider qnn 会静默退回 CPU。要真上 HTP 得自编 ORT(--use_qnn) + sherpa-onnx，\
                 见 docs/npu.zh-CN.md",
                if ort.eps.is_empty() {
                    "无 EP 符号".to_string()
                } else {
                    ort.eps.join("/")
                }
            )),
            Some(_) => report.notes.push(
                "QNN 运行时和 ORT 的 QNN EP 都在：--provider qnn 应该能真正跑在 HTP 上。\
                 首次初始化有 1-2s 开销，warmup 会做掉"
                    .to_string(),
            ),
            None => report.notes.push(
                "QNN 运行时齐了，但搜索目录里没有 libonnxruntime.so，确认不了它有没有编 QNN EP；\
                 跑起来看日志里有没有 'Failed to create' / 'Fallback to cpu'"
                    .to_string(),
            ),
        }
    }
    if report.recommended() == Some("nnapi") {
        report.notes.push(
            "Android 15+（尤其 16/17）已移除 neuralnetworks 服务：libneuralnetworks.so 在，但后端通常不可用；\
             Android 上优先 QNN"
                .to_string(),
        );
    }
    report
}

/// 这份目录里的 ONNX Runtime 本体。
#[cfg(target_os = "android")]
fn is_ort_lib(name: &str) -> bool {
    matches!(
        name,
        "libonnxruntime.so" | "libonnxruntime.dylib" | "onnxruntime.dll" | "libonnxruntime.dll"
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
enum LibKind {
    QnnHtp,
    QnnSystem,
    Nnapi,
    Snpe,
    Other,
    Skip,
}

/// 把一个文件名归到某一类。非共享库（Stub/Skel/其它架构）返回 `Skip`。
#[cfg(target_os = "android")]
fn classify(name: &str) -> Option<LibKind> {
    // HTP 的 Stub/Skel 是给 skel 侧编译用的，不参与 EP 初始化；只记 arch。
    if htp_arch(name).is_some() {
        return Some(LibKind::Skip);
    }
    Some(match name {
        "libQnnHtp.so" | "libQnnHtpV73CalculatorStub.so" => LibKind::QnnHtp,
        "libQnnSystem.so" => LibKind::QnnSystem,
        // 真正的 NNAPI 运行时是 libneuralnetworks.so；packageinfo 那个是给
        // PackageManager 看证书的壳，不算运行时（Android 17 上就只剩它）。
        "libneuralnetworks.so" => LibKind::Nnapi,
        "libneuralnetworks_packageinfo.so" => LibKind::Other,
        "libsnpe.so" => LibKind::Snpe,
        "libQnnTfliteDelegate.so"
        | "libcdsprpc.so"
        | "libadsprpc.so"
        | "libQnnHtpPrepare.so"
        | "libQnnHtpNetRunExtensions.so"
        | "libtensorflowlite_jni.so" => LibKind::Other,
        _ => return None,
    })
}

/// 从 `libQnnHtpV81Stub.so` 里抠出 SoC 的 HTP 版本号。
#[cfg(target_os = "android")]
pub fn htp_arch(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("libQnnHtpV")?;
    let digits = rest.strip_suffix("Stub.so")?;
    digits.parse().ok()
}

/// ORT 给每个编进去的 EP 导出的符号前缀。
#[cfg_attr(not(target_os = "android"), allow(dead_code))]
const ORT_EP_SYMBOL: &[u8] = b"OrtSessionOptionsAppendExecutionProvider_";

/// 扫 `libonnxruntime.so` 里编进了哪些 EP。
///
/// 不 dlopen（Android 上加载 20MB 的库代价不小，而且 `aha npu` 可能在别的平台上
/// 分析一份拷过来的 .so），直接把文件当字节串找符号前缀，再把名字尾巴读出来：
/// `OrtSessionOptionsAppendExecutionProvider_QNN` / `_Nnapi` / `_CPU` …
#[cfg(target_os = "android")]
pub fn scan_ort_eps(path: &Path) -> Vec<String> {
    const MAX_BYTES: u64 = 1 << 30;
    if std::fs::metadata(path)
        .map(|meta| meta.len() > MAX_BYTES)
        .unwrap_or(true)
    {
        return Vec::new();
    }
    let Ok(bytes) = std::fs::read(path) else {
        return Vec::new();
    };
    let mut eps: Vec<String> = Vec::new();
    let mut start = 0;
    while let Some(offset) = find_bytes(&bytes[start..], ORT_EP_SYMBOL) {
        let name_at = start + offset + ORT_EP_SYMBOL.len();
        let mut end = name_at;
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') {
            end += 1;
        }
        if end > name_at {
            let ep = String::from_utf8_lossy(&bytes[name_at..end]).into_owned();
            if !eps.iter().any(|seen| seen.eq_ignore_ascii_case(&ep)) {
                eps.push(ep);
            }
        }
        if name_at >= bytes.len() {
            break;
        }
        start = name_at;
    }
    eps
}

/// 朴素子串查找：needle 很短、命中很少，不值得为它引依赖。
#[cfg(target_os = "android")]
fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || haystack.len() < needle.len() {
        return None;
    }
    let first = needle[0];
    for i in 0..=haystack.len() - needle.len() {
        if haystack[i] == first && &haystack[i..i + needle.len()] == needle {
            return Some(i);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_os = "android")]
    fn htp_arch_is_parsed_from_stub_names() {
        assert_eq!(htp_arch("libQnnHtpV81Stub.so"), Some(81));
        assert_eq!(htp_arch("libQnnHtpV68Stub.so"), Some(68));
        assert_eq!(htp_arch("libQnnHtpV81Skel.so"), None);
        assert_eq!(htp_arch("libQnnHtp.so"), None);
        assert_eq!(htp_arch("libQnnHtpVxxStub.so"), None);
    }

    #[test]
    #[cfg(target_os = "android")]
    fn classify_recognises_runtimes() {
        assert_eq!(classify("libQnnHtp.so"), Some(LibKind::QnnHtp));
        assert_eq!(classify("libQnnSystem.so"), Some(LibKind::QnnSystem));
        assert_eq!(classify("libneuralnetworks.so"), Some(LibKind::Nnapi));
        assert_eq!(classify("libsnpe.so"), Some(LibKind::Snpe));
        assert_eq!(classify("libQnnHtpV81Stub.so"), Some(LibKind::Skip));
        assert_eq!(classify("libc++.so"), None);
    }

    #[test]
    fn provider_names_are_normalised_and_validated() {
        assert_eq!(parse_provider("").unwrap(), None);
        assert_eq!(parse_provider("auto").unwrap(), None);
        assert_eq!(parse_provider(" QNN ").unwrap(), Some("qnn".to_string()));
        assert_eq!(parse_provider("npu").unwrap(), Some("auto-npu".to_string()));
        let err = parse_provider("tpu").unwrap_err().to_string();
        assert!(err.contains("未知的 ONNX Runtime provider"), "{err}");
        assert!(err.contains("qnn"), "{err}");
    }

    #[test]
    fn npu_providers_are_flagged() {
        assert!(is_npu_provider("qnn"));
        assert!(is_npu_provider("NNAPI"));
        assert!(!is_npu_provider("cuda"));
        assert!(!is_npu_provider("cpu"));
        assert!(device_implies_npu(" npu "));
        assert!(!device_implies_npu("gpu"));
    }

    #[test]
    fn plugin_provider_keeps_its_config_path() {
        // 路径分大小写，解析时不能顺手小写化。
        assert_eq!(
            parse_provider("plugin:/data/local/tmp/qnn.conf").unwrap(),
            Some("plugin:/data/local/tmp/qnn.conf".to_string())
        );
        assert_eq!(
            parse_provider("  PLUGIN:D:\\tmp\\Qnn.conf ").unwrap(),
            Some("plugin:D:\\tmp\\Qnn.conf".to_string())
        );
        assert_eq!(
            parse_provider("plugin").unwrap(),
            Some("plugin".to_string())
        );
        // 只有 plugin 能带 `:conf`，别的 provider 带配置是写错了。
        let err = parse_provider("qnn:qnn.conf").unwrap_err().to_string();
        assert!(err.contains("只有 plugin"), "{err}");
        let err = parse_provider("plugin:").unwrap_err().to_string();
        assert!(err.contains("配置文件路径"), "{err}");

        assert!(is_npu_provider("plugin:/data/local/tmp/qnn.conf"));
        assert_eq!(
            plugin_config("plugin:/data/local/tmp/qnn.conf"),
            Some("/data/local/tmp/qnn.conf")
        );
        assert_eq!(plugin_config("plugin"), None);
        assert_eq!(plugin_config("qnn"), None);
    }

    #[test]
    fn report_recommends_qnn_only_when_runtime_and_ort_ep_agree() {
        let hit = |name: &str, path: &str| {
            Some(LibHit {
                name: name.into(),
                path: path.into(),
            })
        };
        let ort = |eps: &[&str]| {
            Some(OrtLib {
                path: "/data/app/lib/libonnxruntime.so".into(),
                eps: eps.iter().map(|ep| ep.to_string()).collect(),
            })
        };

        let mut report = NpuReport {
            ort_linked: true,
            ..Default::default()
        };
        assert_eq!(report.recommended(), None);
        report.qnn_htp = hit("libQnnHtp.so", "/system_ext/lib64/litert/libQnnHtp.so");
        assert_eq!(report.recommended(), None, "缺 libQnnSystem.so 不算齐");
        report.qnn_system = hit(
            "libQnnSystem.so",
            "/system_ext/lib64/litert/libQnnSystem.so",
        );
        // 没找到 ORT 本体 → 无从判断，按"有戏"给 qnn 加提示
        assert_eq!(report.recommended(), Some("qnn"));
        // 运行时齐了，但这套 ORT 只编了 CPU/NNAPI → 别骗用户
        report.ort = ort(&["CPU", "Nnapi"]);
        assert_eq!(report.recommended(), None);
        report.ort = ort(&["CPU", "QNN"]);
        assert_eq!(report.recommended(), Some("qnn"), "QNN 优先于 NNAPI");

        // NNAPI 兜底：库在 + ORT 编了 Nnapi
        let mut nnapi_only = NpuReport {
            ort_linked: true,
            ..Default::default()
        };
        nnapi_only.nnapi = hit("libneuralnetworks.so", "/system/lib64/libneuralnetworks.so");
        nnapi_only.ort = ort(&["CPU", "Nnapi"]);
        assert_eq!(nnapi_only.recommended(), Some("nnapi"));

        // 库在但 ORT 没有 NNAPI → 也不是可用方案
        let mut dead_nnapi = NpuReport {
            ort_linked: true,
            ..Default::default()
        };
        dead_nnapi.nnapi = hit("libneuralnetworks.so", "/system/lib64/libneuralnetworks.so");
        dead_nnapi.ort = ort(&["CPU"]);
        assert_eq!(dead_nnapi.recommended(), None);

        // 本构建没链接 ONNX Runtime：环境再齐全也没用，别给建议
        let mut no_ort_build = NpuReport::default();
        no_ort_build.qnn_htp = hit("libQnnHtp.so", "/system_ext/lib64/litert/libQnnHtp.so");
        no_ort_build.qnn_system = hit(
            "libQnnSystem.so",
            "/system_ext/lib64/litert/libQnnSystem.so",
        );
        no_ort_build.ort = ort(&["CPU", "QNN"]);
        assert_eq!(no_ort_build.recommended(), None);
    }

    #[test]
    #[cfg(target_os = "android")]
    fn scan_ort_eps_reads_exported_ep_symbols() {
        let path = std::env::temp_dir().join(format!("aha-ort-{}.bin", std::process::id()));
        let mut bytes = b"\x7fELF junk OrtSessionOptionsAppendExecutionProvider_CPU\x00".to_vec();
        bytes.extend_from_slice(
            b"....OrtSessionOptionsAppendExecutionProvider_Nnapi@@VERS_1.28.2\x00",
        );
        bytes.extend_from_slice(b"OrtSessionOptionsAppendExecutionProvider_QNN\x00");
        std::fs::write(&path, &bytes).unwrap();
        let eps = scan_ort_eps(&path);
        let _ = std::fs::remove_file(&path);
        assert!(eps.iter().any(|ep| ep == "CPU"), "{eps:?}");
        assert!(eps.iter().any(|ep| ep == "Nnapi"), "{eps:?}");
        assert!(eps.iter().any(|ep| ep == "QNN"), "{eps:?}");
    }

    #[test]
    fn json_report_carries_the_essentials() {
        let mut report = NpuReport {
            os: "android".into(),
            arch: "aarch64".into(),
            htp_arch: Some(81),
            ort_linked: true,
            ..Default::default()
        };
        report.qnn_htp = Some(LibHit {
            name: "libQnnHtp.so".into(),
            path: "/system_ext/lib64/litert/libQnnHtp.so".into(),
        });
        report.qnn_system = Some(LibHit {
            name: "libQnnSystem.so".into(),
            path: "/system_ext/lib64/litert/libQnnSystem.so".into(),
        });
        report.ort = Some(OrtLib {
            path: "/data/app/lib/libonnxruntime.so".into(),
            eps: vec!["CPU".into(), "QNN".into()],
        });
        let json = report.to_json();
        assert_eq!(json["recommended_provider"], "qnn");
        assert_eq!(json["qnn"]["htp_arch"], 81);
        assert_eq!(
            json["qnn"]["htp"]["path"],
            "/system_ext/lib64/litert/libQnnHtp.so"
        );
        assert_eq!(json["ort"]["eps"][1], "QNN");
    }

    #[test]
    fn qnn_sdk_reports_what_is_missing() {
        let mut sdk = QnnSdk::default();
        assert!(!sdk.is_complete());
        assert_eq!(
            sdk.missing(),
            vec![
                "include/QNN",
                "bin/<host>/qnn-onnx-converter",
                "lib/aarch64-android"
            ]
        );
        assert!(!sdk.has_jni());

        sdk.include = Some("/sdk/include/QNN".into());
        sdk.converter = Some("/sdk/bin/x86_64-linux-clang/qnn-onnx-converter".into());
        assert!(!sdk.is_complete(), "少了转换后的运行库还不能用");
        assert_eq!(sdk.missing(), vec!["lib/aarch64-android"]);

        sdk.android_runtime = Some("/sdk/lib/aarch64-android".into());
        assert!(sdk.is_complete());
        assert!(sdk.missing().is_empty());
        assert!(!sdk.has_jni(), "jni 只影响编模型库，不参与完整性判定");

        sdk.jni = Some("/sdk/share/QNN/converter/jni".into());
        assert!(sdk.is_complete());
        assert!(sdk.has_jni());
    }

    #[test]
    fn sdk_section_shows_up_in_json_and_render() {
        let mut report = NpuReport {
            os: "android".into(),
            arch: "aarch64".into(),
            ..Default::default()
        };
        report.sdk = Some(QnnSdk {
            root: Some("/sdk".into()),
            include: Some("/sdk/include/QNN".into()),
            jni: None,
            converter: Some("/sdk/bin/x86_64-linux-clang/qnn-onnx-converter".into()),
            android_runtime: Some("/sdk/lib/aarch64-android".into()),
        });
        let json = report.to_json();
        assert_eq!(json["qnn_sdk"]["complete"], true);
        assert_eq!(json["qnn_sdk"]["root"], "/sdk");
        assert!(report.render().contains("QAIRT SDK"));
    }
}

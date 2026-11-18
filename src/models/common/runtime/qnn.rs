//! 模型无关的 QNN 运行时：加载一个 QAIRT *context binary*，按图名/张量名跑推理。
//!
//! 这里刻意不认识任何具体模型。图的输入输出名字、类型、维度全部来自 context
//! binary 自带的元数据，所以同一套代码可以跑 SenseVoice、SoVITS，或者以后任何用
//! `scripts/qnn_convert.py` 转出来的模型。
//!
//! QNN 的 C 结构体布局很重（`QnnInterface_t`、`QnnSystemContext_GraphInfo_t`），
//! 在 Rust 里手抄一遍必错，所以真正的 FFI 放在 `native/qnn/aha_qnn.{h,c}` 里，由
//! 编译器替我们保证布局。本模块只 `dlopen` 那个薄垫片：垫片不在时功能不可用，
//! 但 aha 本身照常编译运行（默认桌面构建不需要 QNN）。

use anyhow::{Context, Result, bail};
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::{Path, PathBuf};

/// 与 `native/qnn/aha_qnn.h` 的 `AHA_QNN_MAX_DIMS` / `AHA_QNN_MAX_NAME` 对齐。
const MAX_DIMS: usize = 8;
const MAX_NAME: usize = 128;
/// 垫片名字：Android 上随 aha 一起打包，桌面构建没有它就只是能力缺失。
const SHIM_NAME: &str = "libaha_qnn.so";
/// `AHA_QNN_LIB` 可以指定垫片路径，方便测试和非常规部署。
const SHIM_ENV: &str = "AHA_QNN_LIB";

/// 与 `AhaQnnTensor` 逐字段对应（`#[repr(C)]`，字段顺序不能改）。
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct CTensor {
    name: [c_char; MAX_NAME],
    /// QNN 执行时按 id 匹配张量，名字只是给人看的。
    id: i32,
    data_type: i32,
    rank: i32,
    dims: [i32; MAX_DIMS],
    elements: i64,
    data: *mut c_void,
}

/// 一个图上的张量，名字与形状都来自 context binary。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// QNN 的 `Qnn_Tensor_t.id`；执行时靠它匹配，不能省。
    pub id: i32,
    /// QNN 的 `Qnn_DataType_t` 原值；本模块不解释它，只用来算字节数。
    pub data_type: i32,
    pub dims: Vec<i32>,
    pub elements: i64,
}

/// 一张图（转换器通常只产出一张）。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GraphInfo {
    pub name: String,
    pub inputs: Vec<TensorInfo>,
    pub outputs: Vec<TensorInfo>,
}

type OpenFn = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const c_char,
    *mut c_char,
    c_int,
) -> *mut c_void;
type CloseFn = unsafe extern "C" fn(*mut c_void);
type OpenModelLibFn =
    unsafe extern "C" fn(*const c_char, *const c_char, *mut c_char, c_int) -> *mut c_void;
type GraphCountFn = unsafe extern "C" fn(*mut c_void) -> c_int;
type GraphNameFn = unsafe extern "C" fn(*mut c_void, c_int) -> *const c_char;
type TensorCountFn = unsafe extern "C" fn(*mut c_void, c_int, c_int) -> c_int;
type TensorDescFn = unsafe extern "C" fn(*mut c_void, c_int, c_int, c_int, *mut CTensor) -> c_int;
type BytesPerElementFn = unsafe extern "C" fn(i32) -> usize;
type ExecuteFn = unsafe extern "C" fn(
    *mut c_void,
    c_int,
    *const CTensor,
    c_int,
    *mut CTensor,
    c_int,
    *mut c_char,
    c_int,
) -> c_int;

/// 垫片里的函数指针集合；全是 `Copy`，`Runtime` 直接拿一份即可。
#[derive(Clone, Copy)]
struct Fns {
    close: CloseFn,
    open_model_lib: OpenModelLibFn,
    tensor_count: TensorCountFn,
    tensor_desc: TensorDescFn,
    bytes_per_element: BytesPerElementFn,
    execute: ExecuteFn,
}

/// 一个已打开的 context binary。
///
/// 图的元数据在 `open` 时就全部读出来，`run` 只负责把缓冲区接上去，所以调用方
/// 不需要碰任何 QNN 类型。
pub struct Runtime {
    session: *mut c_void,
    fns: Fns,
    graphs: Vec<GraphInfo>,
    /// 串行化 QNN 调用：同一个 session 上的 graphExecute 不是并发安全的。
    lock: std::sync::Mutex<()>,
    /// 保持垫片处于已加载状态，Drop 时最后卸载。
    _shim: Shim,
    /// 产物路径（模型库或 context binary）。进程内执行被 HTP 拒绝时要靠它回退到
    /// SDK 自带的 `qnn-net-run`。
    artifact: PathBuf,
    backend_lib: String,
    /// 产物是模型库（`--model`）还是 context binary（`--retrieve_context`）。
    model_lib: bool,
    /// 这个产物在进程内能不能执行：被 HTP 拒过一次（1100）之后就不再试，直接走回退。
    in_process_usable: std::sync::atomic::AtomicBool,
}

// 安全理由：`session` 是 QNN context 句柄，不绑定线程；所有跨线程访问都经 `lock`
// 串行化，`fns` 只是函数指针表（Copy），`graphs` 是不可变的元数据。模型实例要放进
// 异步服务里，所以必须能 Send/Sync。
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

impl Runtime {
    /// 打开一个 context binary。
    ///
    /// `backend_lib` / `system_lib` 一般是 `libQnnHtp.so` / `libQnnSystem.so`（Android
    /// 上建议传绝对路径，例如 `/system_ext/lib64/litert/libQnnHtp.so`）。
    pub fn open(context_bin: &Path, backend_lib: &str, system_lib: &str) -> Result<Self> {
        let shim = Shim::open()?;
        let session = shim.open_session(context_bin, backend_lib, system_lib)?;
        Self::from_session(shim, session, context_bin, backend_lib, false)
    }

    /// 打开一个模型库（`qnn-model-lib-generator` 产出的 `lib<name>.so`）。
    ///
    /// 模型库自己 compose 图并把张量描述符交回来，所以这条路径对"context binary
    /// 执行时报 DMA 错误"的 float32 IO 图也能用（HTP v81 上实测）。
    pub fn open_model_lib(model_lib: &Path, backend_lib: &str) -> Result<Self> {
        let shim = Shim::open()?;
        let session = shim.open_model_session(model_lib, backend_lib)?;
        Self::from_session(shim, session, model_lib, backend_lib, true)
    }

    fn from_session(
        shim: Shim,
        session: *mut c_void,
        artifact: &Path,
        backend_lib: &str,
        model_lib: bool,
    ) -> Result<Self> {
        // 元数据里的名字由垫片读出来，借用 session 生命周期，因此一次性拷成 Rust
        // 字符串，之后 close 也不影响。
        let mut graphs = Vec::new();
        let count = unsafe { (shim.graph_count)(session) };
        if count < 0 {
            unsafe { (shim.fns.close)(session) };
            bail!("QNN session 没有报告任何图");
        }
        for g in 0..count {
            let name = unsafe {
                let p = (shim.graph_name)(session, g);
                if p.is_null() {
                    String::new()
                } else {
                    CStr::from_ptr(p).to_string_lossy().into_owned()
                }
            };
            let inputs = unsafe { read_tensors(&shim.fns, session, g, 1) }?;
            let outputs = unsafe { read_tensors(&shim.fns, session, g, 0) }?;
            graphs.push(GraphInfo {
                name,
                inputs,
                outputs,
            });
        }
        Ok(Self {
            session,
            fns: shim.fns,
            graphs,
            lock: std::sync::Mutex::new(()),
            _shim: shim,
            artifact: artifact.to_path_buf(),
            backend_lib: backend_lib.to_string(),
            model_lib,
            in_process_usable: std::sync::atomic::AtomicBool::new(true),
        })
    }

    pub fn graphs(&self) -> &[GraphInfo] {
        &self.graphs
    }

    /// 每个元素多少字节。交给垫片判断，Rust 侧不复制一遍 QNN 的枚举。
    pub fn bytes_per_element(&self, data_type: i32) -> usize {
        unsafe { (self.fns.bytes_per_element)(data_type) }
    }

    /// 跑第 `graph` 张图。`inputs` / `outputs` 必须与 `graphs()[graph]` 的顺序、
    /// 元素个数一一对应；缓冲区长度不对时直接报错，不猜。
    pub fn run(&self, graph: usize, inputs: &mut [Vec<u8>], outputs: &mut [Vec<u8>]) -> Result<()> {
        let g = self
            .graphs
            .get(graph)
            .with_context(|| format!("没有第 {graph} 张图"))?;
        if inputs.len() != g.inputs.len() {
            bail!(
                "图 {} 需要 {} 个输入，给了 {}",
                g.name,
                g.inputs.len(),
                inputs.len()
            );
        }
        if outputs.len() != g.outputs.len() {
            bail!(
                "图 {} 需要 {} 个输出，给了 {}",
                g.name,
                g.outputs.len(),
                outputs.len()
            );
        }
        // 这个产物在进程内跑不了时（HTP 拒绝 IO 的 DMA），记住结论：后面的调用直接
        // 走回退，省掉每次好几百毫秒的注定失败的尝试。`AHA_QNN_SKIP_INPROCESS=1`
        // 可以不用先失败一次就跳过。
        let skip_requested = matches!(std::env::var(SKIP_INPROCESS_ENV).ok().as_deref(), Some("1"));
        let fallback_enabled = std::env::var(NETRUN_ENV).ok().as_deref() != Some("0");
        let try_in_process = !skip_requested
            && self
                .in_process_usable
                .load(std::sync::atomic::Ordering::Relaxed);
        if !try_in_process {
            if !fallback_enabled {
                bail!("进程内执行已跳过（{SKIP_INPROCESS_ENV}=1），而 {NETRUN_ENV}=0 关掉了回退");
            }
            return self.run_via_net_run(g, inputs, outputs);
        }
        let msg = {
            // 同一个 session 不能被并发 graphExecute，拿锁串行化；中毒时沿用内部值，
            // 因为缓冲区已经校验过，属于纯数据。
            let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            let mut c_in = build_ctensors(&self.fns, &g.inputs, inputs)?;
            let mut c_out = build_ctensors(&self.fns, &g.outputs, outputs)?;
            let mut err = [0 as c_char; 512];
            // SAFETY: 描述符与缓冲区都在本次调用期间有效，垫片只读描述符、读写缓冲区。
            let rc = unsafe {
                (self.fns.execute)(
                    self.session,
                    graph as c_int,
                    c_in.as_mut_ptr(),
                    c_in.len() as c_int,
                    c_out.as_mut_ptr(),
                    c_out.len() as c_int,
                    err.as_mut_ptr(),
                    err.len() as c_int,
                )
            };
            if rc == 0 {
                return Ok(());
            }
            c_message(&err)
        };
        // HTP v81 会拒绝某些图的 IO DMA（`Qnn_Common_Error_General (1100)`），而同一
        // 份产物交给 SDK 自带的 `qnn-net-run` 却能跑。回退到它，至少让产物可用：
        // 写同样的 raw 文件和一个 input list，跑完把输出读回来。
        if !msg.contains("1100") || !fallback_enabled {
            bail!("QNN 执行失败：{msg}");
        }
        self.in_process_usable
            .store(false, std::sync::atomic::Ordering::Relaxed);
        match self.run_via_net_run(g, inputs, outputs) {
            Ok(()) => Ok(()),
            Err(fallback) => bail!("QNN 执行失败：{msg}（回退 qnn-net-run 也失败：{fallback:#}）"),
        }
    }

    /// 用 SDK 自带的 `qnn-net-run` 跑一次推理（子进程），结果写回 `outputs`。
    ///
    /// 只在进程内 `graphExecute` 被 DSP 拒绝时才走这里。`qnn-net-run` 与产物同在
    /// 一个目录（`scripts/qnn_convert.py --stage` 就会把它放进去），工作目录也设在
    /// 那里，这样模型库的权重和 QNN 运行库都能按相对路径解析。
    fn run_via_net_run(
        &self,
        g: &GraphInfo,
        inputs: &[Vec<u8>],
        outputs: &mut [Vec<u8>],
    ) -> Result<()> {
        if self.graphs.len() != 1 {
            bail!("回退只支持单图产物，这个产物有 {} 张图", self.graphs.len());
        }
        let dir = self
            .artifact
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf();
        let runner = net_run_binary(&dir)?;
        let artifact_name = self
            .artifact
            .file_name()
            .and_then(|n| n.to_str())
            .with_context(|| format!("产物路径没有文件名：{}", self.artifact.display()))?;
        // 工作目录用相对路径：net-run 的 input_list 与 --output_dir 都按 cwd 解析。
        let scratch_name = format!(".aha-netrun-{}", std::process::id());
        let scratch = dir.join(&scratch_name);
        let _ = std::fs::remove_dir_all(&scratch);
        std::fs::create_dir_all(&scratch)
            .with_context(|| format!("建不了工作目录 {}（产物目录不可写？）", scratch.display()))?;

        let mut list_line = String::new();
        for (info, buf) in g.inputs.iter().zip(inputs.iter()) {
            let want = self.expect_bytes(info)?;
            if buf.len() != want {
                let _ = std::fs::remove_dir_all(&scratch);
                bail!(
                    "输入 {} 长度 {} 与图要求的 {want} 不符",
                    info.name,
                    buf.len()
                );
            }
            let file = format!("{}.raw", file_stem(&info.name));
            std::fs::write(scratch.join(&file), buf)
                .with_context(|| format!("写不了输入文件 {file}"))?;
            if !list_line.is_empty() {
                list_line.push(' ');
            }
            list_line.push_str(&format!("./{scratch_name}/{file}"));
        }
        std::fs::write(scratch.join("input_list.txt"), format!("{list_line}\n"))
            .context("写不了 input_list.txt")?;

        let mut cmd = std::process::Command::new(&runner);
        cmd.current_dir(&dir);
        for arg in net_run_args(
            &self.backend_lib,
            artifact_name,
            self.model_lib,
            &scratch_name,
        ) {
            cmd.arg(arg);
        }
        // 设备上的 QNN 运行库通常和产物放在一起，子进程也要能找到。
        for key in ["LD_LIBRARY_PATH", "ADSP_LIBRARY_PATH"] {
            let mut paths = vec![dir.display().to_string()];
            if let Some(old) = std::env::var_os(key) {
                if !old.is_empty() {
                    paths.push(old.to_string_lossy().into_owned());
                }
            }
            cmd.env(key, paths.join(":"));
        }
        let output = cmd
            .output()
            .with_context(|| format!("跑不起来 {}", runner.display()))?;
        if !output.status.success() {
            let log = tail(&String::from_utf8_lossy(&output.stdout), 12);
            let err = tail(&String::from_utf8_lossy(&output.stderr), 12);
            let _ = std::fs::remove_dir_all(&scratch);
            bail!(
                "qnn-net-run 退出码 {:?}\n--- stdout ---\n{log}\n--- stderr ---\n{err}",
                output.status.code()
            );
        }
        for (info, buf) in g.outputs.iter().zip(outputs.iter_mut()) {
            let file = scratch
                .join("out")
                .join("Result_0")
                .join(format!("{}.raw", file_stem(&info.name)));
            let data = std::fs::read(&file)
                .with_context(|| format!("qnn-net-run 没产出 {}", file.display()))?;
            let want = self.expect_bytes(info)?;
            if data.len() < want {
                let _ = std::fs::remove_dir_all(&scratch);
                bail!("输出 {} 只有 {} 字节，图要求 {want}", info.name, data.len());
            }
            buf.clear();
            buf.extend_from_slice(&data[..want]);
        }
        let _ = std::fs::remove_dir_all(&scratch);
        Ok(())
    }

    /// 一个张量在图里占多少字节。
    fn expect_bytes(&self, info: &TensorInfo) -> Result<usize> {
        let width = self.bytes_per_element(info.data_type);
        if width == 0 {
            bail!("张量 {} 的类型 {} 不认识", info.name, info.data_type);
        }
        Ok(width * usize::try_from(info.elements.max(0)).unwrap_or(0))
    }
}

impl Drop for Runtime {
    fn drop(&mut self) {
        // SAFETY: session 只在这里释放一次，之后不再使用。
        unsafe { (self.fns.close)(self.session) };
        self.session = std::ptr::null_mut();
    }
}

/// 子进程回退开关：`AHA_QNN_NETRUN=0` 只用进程内执行，`AHA_QNN_NETRUN=<路径>`
/// 指定运行器位置。
const NETRUN_ENV: &str = "AHA_QNN_NETRUN";
/// 直接跳过进程内执行：设成 1 时第一次就交给 `qnn-net-run`。
const SKIP_INPROCESS_ENV: &str = "AHA_QNN_SKIP_INPROCESS";
/// SDK 自带的运行器，`scripts/qnn_convert.py --stage` 会把它放在产物目录里。
const NETRUN_BIN: &str = "qnn-net-run";

/// 找 `qnn-net-run`：`AHA_QNN_NETRUN` 给了路径就用它，否则看产物目录。
fn net_run_binary(dir: &Path) -> Result<PathBuf> {
    if let Ok(custom) = std::env::var(NETRUN_ENV) {
        if !custom.is_empty() && custom != "0" && custom != "1" {
            let path = PathBuf::from(&custom);
            if path.is_file() {
                return Ok(path);
            }
            bail!("{NETRUN_ENV} 指向的文件不存在：{}", path.display());
        }
    }
    let candidate = dir.join(NETRUN_BIN);
    if candidate.is_file() {
        return Ok(candidate);
    }
    bail!("产物目录里没有 {NETRUN_BIN}（`qnn_convert.py --stage` 会带上它）")
}

/// `qnn-net-run` 的命令行：模型库走 `--model`，context binary 走 `--retrieve_context`。
/// 输入清单与输出目录都用相对 cwd 的路径，因为子进程的工作目录就是产物目录。
fn net_run_args(
    backend_lib: &str,
    artifact_name: &str,
    model_lib: bool,
    scratch_name: &str,
) -> Vec<String> {
    vec![
        "--backend".to_string(),
        backend_lib.to_string(),
        if model_lib {
            "--model"
        } else {
            "--retrieve_context"
        }
        .to_string(),
        artifact_name.to_string(),
        "--input_list".to_string(),
        format!("./{scratch_name}/input_list.txt"),
        "--output_dir".to_string(),
        format!("./{scratch_name}/out"),
    ]
}

/// 张量名当文件名用：只保留安全字符。
fn file_stem(name: &str) -> String {
    name.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// 摘日志尾部若干行：报错时够定位，又不会把整份输出刷进错误消息。
fn tail(text: &str, lines: usize) -> String {
    let all: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    all[all.len().saturating_sub(lines)..].join("\n")
}

fn build_ctensors(
    fns: &Fns,
    infos: &[TensorInfo],
    buffers: &mut [Vec<u8>],
) -> Result<Vec<CTensor>> {
    let mut out = Vec::with_capacity(infos.len());
    for (i, info) in infos.iter().enumerate() {
        let bytes = unsafe { (fns.bytes_per_element)(info.data_type) };
        if bytes == 0 {
            bail!("张量 {} 的类型 {} 不认识", info.name, info.data_type);
        }
        let want = bytes * usize::try_from(info.elements.max(0)).unwrap_or(0);
        let buf = &mut buffers[i];
        if buf.len() != want {
            bail!(
                "张量 {} 需要 {want} 字节（{} 个元素 × {bytes}），给了 {}",
                info.name,
                info.elements,
                buf.len()
            );
        }
        let mut name = [0 as c_char; MAX_NAME];
        for (dst, src) in name.iter_mut().zip(info.name.as_bytes().iter()) {
            *dst = *src as c_char;
        }
        let mut dims = [0i32; MAX_DIMS];
        for (dst, src) in dims.iter_mut().zip(info.dims.iter()) {
            *dst = *src;
        }
        out.push(CTensor {
            name,
            id: info.id,
            data_type: info.data_type,
            rank: (info.dims.len().min(MAX_DIMS)) as i32,
            dims,
            elements: info.elements,
            data: buf.as_mut_ptr().cast::<c_void>(),
        });
    }
    Ok(out)
}

unsafe fn read_tensors(
    fns: &Fns,
    session: *mut c_void,
    graph: c_int,
    is_input: c_int,
) -> Result<Vec<TensorInfo>> {
    let mut out = Vec::new();
    // SAFETY: 调用方保证 session 有效；索引由垫片自己校验。
    unsafe {
        let count = (fns.tensor_count)(session, graph, is_input);
        if count < 0 {
            return Ok(out);
        }
        for i in 0..count {
            let mut t = CTensor {
                name: [0; MAX_NAME],
                id: 0,
                data_type: 0,
                rank: 0,
                dims: [0; MAX_DIMS],
                elements: 0,
                data: std::ptr::null_mut(),
            };
            if (fns.tensor_desc)(session, graph, is_input, i, &mut t) != 0 {
                continue;
            }
            let name = CStr::from_ptr(t.name.as_ptr())
                .to_string_lossy()
                .into_owned();
            let rank = t.rank.clamp(0, MAX_DIMS as i32) as usize;
            out.push(TensorInfo {
                name,
                id: t.id,
                data_type: t.data_type,
                dims: t.dims[..rank].to_vec(),
                elements: t.elements,
            });
        }
    }
    Ok(out)
}

fn c_message(buf: &[c_char]) -> String {
    // SAFETY: 缓冲区由本模块分配，垫片保证以 NUL 结尾。
    unsafe { CStr::from_ptr(buf.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

/// 垫片句柄：`dlopen` 一次，`Runtime` 持有它，Drop 时 `dlclose`。
struct Shim {
    lib: *mut c_void,
    open: OpenFn,
    graph_count: GraphCountFn,
    graph_name: GraphNameFn,
    fns: Fns,
}

impl Shim {
    fn open() -> Result<Self> {
        let path = shim_path().context(
            "找不到 QNN 垫片 libaha_qnn.so（native/qnn 的编译产物），可用 AHA_QNN_LIB 指定路径",
        )?;
        let lib = unsafe { dlopen_lib(&path) }?;
        let open = unsafe { dlsym_fn::<OpenFn>(lib, "aha_qnn_open") }
            .with_context(|| format!("{} 缺少 aha_qnn_open", path.display()))?;
        let graph_count = unsafe { dlsym_fn::<GraphCountFn>(lib, "aha_qnn_graph_count") }
            .with_context(|| format!("{} 缺少 aha_qnn_graph_count", path.display()))?;
        let graph_name = unsafe { dlsym_fn::<GraphNameFn>(lib, "aha_qnn_graph_name") }
            .with_context(|| format!("{} 缺少 aha_qnn_graph_name", path.display()))?;
        let fns = Fns {
            close: unsafe { dlsym_fn::<CloseFn>(lib, "aha_qnn_close") }
                .with_context(|| format!("{} 缺少 aha_qnn_close", path.display()))?,
            tensor_count: unsafe { dlsym_fn::<TensorCountFn>(lib, "aha_qnn_tensor_count") }
                .with_context(|| format!("{} 缺少 aha_qnn_tensor_count", path.display()))?,
            tensor_desc: unsafe { dlsym_fn::<TensorDescFn>(lib, "aha_qnn_tensor_desc") }
                .with_context(|| format!("{} 缺少 aha_qnn_tensor_desc", path.display()))?,
            bytes_per_element: unsafe {
                dlsym_fn::<BytesPerElementFn>(lib, "aha_qnn_bytes_per_element")
            }
            .with_context(|| format!("{} 缺少 aha_qnn_bytes_per_element", path.display()))?,
            execute: unsafe { dlsym_fn::<ExecuteFn>(lib, "aha_qnn_execute") }
                .with_context(|| format!("{} 缺少 aha_qnn_execute", path.display()))?,
            open_model_lib: unsafe { dlsym_fn::<OpenModelLibFn>(lib, "aha_qnn_open_model_lib") }
                .with_context(|| format!("{} 缺少 aha_qnn_open_model_lib", path.display()))?,
        };
        Ok(Self {
            lib,
            open,
            graph_count,
            graph_name,
            fns,
        })
    }

    fn open_session(
        &self,
        context_bin: &Path,
        backend_lib: &str,
        system_lib: &str,
    ) -> Result<*mut c_void> {
        let context = CString::new(context_bin.to_string_lossy().as_bytes())
            .with_context(|| format!("context binary 路径里有 NUL：{}", context_bin.display()))?;
        let backend = CString::new(backend_lib)?;
        let system = CString::new(system_lib)?;
        let mut err = [0 as c_char; 512];
        // SAFETY: 三个 C 字符串在本调用期间有效。
        let session = unsafe {
            (self.open)(
                backend.as_ptr(),
                system.as_ptr(),
                context.as_ptr(),
                err.as_mut_ptr(),
                err.len() as c_int,
            )
        };
        if session.is_null() {
            bail!("加载 context binary 失败：{}", c_message(&err));
        }
        Ok(session)
    }

    fn open_model_session(&self, model_lib: &Path, backend_lib: &str) -> Result<*mut c_void> {
        let model = CString::new(model_lib.to_string_lossy().as_bytes())
            .with_context(|| format!("模型库路径里有 NUL：{}", model_lib.display()))?;
        let backend = CString::new(backend_lib)?;
        let mut err = [0 as c_char; 512];
        // SAFETY: 两个 C 字符串在本调用期间有效。
        let session = unsafe {
            (self.fns.open_model_lib)(
                backend.as_ptr(),
                model.as_ptr(),
                err.as_mut_ptr(),
                err.len() as c_int,
            )
        };
        if session.is_null() {
            bail!("加载模型库失败：{}", c_message(&err));
        }
        Ok(session)
    }
}

impl Drop for Shim {
    fn drop(&mut self) {
        // SAFETY: 句柄来自 dlopen，且 Runtime 已经保证 session 先释放。
        unsafe { dlclose_lib(self.lib) };
        self.lib = std::ptr::null_mut();
    }
}

/// 垫片位置：`AHA_QNN_LIB` → 可执行文件同目录 → 当前目录。
pub fn shim_path() -> Option<PathBuf> {
    if let Ok(p) = std::env::var(SHIM_ENV) {
        let p = PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(dir) = exe.parent()
    {
        let p = dir.join(SHIM_NAME);
        if p.is_file() {
            return Some(p);
        }
    }
    let p = PathBuf::from(SHIM_NAME);
    if p.is_file() { Some(p) } else { None }
}

#[cfg(unix)]
unsafe fn dlopen_lib(path: &Path) -> Result<*mut c_void> {
    let c = CString::new(path.to_string_lossy().as_bytes())?;
    // SAFETY: NUL 结尾的路径字符串。
    let handle = unsafe { libc::dlopen(c.as_ptr(), libc::RTLD_NOW | libc::RTLD_LOCAL) };
    if handle.is_null() {
        bail!("dlopen {} 失败：{}", path.display(), dlerror_message());
    }
    Ok(handle)
}

#[cfg(unix)]
unsafe fn dlclose_lib(handle: *mut c_void) {
    // SAFETY: 句柄由 dlopen 得到，只会关一次。
    unsafe { libc::dlclose(handle) };
}

#[cfg(unix)]
unsafe fn dlsym_fn<T: Copy>(lib: *mut c_void, name: &str) -> Result<T> {
    let c = CString::new(name)?;
    // SAFETY: 传入的是有效句柄与 NUL 结尾的符号名。
    let sym = unsafe { libc::dlsym(lib, c.as_ptr()) };
    if sym.is_null() {
        bail!("符号 {name} 不存在");
    }
    // SAFETY: 调用方用正确的函数指针类型实例化 T。
    Ok(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&sym) })
}

#[cfg(unix)]
fn dlerror_message() -> String {
    // SAFETY: dlerror 返回线程局部的静态字符串。
    let p = unsafe { libc::dlerror() };
    if p.is_null() {
        String::from("未知错误")
    } else {
        // SAFETY: dlerror 保证返回 NUL 结尾字符串。
        unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
    }
}

#[cfg(not(unix))]
unsafe fn dlopen_lib(_path: &Path) -> Result<*mut c_void> {
    bail!("QNN 运行时目前只接线到 Android/Linux（垫片依赖 dlopen）")
}

#[cfg(not(unix))]
unsafe fn dlclose_lib(_handle: *mut c_void) {}

#[cfg(not(unix))]
unsafe fn dlsym_fn<T: Copy>(_lib: *mut c_void, _name: &str) -> Result<T> {
    bail!("QNN 运行时目前只接线到 Android/Linux（垫片依赖 dlopen）")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn net_run_args_pick_model_or_context() {
        let lib = net_run_args("libQnnHtp.so", "libvitsF.so", true, ".aha-netrun-1");
        assert_eq!(lib[0], "--backend");
        assert_eq!(lib[1], "libQnnHtp.so");
        assert_eq!(lib[2], "--model");
        assert_eq!(lib[3], "libvitsF.so");
        assert_eq!(lib[4], "--input_list");
        assert_eq!(lib[5], "./.aha-netrun-1/input_list.txt");
        assert_eq!(lib[6], "--output_dir");
        assert_eq!(lib[7], "./.aha-netrun-1/out");

        let bin = net_run_args("libQnnHtp.so", "vitsS_ctx.bin", false, ".aha-netrun-2");
        assert_eq!(bin[2], "--retrieve_context");
        assert_eq!(bin[3], "vitsS_ctx.bin");
    }

    #[test]
    fn tensor_names_become_safe_file_names() {
        assert_eq!(file_stem("audio"), "audio");
        assert_eq!(file_stem("ref_emb-1"), "ref_emb-1");
        assert_eq!(file_stem("a/b:c"), "a_b_c");
        assert_eq!(file_stem("注意力"), "___");
    }

    #[test]
    fn tail_keeps_the_last_lines() {
        assert_eq!(tail("a\n\nb\nc\nd\n", 2), "c\nd");
        assert_eq!(tail("only", 3), "only");
        assert_eq!(tail("", 3), "");
    }

    #[test]
    fn tensor_info_keeps_shape_and_element_count() {
        let info = TensorInfo {
            name: "codes".into(),
            id: 7,
            data_type: 0,
            dims: vec![1, 1, 64],
            elements: 64,
        };
        let mut buffers = vec![Vec::new()];
        // 未知类型 → 直接报错，不去猜字节数。
        let fns = Fns {
            close: noop_close,
            tensor_count: noop_count,
            tensor_desc: noop_desc,
            bytes_per_element: zero_bytes,
            execute: noop_execute,
            open_model_lib: noop_open_model_lib,
        };
        let err = build_ctensors(&fns, std::slice::from_ref(&info), &mut buffers).unwrap_err();
        assert!(err.to_string().contains("不认识"), "{err}");
    }

    #[test]
    fn ctensor_layout_matches_the_c_header() {
        // 名字 128 + id 4 + 类型 4 + rank 4 + 8 个维度 32 = 172，i64 要对齐到 8 所以补 4，
        // 再加元素数 8 与指针 8 → 192。C 那边同一份声明，规则相同。
        assert_eq!(std::mem::offset_of!(CTensor, name), 0);
        assert_eq!(std::mem::offset_of!(CTensor, id), 128);
        assert_eq!(std::mem::offset_of!(CTensor, data_type), 132);
        assert_eq!(std::mem::offset_of!(CTensor, rank), 136);
        assert_eq!(std::mem::offset_of!(CTensor, dims), 140);
        assert_eq!(std::mem::offset_of!(CTensor, elements), 176);
        assert_eq!(std::mem::offset_of!(CTensor, data), 184);
        assert_eq!(std::mem::size_of::<CTensor>(), 192);
    }

    #[test]
    fn buffers_must_match_declared_size() {
        let fns = Fns {
            close: noop_close,
            tensor_count: noop_count,
            tensor_desc: noop_desc,
            bytes_per_element: four_bytes,
            execute: noop_execute,
            open_model_lib: noop_open_model_lib,
        };
        let info = TensorInfo {
            name: "x".into(),
            id: 3,
            data_type: 0,
            dims: vec![2],
            elements: 2,
        };
        let mut wrong = vec![vec![0u8; 4]];
        let err = build_ctensors(&fns, std::slice::from_ref(&info), &mut wrong).unwrap_err();
        assert!(err.to_string().contains("需要 8 字节"), "{err}");
        let mut right = vec![vec![0u8; 8]];
        let built = build_ctensors(&fns, std::slice::from_ref(&info), &mut right).unwrap();
        assert_eq!(built.len(), 1);
        assert_eq!(built[0].rank, 1);
        assert_eq!(built[0].elements, 2);
    }

    unsafe extern "C" fn noop_close(_: *mut c_void) {}
    unsafe extern "C" fn noop_count(_: *mut c_void, _: c_int, _: c_int) -> c_int {
        0
    }
    unsafe extern "C" fn noop_desc(
        _: *mut c_void,
        _: c_int,
        _: c_int,
        _: c_int,
        _: *mut CTensor,
    ) -> c_int {
        -1
    }
    unsafe extern "C" fn zero_bytes(_: i32) -> usize {
        0
    }
    unsafe extern "C" fn four_bytes(_: i32) -> usize {
        4
    }
    unsafe extern "C" fn noop_open_model_lib(
        _: *const c_char,
        _: *const c_char,
        _: *mut c_char,
        _: c_int,
    ) -> *mut c_void {
        std::ptr::null_mut()
    }
    unsafe extern "C" fn noop_execute(
        _: *mut c_void,
        _: c_int,
        _: *const CTensor,
        _: c_int,
        _: *mut CTensor,
        _: c_int,
        _: *mut c_char,
        _: c_int,
    ) -> c_int {
        -1
    }
}

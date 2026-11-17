pub mod audio_utils;
pub mod http;
pub mod npu;

/// 解析 `--device`：`auto`（None，由模型自己决定）/ `cpu` / `cuda` / `npu`。
///
/// `npu` 是给手机准备的**组合快捷键**：candle 侧没有 NPU 后端，所以这里仍然返回
/// `None`（= CPU），真正的 NPU 加速由 ONNX Runtime 的 provider 完成——
/// CLI 那边看到 `--device npu` 会把它翻成 `--provider qnn`（见
/// [`crate::cli`] 的 `init_inference_backend`）。这样桌面端 `--device npu`
/// 不会因为 candle 不认识 npu 直接报错，模型也能照常跑。
pub fn resolve_device(spec: &str) -> anyhow::Result<Option<candle_core::Device>> {
    match spec.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => Ok(None),
        "cpu" => Ok(Some(candle_core::Device::Cpu)),
        // NPU：candle 不接，回落 CPU；ORT 侧走 provider（由上层设置）。
        "npu" | "npu-auto" => Ok(Some(candle_core::Device::Cpu)),
        "cuda" | "gpu" => {
            #[cfg(feature = "cuda")]
            {
                Ok(Some(candle_core::Device::new_cuda(0)?))
            }
            #[cfg(not(feature = "cuda"))]
            {
                anyhow::bail!("当前二进制没编译 cuda 支持，请用 --features cuda 重新构建")
            }
        }
        other => anyhow::bail!("未知推理设备 {other:?}（可选 auto / cpu / cuda / npu）"),
    }
}
pub mod img_utils;
pub mod interpolate;
pub mod response_utils;
pub mod tensor_utils;
pub mod video_utils;

use std::fs::File;
use std::io::{Cursor, Read, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::Duration,
};

use crate::models::common::model_mapping::WhichModel;
use crate::params::chat::{
    ChatCompletionParameters, ChatMessage, ChatMessageContent, ChatMessageContentPart,
};
use anyhow::{Result, anyhow};
use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{
    Context, DType, Device, Shape, Tensor,
    pickle::{Object, Stack, TensorInfo, read_all_with_key},
};
use candle_nn::VarBuilder;
use dirs::home_dir;
use futures_util::StreamExt;
use half::{bf16, f16, slice::HalfFloatSliceExt};
use tokio::time::sleep;
use zip::ZipArchive;

pub fn get_device(device: Option<&Device>) -> Device {
    match device {
        Some(d) => d.clone(),
        None => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).unwrap_or(Device::Cpu)
            }
            #[cfg(all(not(feature = "cuda"), feature = "metal"))]
            {
                Device::new_metal(0).unwrap_or(Device::Cpu)
            }
            #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
            {
                Device::Cpu
            }
        }
    }
}

pub fn get_gpu_sm_arch() -> Result<f32> {
    let output = Command::new("nvidia-smi")
        .arg("--query-gpu=compute_cap")
        .arg("--format=csv,noheader")
        .output()
        .map_err(|e| anyhow::anyhow!(format!("Failed to execute nvidia-smi: {}", e)))?;
    if !output.status.success() {
        return Err(anyhow::anyhow!(format!(
            "nvidia-smi failed with status: {}\nError: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        )));
    }
    let output_str = String::from_utf8_lossy(&output.stdout);
    let output_str = output_str.trim();
    let sm_float = match output_str.parse::<f32>() {
        Ok(num) => num,
        Err(_) => {
            return Err(anyhow::anyhow!(format!(
                "gpr sm arch: {} parse float32 error",
                output_str
            )));
        }
    };
    Ok(sm_float)
}

pub fn get_dtype(dtype: Option<DType>, cfg_dtype: &str) -> DType {
    match dtype {
        Some(d) => d,
        None => {
            #[cfg(feature = "cuda")]
            {
                match cfg_dtype {
                    "float32" | "float" => DType::F32,
                    "float64" | "double" => DType::F64,
                    "float16" => DType::F16,
                    "bfloat16" => {
                        let arch = get_gpu_sm_arch();
                        match arch {
                            Err(_) => DType::F16,
                            Ok(a) => {
                                // nvidia显卡sm架构>=8.0的才支持BF16
                                if a >= 8.0 { DType::BF16 } else { DType::F16 }
                            }
                        }
                    }
                    "uint8" => DType::U8,
                    "int8" | "int16" | "int32" | "int64" => DType::I64,
                    _ => DType::F32,
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                match cfg_dtype {
                    "float32" | "float" => DType::F32,
                    "float64" | "double" => DType::F64,
                    "float16" | "bfloat16" => DType::F16, // cpu上bfloat16有问题
                    "uint8" => DType::U8,
                    "int8" | "int16" | "int32" | "int64" => DType::I64,
                    _ => DType::F32,
                }
            }
        }
    }
}

pub fn string_to_static_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

pub fn find_type_files(path: &str, extension_type: &str) -> Result<Vec<String>> {
    let mut files = Vec::new();

    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        let file_path = entry.path();

        if file_path.is_file()
            && let Some(extension) = file_path.extension()
            && extension == extension_type
        {
            files.push(file_path.to_string_lossy().to_string());
        }
    }

    Ok(files)
}

pub fn get_vb_model_path(
    model_path: String,
    dtype: DType,
    device: Device,
    key: Option<&'_ str>,
) -> Result<VarBuilder<'_>> {
    let mut dict_to_hashmap = HashMap::new();
    let dict = read_all_with_key(&model_path, key)?;
    for (k, v) in dict {
        dict_to_hashmap.insert(k, v);
    }
    let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, &device);
    Ok(vb)
}

pub fn get_vb_extension(
    path: String,
    extension_type: String,
    dtype: DType,
    device: Device,
    key: Option<&'_ str>,
) -> Result<VarBuilder<'_>> {
    let model_list = find_type_files(&path, &extension_type)?;
    let mut dict_to_hashmap = HashMap::new();
    for m in model_list {
        let dict = read_all_with_key(m, key)?;
        for (k, v) in dict {
            dict_to_hashmap.insert(k, v);
        }
    }
    let vb = VarBuilder::from_tensors(dict_to_hashmap, dtype, &device);
    Ok(vb)
}

pub fn crate_tensor_from_reader<R: std::io::Read>(
    shape: Shape,
    dtype: DType,
    reader: &mut R,
) -> Result<Tensor> {
    let elem_count = shape.elem_count();
    match dtype {
        DType::BF16 => {
            let mut data_t = vec![bf16::ZERO; elem_count];
            reader.read_u16_into::<LittleEndian>(data_t.reinterpret_cast_mut())?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::F16 => {
            let mut data_t = vec![f16::ZERO; elem_count];
            reader.read_u16_into::<LittleEndian>(data_t.reinterpret_cast_mut())?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::F32 => {
            let mut data_t = vec![0f32; elem_count];
            reader.read_f32_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::F64 => {
            let mut data_t = vec![0f64; elem_count];
            reader.read_f64_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::U8 => {
            let mut data_t = vec![0u8; elem_count];
            reader.read_exact(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::U32 => {
            let mut data_t = vec![0u32; elem_count];
            reader.read_u32_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::I16 => {
            let mut data_t = vec![0i16; elem_count];
            reader.read_i16_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::I32 => {
            let mut data_t = vec![0i32; elem_count];
            reader.read_i32_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        DType::I64 => {
            let mut data_t = vec![0i64; elem_count];
            reader.read_i64_into::<LittleEndian>(&mut data_t)?;
            Ok(Tensor::from_vec(data_t, shape, &Device::Cpu)?)
        }
        // `DType` is `#[non_exhaustive]` since candle 0.11, so keep a catch-all arm.
        _ => Err(anyhow!(format!("UnsupportedDTypeForOp '{:?}'", dtype))),
    }
}

pub fn read_pth_tensor_info_cycle<P: AsRef<std::path::Path>>(
    path: P,
    key: Option<&str>,
) -> Result<HashMap<String, Tensor>> {
    let file = std::fs::File::open(path.as_ref())?;
    let zip_reader = std::io::BufReader::new(file);
    let mut zip = zip::ZipArchive::new(zip_reader)?;
    let zip_file_names = zip
        .file_names()
        .map(|f| f.to_string())
        .collect::<Vec<String>>();

    let mut tensor_infos = vec![];
    for file_name in zip_file_names.iter() {
        if !file_name.ends_with("data.pkl") {
            continue;
        }
        let dir_name = std::path::PathBuf::from(file_name.strip_suffix(".pkl").context("no .pkl")?);
        let reader = zip.by_name(file_name)?;
        let mut reader = std::io::BufReader::new(reader);
        let mut stack = Stack::empty();
        stack.read_loop(&mut reader)?;
        let obj = stack.finalize()?;

        let obj = match obj {
            Object::Build { callable, args } => match *callable {
                Object::Reduce { callable, args: _ } => match *callable {
                    Object::Class {
                        module_name,
                        class_name,
                    } if module_name == "__torch__" && class_name == "Module" => *args,
                    _ => continue,
                },
                _ => continue,
            },
            obj => obj,
        };

        // If key is provided, then we need to extract the state_dict from the object.
        let obj = if let Some(key) = key {
            let multi_key: Vec<&str> = key.split(".").collect();
            if multi_key.len() > 1 {
                let mut current_obj = obj;
                for k in multi_key.iter() {
                    if let Object::Dict(key_values) = current_obj {
                        current_obj = key_values
                            .into_iter()
                            .find(|(key_obj, _)| *key_obj == Object::Unicode(k.to_string()))
                            .map(|(_, v)| v)
                            .ok_or_else(|| anyhow!(format!("key '{}' not found", k)))?;
                    } else {
                        return Err(anyhow!(format!(
                            "Expected dictionary at key '{}', but found other type",
                            k
                        )));
                    }
                }
                current_obj
            } else if let Object::Dict(key_values) = obj {
                key_values
                    .into_iter()
                    .find(|(k, _)| *k == Object::Unicode(key.to_owned()))
                    .map(|(_, v)| v)
                    .ok_or_else(|| anyhow!(format!("key {key} not found")))?
            } else {
                obj
            }
        } else {
            obj
        };

        // If the object is a dict, then we can extract the tensor info from it.
        // NOTE: We are assuming that the `obj` is state_dict by this stage.
        if let Object::Dict(key_values) = obj {
            for (name, value) in key_values.into_iter() {
                match value.into_tensor_info(name, &dir_name) {
                    Ok(Some(tensor_info)) => tensor_infos.push(tensor_info),
                    Ok(None) => {}
                    Err(err) => eprintln!("skipping: {err:?}"),
                }
            }
        }
    }
    let tensor_infos: HashMap<String, TensorInfo> = tensor_infos
        .into_iter()
        .map(|ti| (ti.name.to_string(), ti))
        .collect();

    let tensor_names = tensor_infos.keys();
    let mut tensors = Vec::with_capacity(tensor_names.len());
    for name in tensor_names {
        match tensor_infos.get(name) {
            None => {}
            Some(tensor_info) => {
                let zip_reader = std::io::BufReader::new(std::fs::File::open(&path)?);
                let mut zip = zip::ZipArchive::new(zip_reader)?;
                let mut reader = zip.by_name(&tensor_info.path)?;
                let is_fortran_contiguous = tensor_info.layout.is_fortran_contiguous();
                let rank = tensor_info.layout.shape().rank();

                // Reading the data is a bit tricky as it can be strided, for now only support the basic
                // case and when the tensor is fortran contiguous.
                if !tensor_info.layout.is_contiguous() && !is_fortran_contiguous {
                    return Err(anyhow!(format!(
                        "cannot retrieve non-contiguous tensors {:?}",
                        tensor_info.layout
                    )));
                }
                let start_offset = tensor_info.layout.start_offset();
                if start_offset > 0 {
                    std::io::copy(
                        &mut reader.by_ref().take(start_offset as u64),
                        &mut std::io::sink(),
                    )?;
                }
                let tensor = crate_tensor_from_reader(
                    tensor_info.layout.shape().clone(),
                    tensor_info.dtype,
                    &mut reader,
                )?;

                if rank > 1 && is_fortran_contiguous {
                    // Reverse the shape, e.g. Shape(2, 3, 4) -> Shape(4, 3, 2)
                    let shape_reversed: Vec<_> =
                        tensor_info.layout.dims().iter().rev().cloned().collect();
                    let tensor = tensor.reshape(shape_reversed)?;

                    // Permute (transpose) the dimensions, e.g. Shape(4, 3, 2) -> Shape(2, 3, 4)
                    let dim_indeces_reversed: Vec<_> = (0..rank).rev().collect();
                    let tensor = tensor.permute(dim_indeces_reversed)?;
                    // Ok(Some(tensor))
                    tensors.push((name.clone(), tensor));
                } else {
                    tensors.push((name.clone(), tensor));
                }
            }
        };
    }
    let mut dict_to_hashmap = HashMap::new();
    for (k, v) in tensors {
        dict_to_hashmap.insert(k, v);
    }
    Ok(dict_to_hashmap)
}

// 获取秒级时间戳
pub fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

// 获取毫秒级时间戳
pub fn timestamp_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
}

pub fn round_by_factor(num: u32, factor: u32) -> u32 {
    let round = (num as f32 / factor as f32).round() as u32;
    round * factor
}

pub fn floor_by_factor(num: f32, factor: u32) -> u32 {
    let floor = (num / factor as f32).floor() as u32;
    floor * factor
}

pub fn ceil_by_factor(num: f32, factor: u32) -> u32 {
    let ceil = (num / factor as f32).ceil() as u32;
    ceil * factor
}

pub fn extract_mes(mes: &ChatCompletionParameters) -> Result<Vec<(String, String)>> {
    let mut mes_vec = Vec::new();
    for chat_mes in mes.messages.clone() {
        if let ChatMessage::User { content, .. } = chat_mes.clone()
            && let ChatMessageContent::ContentPart(part_vec) = content
        {
            for part in part_vec {
                if let ChatMessageContentPart::Text(text_part) = part {
                    let text = text_part.text;
                    mes_vec.push(("<|User|>".to_string(), text));
                }
            }
        } else if let ChatMessage::Assistant { content, .. } = chat_mes.clone()
            && let Some(cont) = content
            && let ChatMessageContent::Text(c) = cont
        {
            mes_vec.push(("<|Assistant|>".to_string(), c));
        }
    }
    Ok(mes_vec)
}

pub fn extract_metadata_value<T>(
    metadata: &Option<std::collections::HashMap<String, String>>,
    key: &str,
) -> Option<T>
where
    T: std::str::FromStr + Clone + PartialEq,
{
    if let Some(map) = metadata
        && let Some(value_str) = map.get(key)
        && let Ok(value) = value_str.parse::<T>()
    {
        return Some(value);
    }
    None
}

pub fn extract_user_text(mes: &ChatCompletionParameters) -> Result<String> {
    let mut ret = "".to_string();
    for chat_mes in mes.messages.clone() {
        if let ChatMessage::User { content, .. } = chat_mes.clone() {
            match content {
                ChatMessageContent::Text(text) => ret = ret + &text + "\n",
                ChatMessageContent::ContentPart(part_vec) => {
                    for part in part_vec {
                        if let ChatMessageContentPart::Text(text_part) = part {
                            let text = text_part.text;
                            if text.chars().count() > 0 {
                                ret = ret + &text + "\n"
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    ret = ret.trim().to_string();
    Ok(ret)
}

pub fn extract_user_text_vec(mes: &ChatCompletionParameters) -> Result<Vec<String>> {
    let mut ret = vec![];
    for chat_mes in mes.messages.clone() {
        if let ChatMessage::User { content, .. } = chat_mes.clone()
            && let ChatMessageContent::ContentPart(part_vec) = content
        {
            for part in part_vec {
                if let ChatMessageContentPart::Text(text_part) = part {
                    let text = text_part.text;
                    if text.chars().count() > 0 {
                        ret.push(text);
                    }
                }
            }
        }
    }
    Ok(ret)
}

pub fn get_default_save_dir() -> Option<String> {
    // 宿主进程（桌宠）用 AHA_HOME 指定：Android 上没有 HOME，~/.aha 不可写；
    // 桌面上不设这个变量，仍然落在 ~/.aha。
    let dir = match std::env::var_os("AHA_HOME").filter(|v| !v.is_empty()) {
        Some(v) => PathBuf::from(v),
        None => home_dir()?.join(".aha"),
    };
    if let Err(e) = fs::create_dir_all(&dir) {
        eprintln!("Failed to create directory {:?}: {}", dir, e);
    }
    Some(dir.to_string_lossy().to_string())
}

pub async fn download_model(
    model_id: &str,
    save_dir: &str,
    max_retries: u32,
) -> anyhow::Result<()> {
    let dir = PathBuf::from(save_dir).join(model_id);
    download_files_to(model_id, &dir, max_retries, DownloadFilter::ALL)
        .await
        .map(|_| ())
}

/// 下载过滤：`only` 非空时只下这些相对路径——桌宠的 sherpa/laya 只要几个权重，
/// 而镜像仓库里通常还放着一份 fp32 的重复模型（SenseVoice 的 `model.onnx` 就有
/// 900MB+），没必要下。`skip_prefixes` 按路径前缀整段跳过。
#[derive(Clone, Copy)]
pub struct DownloadFilter<'a> {
    /// 只要这些文件（相对仓库根的路径）；`None` = 仓库里全部文件。
    pub only: Option<&'a [&'a str]>,
    /// 跳过这些路径前缀。
    pub skip_prefixes: &'a [&'a str],
}

impl DownloadFilter<'static> {
    /// 整个仓库都下（`aha download` 的默认行为）。
    pub const ALL: DownloadFilter<'static> = DownloadFilter {
        only: None,
        skip_prefixes: &[],
    };
}

/// 一次下载的进度快照。`done` / `total` 是整次任务，`file_done` / `file_total` 是当前文件。
#[derive(Clone, Copy)]
pub struct DownloadProgress<'a> {
    pub path: &'a str,
    pub file_done: u64,
    pub file_total: u64,
    pub done: u64,
    pub total: u64,
    pub files_done: usize,
    pub files_total: usize,
}

/// 下载控制：进度回调与取消检查都可以不给。
///
/// 回调在下载线程里同步触发（约 100ms 节流一次），别在里面做耗时操作；
/// FFI 那层把它转成 C 函数指针后，调用方要保证 `user` 在这次下载期间一直有效。
#[derive(Clone, Copy, Default)]
pub struct DownloadCtl<'a> {
    pub progress: Option<&'a dyn Fn(&DownloadProgress<'_>)>,
    pub cancel: Option<&'a dyn Fn() -> bool>,
}

impl DownloadCtl<'_> {
    fn cancelled(&self) -> bool {
        self.cancel.is_some_and(|cancel| cancel())
    }
}

/// 下载结果：真正写盘的文件（相对仓库根的路径, 字节数）。
#[derive(Default, Debug)]
pub struct DownloadSummary {
    pub files: Vec<(String, u64)>,
    pub skipped: usize,
}

struct ProgressTracker<'a> {
    ctl: DownloadCtl<'a>,
    total: u64,
    done: u64,
    files_done: usize,
    files_total: usize,
    last: Instant,
}

impl<'a> ProgressTracker<'a> {
    fn new(ctl: DownloadCtl<'a>, total: u64, files_total: usize) -> Self {
        Self {
            ctl,
            total,
            done: 0,
            files_done: 0,
            files_total,
            last: Instant::now(),
        }
    }

    fn check_cancel(&self) -> anyhow::Result<()> {
        if self.ctl.cancelled() {
            anyhow::bail!("下载已取消");
        }
        Ok(())
    }

    fn skip(&mut self, size: u64) {
        self.done += size;
        self.files_done += 1;
    }

    /// 当前文件又收到 `n` 字节。
    fn advance(&mut self, path: &str, n: u64, file_done: u64, file_total: u64) {
        self.done += n;
        self.emit(path, file_done, file_total, false);
    }

    fn finish_file(&mut self, path: &str, size: u64) {
        self.files_done += 1;
        self.emit(path, size, size, true);
    }

    fn emit(&mut self, path: &str, file_done: u64, file_total: u64, force: bool) {
        if !force && self.last.elapsed() < Duration::from_millis(100) {
            return;
        }
        self.last = Instant::now();
        if let Some(cb) = self.ctl.progress {
            cb(&DownloadProgress {
                path,
                file_done,
                file_total,
                done: self.done,
                total: self.total,
                files_done: self.files_done,
                files_total: self.files_total,
            });
        }
    }
}

/// 把 `repo_id` 指定的 ModelScope 仓库文件下到 `dir`，失败按 `max_retries` 重试。
///
/// 已存在且大小一致的文件跳过；剩下的从
/// `{MS_RESOLVE}/{repo_id}/resolve/master/{path}` 流式写盘。
pub async fn download_files_to(
    repo_id: &str,
    dir: &Path,
    max_retries: u32,
    filter: DownloadFilter<'_>,
) -> anyhow::Result<DownloadSummary> {
    download_files_to_with_ctl(repo_id, dir, max_retries, filter, DownloadCtl::default()).await
}

/// 同上，但可以挂进度回调和取消检查（FFI 层用这个）。
pub async fn download_files_to_with_ctl(
    repo_id: &str,
    dir: &Path,
    max_retries: u32,
    filter: DownloadFilter<'_>,
    ctl: DownloadCtl<'_>,
) -> anyhow::Result<DownloadSummary> {
    let max_retries = max_retries.max(1);
    let mut attempts = 0u32;
    loop {
        attempts += 1;
        println!("Attempting to download {repo_id} (attempt {attempts}/{max_retries})");

        match download_repo(repo_id, dir, filter, ctl).await {
            Ok(summary) => {
                println!("Model downloaded successfully");
                return Ok(summary);
            }
            Err(e) => {
                if attempts >= max_retries || ctl.cancelled() {
                    return Err(anyhow::anyhow!(
                        "Failed to download {repo_id} after {attempts} attempts. Last error: {e}"
                    ));
                }

                println!("Download failed (attempt {attempts}): {e}. Retrying in 2 seconds...");
                sleep(Duration::from_secs(2)).await;
            }
        }
    }
}

/// ModelScope 上模型的文件列表；字段名是接口原始的大写驼峰。
#[derive(serde::Deserialize)]
struct RepoFilesResponse {
    #[serde(rename = "Success")]
    success: bool,
    #[serde(rename = "Message")]
    message: String,
    #[serde(rename = "Data")]
    data: Option<RepoFilesData>,
}

#[derive(serde::Deserialize)]
struct RepoFilesData {
    #[serde(rename = "Files")]
    files: Vec<RepoFile>,
}

#[derive(serde::Deserialize)]
struct RepoFile {
    #[serde(rename = "Path")]
    path: String,
    #[serde(rename = "Size")]
    size: u64,
    #[serde(rename = "Type")]
    kind: String,
}

const MS_API: &str = "https://modelscope.cn/api/v1/models";
const MS_RESOLVE: &str = "https://modelscope.cn/models";
const MS_UA: &str = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 \
                     (KHTML, like Gecko) Chrome/89.0.4389.90 Safari/537.36";
/// 小于这个字节数的文件不核实际大小（tokens.txt 这类只有几百字节）。
const TINY_BYTES: u64 = 4096;

/// 把一个 ModelScope 仓库的文件下到 `dir`（已存在且大小一致的跳过）。
///
/// 自己写而不用 `modelscope` crate：它的 reqwest 走平台证书校验，Android 上直接 panic，
/// 见 [`http`]。顺手也去掉了那个依赖。
async fn download_repo(
    repo_id: &str,
    dir: &Path,
    filter: DownloadFilter<'_>,
    ctl: DownloadCtl<'_>,
) -> Result<DownloadSummary> {
    if ctl.cancelled() {
        anyhow::bail!("下载已取消");
    }
    fs::create_dir_all(dir)?;
    println!("\nDownloading model {repo_id} to: {}\n", dir.display());

    let client = http::client()?;
    let resp = client
        .get(format!("{MS_API}/{repo_id}/repo/files?Recursive=true"))
        .header("User-Agent", MS_UA)
        .send()
        .await?;
    if !resp.status().is_success() {
        anyhow::bail!(
            "Failed to list files of {repo_id}: HTTP {}（模型 id 写错了，或者需要登录）",
            resp.status()
        );
    }
    let body: RepoFilesResponse = resp.json().await?;
    if !body.success {
        anyhow::bail!("Failed to list files of {repo_id}: {}", body.message);
    }
    let files = body
        .data
        .ok_or_else(|| anyhow!("Failed to list files of {repo_id}: empty response"))?
        .files;
    let blobs: Vec<RepoFile> = files
        .into_iter()
        .filter(|f| f.kind == "blob")
        .filter(|f| match filter.only {
            Some(only) => only.contains(&f.path.as_str()),
            None => true,
        })
        .filter(|f| {
            !filter
                .skip_prefixes
                .iter()
                .any(|prefix| f.path.starts_with(prefix))
        })
        .collect();
    if blobs.is_empty() {
        match filter.only {
            Some(only) => anyhow::bail!(
                "{repo_id} 里没有这些文件：{}（仓库的文件列表变了？）",
                only.join(", ")
            ),
            None => anyhow::bail!("{repo_id} 的文件列表是空的"),
        }
    }

    let mut summary = DownloadSummary::default();
    let total: u64 = blobs.iter().map(|file| file.size).sum();
    let mut tracker = ProgressTracker::new(ctl, total, blobs.len());
    for file in &blobs {
        tracker.check_cancel()?;
        let dst = dir.join(&file.path);
        if fs::metadata(&dst)
            .map(|m| m.len() == file.size)
            .unwrap_or(false)
        {
            summary.skipped += 1;
            tracker.skip(file.size);
            continue;
        }
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        download_file(&client, repo_id, file, &dst, &mut tracker).await?;
        println!("  {} ({:.1} MB)", file.path, file.size as f64 / 1_048_576.0);
        tracker.finish_file(&file.path, file.size);
        summary.files.push((file.path.clone(), file.size));
    }
    Ok(summary)
}

async fn download_file(
    client: &reqwest::Client,
    repo_id: &str,
    file: &RepoFile,
    dst: &Path,
    tracker: &mut ProgressTracker<'_>,
) -> Result<()> {
    // shortcut: 单个文件不支持断点续传（整个文件重下）；已完成的文件按大小跳过，
    // 只在网络很差的大 shard 上才会浪费带宽。
    let url = format!("{MS_RESOLVE}/{repo_id}/resolve/master/{}", file.path);
    let resp = client.get(&url).header("User-Agent", MS_UA).send().await?;
    if !resp.status().is_success() {
        anyhow::bail!("Failed to download {}: HTTP {}", file.path, resp.status());
    }

    // 先写 `<file>.part`，校验大小后再改名：进程中途被杀也不会留下"看起来完整"的半截文件
    // （`ensure_downloaded` 这类调用只按文件是否存在判断，半截文件会被当成已下完）。
    let part = dst.with_file_name(format!(
        "{}.part",
        dst.file_name().unwrap_or_default().to_string_lossy()
    ));
    let mut out = File::create(&part)?;
    let mut stream = resp.bytes_stream();
    let mut written = 0u64;
    while let Some(chunk) = stream.next().await {
        // 取消/退出时立即中断，`.part` 留着，下次从零重下这个文件
        tracker.check_cancel()?;
        let chunk = chunk?;
        written += chunk.len() as u64;
        out.write_all(&chunk)?;
        tracker.advance(&file.path, chunk.len() as u64, written, file.size);
    }
    out.flush()?;
    drop(out);
    // 大小对不上就是没下完（CDN 偶尔截断）：留着 .part，下次重下
    if file.size >= TINY_BYTES && written != file.size {
        anyhow::bail!(
            "下载不完整：{} 得到 {written} 字节，期望 {}",
            file.path,
            file.size
        );
    }
    fs::rename(&part, dst)?;
    Ok(())
}

pub fn get_file_path(file: &str) -> Result<PathBuf> {
    let path = url::Url::parse(file)?;
    let path = path.to_file_path();
    let path = match path {
        Ok(path) => path,
        Err(_) => {
            let mut path = file.to_owned();
            path = path.split_off(7);
            PathBuf::from(path)
        }
    };
    Ok(path)
}

pub fn capitalize_first_letter(input: &str) -> String {
    if input.is_empty() {
        return input.to_string();
    }

    let mut chars = input.chars();
    let first_char = chars.next().unwrap().to_uppercase().collect::<String>();
    let remaining = chars.as_str().to_lowercase();
    format!("{}{}", first_char, remaining)
}

pub fn load_tensor_from_pt(
    path: &str,
    zip_name: &str,
    shape: Shape,
    device: &Device,
) -> Result<Tensor> {
    let file = File::open(path)?;
    let mut archive = ZipArchive::new(file)?;
    // // 列出所有文件（调试用）
    // for i in 0..archive.len() {
    //     let file = archive.by_index(i)?;
    //     println!("File: {} ({} bytes)", file.name(), file.size());
    // }
    // 读取原始字节数据
    let mut data_file = archive.by_name(zip_name)?;
    let mut buffer = Vec::new();
    data_file.read_to_end(&mut buffer)?;
    // 将字节转换为 f32 (little endian)
    let mut cursor = Cursor::new(buffer);
    let num_elements = shape.elem_count();
    let mut data = Vec::with_capacity(num_elements);

    for _ in 0..num_elements {
        let val = cursor.read_f32::<LittleEndian>()?;
        data.push(val);
    }
    let t = Tensor::from_vec(data, shape, device)?;
    Ok(t)
}

/// Map OpenAI language code to full language name for ASR models
///
/// Supports 29 languages as per Qwen3ASR specification
pub fn map_language_code(code: &str) -> Option<String> {
    match code.to_lowercase().as_str() {
        "zh" => Some("Chinese".to_string()),
        "en" => Some("English".to_string()),
        "yue" => Some("Cantonese".to_string()),
        "ar" => Some("Arabic".to_string()),
        "de" => Some("German".to_string()),
        "fr" => Some("French".to_string()),
        "es" => Some("Spanish".to_string()),
        "pt" => Some("Portuguese".to_string()),
        "id" => Some("Indonesian".to_string()),
        "it" => Some("Italian".to_string()),
        "ko" => Some("Korean".to_string()),
        "ru" => Some("Russian".to_string()),
        "th" => Some("Thai".to_string()),
        "vi" => Some("Vietnamese".to_string()),
        "ja" => Some("Japanese".to_string()),
        "tr" => Some("Turkish".to_string()),
        "hi" => Some("Hindi".to_string()),
        "ms" => Some("Malay".to_string()),
        "nl" => Some("Dutch".to_string()),
        "sv" => Some("Swedish".to_string()),
        "da" => Some("Danish".to_string()),
        "fi" => Some("Finnish".to_string()),
        "pl" => Some("Polish".to_string()),
        "cs" => Some("Czech".to_string()),
        "fil" => Some("Filipino".to_string()),
        "fa" => Some("Persian".to_string()),
        "el" => Some("Greek".to_string()),
        "ro" => Some("Romanian".to_string()),
        "hu" => Some("Hungarian".to_string()),
        "mk" => Some("Macedonian".to_string()),
        _ => None,
    }
}

/// Clean ASR model output by extracting pure text from model-specific format
///
/// Qwen3ASR outputs format: "language English<asr_text>The morning sun..."
/// This function extracts the text after "<asr_text>" marker.
/// If no marker is found, returns the original text trimmed (for compatibility).
pub fn clean_asr_response(raw: &str) -> String {
    if let Some(start) = raw.find("<asr_text>") {
        raw[start + "<asr_text>".len()..].trim().to_string()
    } else {
        raw.trim().to_string()
    }
}

/// Get the default weight path for a given model
/// Returns ~/.aha/{model_id} e.g., ~/.aha/OpenBMB/VoxCPM1.5
pub fn get_default_weight_path(model: WhichModel) -> String {
    let model_id = model.as_string();
    let save_dir = get_default_save_dir().expect("Failed to get home directory");
    format!("{}/{}", save_dir, model_id)
}

/// Check if a model is downloaded by verifying the model directory exists
/// Returns true if ~/.aha/{model_id} directory exists, false otherwise
pub fn is_model_downloaded(model: WhichModel) -> bool {
    let model_id = model.as_string();
    let save_dir = match get_default_save_dir() {
        Some(dir) => dir,
        None => return false,
    };
    let model_path = format!("{}/{}", save_dir, model_id);
    std::path::Path::new(&model_path).exists()
}

/// Calculate total size of a directory recursively
pub fn dir_size(path: &std::path::Path) -> anyhow::Result<u64> {
    let mut total = 0;
    if path.is_dir() {
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let entry_path = entry.path();
            if entry_path.is_dir() {
                total += dir_size(&entry_path)?;
            } else {
                total += entry.metadata()?.len();
            }
        }
    } else {
        total = std::fs::metadata(path)?.len();
    }
    Ok(total)
}

/// Convert bytes to human readable format
pub fn bytes_to_human(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

pub fn bucketize(input: usize, boundaries: &[usize]) -> Result<usize> {
    if boundaries.is_empty() {
        return Err(anyhow!("bucketize param boundaries can not be empty"));
    }
    match boundaries.binary_search(&input) {
        Ok(i) => Ok(i),
        Err(i) => Ok(i),
    }
    // let mut index = 0;
    // let mut change = false;
    // for i in 0..boundaries.len() {
    //     if input <= boundaries[i] {
    //         index = i;
    //         change = true;
    //         break;
    //     }
    // }
    // if !change {
    //     index = boundaries.len();
    // }
    // Ok(index)
}

pub fn contains_cjk(text: &str) -> bool {
    for ch in text.chars() {
        let c = ch as u32;
        if (0x4e00..=0x9fff).contains(&c) // CJK Unified Ideographs
            || (0x3400..=0x4dbf).contains(&c) // CJK Unified Ideographs Extension A
            || (0x3040..=0x30ff).contains(&c) // Hiragana and Katakana
            || (0xac00..=0xd7af).contains(&c)
        // Hangul Syllables
        {
            return true;
        }
    }
    false
}

pub fn prepare_tts_text(text: &str) -> Result<String> {
    let mut normalized_text = text.trim().to_string();
    if normalized_text.is_empty() {
        return Err(anyhow!("Text cannot be empty."));
    }
    normalized_text = normalized_text.replace(['\n', '\r'], " ");
    while normalized_text.contains("  ") {
        normalized_text = normalized_text.replace("  ", " ");
    }

    if contains_cjk(&normalized_text) {
        let cjk_end_punctuations = ['。', '！', '？', '…', '.', '!', '?'];
        if !normalized_text.ends_with(|c: char| cjk_end_punctuations.contains(&c)) {
            normalized_text.push('。');
        }
        return Ok(normalized_text);
    }

    // Non-CJK (English/Western) logic
    // Capitalize first letter if it's lowercase alphabetic
    if let Some(first_char) = normalized_text.chars().next()
        && first_char.is_ascii_lowercase()
    {
        let mut chars = normalized_text.chars();
        chars.next(); // consume first char
        let rest: String = chars.collect();
        normalized_text = format!("{}{}", first_char.to_ascii_uppercase(), rest);
    }

    // Add period if ends with alphanumeric
    if let Some(last_char) = normalized_text.chars().last()
        && last_char.is_alphanumeric()
    {
        normalized_text.push('.');
    }

    // Add padding if less than 5 words
    // Split by whitespace to count words
    let word_count = normalized_text.split_whitespace().count();
    if word_count < 5 {
        normalized_text = format!("        {}", normalized_text); // 8 spaces
    }

    Ok(normalized_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map_language_code_chinese() {
        assert_eq!(map_language_code("zh"), Some("Chinese".to_string()));
    }

    #[test]
    fn test_map_language_code_english() {
        assert_eq!(map_language_code("en"), Some("English".to_string()));
    }

    #[test]
    fn test_map_language_code_case_insensitive() {
        assert_eq!(map_language_code("ZH"), Some("Chinese".to_string()));
        assert_eq!(map_language_code("EN"), Some("English".to_string()));
    }

    #[test]
    fn test_map_language_code_invalid() {
        assert_eq!(map_language_code("xx"), None);
    }

    #[test]
    fn test_clean_asr_response_standard_format() {
        let raw = "language English<asr_text>The morning sun cast golden light";
        let cleaned = clean_asr_response(raw);
        assert_eq!(cleaned, "The morning sun cast golden light");
    }

    #[test]
    fn test_clean_asr_response_chinese_format() {
        let raw = "language Chinese<asr_text>科技不断改变着我们的生活";
        let cleaned = clean_asr_response(raw);
        assert_eq!(cleaned, "科技不断改变着我们的生活");
    }

    #[test]
    fn test_clean_asr_response_with_newlines() {
        let raw = "language English<asr_text>\n\n  Hello world\n  ";
        let cleaned = clean_asr_response(raw);
        assert_eq!(cleaned, "Hello world");
    }

    #[test]
    fn test_clean_asr_response_no_marker() {
        let raw = "  Plain text without marker  ";
        let cleaned = clean_asr_response(raw);
        assert_eq!(cleaned, "Plain text without marker");
    }
}

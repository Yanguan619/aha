//! 设备 / dtype 解析：把 `--device` / `--dtype` 这类用户输入落到 candle 的类型上。

use std::process::Command;

use anyhow::Result;
use candle_core::{DType, Device};

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

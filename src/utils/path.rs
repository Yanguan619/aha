//! 权重目录与文件查找：`~/.aha` 位置、按后缀找文件、变体模型（vb）路径。

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device, pickle::read_all_with_key};
use candle_nn::VarBuilder;
use dirs::home_dir;

use crate::models::common::model_mapping::WhichModel;

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

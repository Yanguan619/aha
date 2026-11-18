pub mod audio_utils;
pub mod http;
pub mod npu;

pub mod device;
pub mod format;
pub mod img_utils;
pub mod interpolate;
pub mod path;
pub mod response_utils;
pub mod tensor_utils;
pub mod text;
pub mod video_utils;

pub use device::{get_device, get_dtype, get_gpu_sm_arch, resolve_device};

pub use path::{
    find_type_files, get_default_save_dir, get_default_weight_path, get_file_path,
    get_vb_extension, get_vb_model_path, is_model_downloaded,
};

pub use format::{bytes_to_human, dir_size, timestamp, timestamp_millis};

pub use text::{
    bucketize, capitalize_first_letter, ceil_by_factor, clean_asr_response, contains_cjk,
    extract_mes, extract_metadata_value, extract_user_text, extract_user_text_vec, floor_by_factor,
    map_language_code, prepare_tts_text, round_by_factor,
};

use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};

use anyhow::{Result, anyhow};
use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::{
    Context, DType, Device, Shape, Tensor,
    pickle::{Object, Stack, TensorInfo},
};
use half::{bf16, f16, slice::HalfFloatSliceExt};
use zip::ZipArchive;

pub fn string_to_static_str(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
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

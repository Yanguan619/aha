//! 宽松版 GGUF 头解析。
//!
//! candle 0.11 的 `gguf_file::Content::read` 在解析阶段就会对每个张量调用
//! `GgmlDType::from_u32`，遇到 I8/I16/I32/I64、IQ* 等类型直接 `bail`，整个文件打不开。
//! index-tts2.5 的 audiocpp 打包偏偏带了 123 个 I64（全是 BatchNorm 的
//! `num_batches_tracked`，推理根本不读），正好踩中这个坑。
//!
//! 这里自己解析头部，直接产出 candle 的 `gguf_file::Content`（字段全部 pub），
//! 把不认识的 dtype 跳过并记录到报告里。之后 `Gguf::new` / `QMatMul` /
//! `TensorInfo::read` 全部原样复用，**权重文件一个字节都不用改**。
//!
//! 顺带处理两件 audiocpp 特有的事：
//! 1. 部分张量名被压成 `_audiocpp.<idx>`，原始名在 metadata 的
//!    `audiocpp.tensor_names` 里（按张量顺序对齐），这里两种名字都登记；
//! 2. 内嵌的 config/tokenizer（`audiocpp.embedded_files.*`）体积可达十几 MB，
//!    如果走 `Value::Array` 会膨胀成几百 MB 内存，所以这里对大的 U8 数组只记录
//!    字节区间、不materialize，需要时用 [`embedded_files`] 单独读。

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

use anyhow::{Result, bail};
use byteorder::{LittleEndian, ReadBytesExt};
use candle_core::Shape;
use candle_core::quantized::{
    GgmlDType,
    gguf_file::{Content, DEFAULT_ALIGNMENT, TensorInfo, Value, VersionedMagic},
};

/// GGUF magic：`GGUF` 的小端/大端写法都接受（与 candle 一致）
const GGUF_MAGIC_A: u32 = 0x4655_4747;
const GGUF_MAGIC_B: u32 = 0x4747_5546;

const MAX_STRING_LENGTH: u64 = 1 << 30;
const MAX_ARRAY_ELEMENTS: u64 = 1 << 30;
const MAX_TENSOR_DIMS: u32 = 4;
const MAX_VALUE_DEPTH: usize = 64;

/// 超过这个长度的 U8 数组只记录区间，不进 `Value`（避免几 MB 的字节被展开成
/// 每个元素一个 `Value`）
const BYTE_ARRAY_INLINE_LIMIT: u64 = 1 << 16;

// metadata value type（对照 gguf 规范）
const T_U8: u32 = 0;
const T_ARRAY: u32 = 9;

/// 被跳过的张量（dtype 是 candle 不支持的）
#[derive(Debug, Clone)]
pub struct SkippedTensor {
    /// 文件里的原始名（可能就是 `_audiocpp.<idx>`）
    pub name: String,
    pub dtype_id: u32,
    pub dtype: &'static str,
    pub shape: Vec<usize>,
}

/// 解析报告，便于冒烟测试/排查用
#[derive(Debug, Default)]
pub struct LenientReport {
    pub version: u32,
    /// 文件头声明的张量总数
    pub tensor_count: usize,
    /// metadata kv 数量
    pub metadata_count: usize,
    /// 用 `audiocpp.tensor_names` 还原出原始名的张量数
    pub renamed: usize,
    /// 张量数据区起始偏移（已对齐）
    pub tensor_data_offset: u64,
    pub skipped: Vec<SkippedTensor>,
}

impl LenientReport {
    /// 被跳过的 dtype 汇总，例如 `{"I64": 123}`
    pub fn skipped_by_dtype(&self) -> HashMap<&'static str, usize> {
        let mut m = HashMap::new();
        for s in &self.skipped {
            *m.entry(s.dtype).or_insert(0) += 1;
        }
        m
    }
}

/// 大块 U8 数组在文件里的字节区间（用于内嵌文件）
#[derive(Debug, Clone, Copy)]
pub struct ByteRange {
    pub start: u64,
    pub len: u64,
}

/// 解析结果：`content` 直接喂给 `Gguf::new`
pub struct LenientContent {
    pub content: Content,
    pub report: LenientReport,
    /// metadata 里被“外置”的大 U8 数组，key 就是 metadata 的 key
    pub byte_arrays: HashMap<String, ByteRange>,
}

impl LenientContent {
    pub fn into_content(self) -> Content {
        self.content
    }

    /// 每个组件的张量数 + dtype 分布（按张量数据偏移去重，别名不重复计数）
    pub fn summary(&self) -> String {
        // 每个 offset 选一个“最好的名字”：优先用还原出来的原始名，
        // 而不是 `_audiocpp.<idx>` 这种压缩名（HashMap 迭代顺序不固定）
        let mut best: HashMap<u64, &str> = HashMap::new();
        for (name, info) in self.content.tensor_infos.iter() {
            let slot = best.entry(info.offset).or_insert(name.as_str());
            if slot.starts_with("_audiocpp.") && !name.starts_with("_audiocpp.") {
                *slot = name.as_str();
            }
        }

        let mut per: HashMap<&str, HashMap<String, usize>> = HashMap::new();
        for (name, info) in self.content.tensor_infos.iter() {
            if best.get(&info.offset) != Some(&name.as_str()) {
                continue;
            }
            let comp = match name.split_once('/') {
                Some((comp, _)) => comp,
                None => name.split('.').next().unwrap_or(name),
            };
            *per.entry(comp)
                .or_default()
                .entry(format!("{:?}", info.ggml_dtype))
                .or_insert(0) += 1;
        }
        let mut comps: Vec<_> = per.into_iter().collect();
        comps.sort_by_key(|(_, m)| std::cmp::Reverse(m.values().sum::<usize>()));
        let mut out = String::new();
        for (comp, dtypes) in comps {
            let total: usize = dtypes.values().sum();
            let mut ds: Vec<_> = dtypes.into_iter().collect();
            ds.sort_by_key(|(_, v)| std::cmp::Reverse(*v));
            let ds: Vec<String> = ds.into_iter().map(|(k, v)| format!("{k}={v}")).collect();
            out.push_str(&format!("  {comp:<22} {total:>5}  [{}]\n", ds.join(", ")));
        }
        out
    }
}

/// dtype id → candle 的 `GgmlDType`，不支持的返回 `None`
fn ggml_dtype_from_id(id: u32) -> Option<GgmlDType> {
    Some(match id {
        0 => GgmlDType::F32,
        1 => GgmlDType::F16,
        2 => GgmlDType::Q4_0,
        3 => GgmlDType::Q4_1,
        6 => GgmlDType::Q5_0,
        7 => GgmlDType::Q5_1,
        8 => GgmlDType::Q8_0,
        9 => GgmlDType::Q8_1,
        10 => GgmlDType::Q2K,
        11 => GgmlDType::Q3K,
        12 => GgmlDType::Q4K,
        13 => GgmlDType::Q5K,
        14 => GgmlDType::Q6K,
        15 => GgmlDType::Q8K,
        30 => GgmlDType::BF16,
        _ => return None,
    })
}

/// dtype id 的可读名（只用于报告）
pub fn dtype_name(id: u32) -> &'static str {
    match id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        10 => "Q2_K",
        11 => "Q3_K",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        15 => "Q8_K",
        16 => "IQ2_XXS",
        17 => "IQ2_XS",
        18 => "IQ3_XXS",
        19 => "IQ1_S",
        20 => "IQ4_NL",
        21 => "IQ3_S",
        22 => "IQ2_S",
        23 => "IQ4_XS",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        27 => "I64",
        28 => "F64",
        29 => "IQ1_M",
        30 => "BF16",
        34 => "TQ1_0",
        35 => "TQ2_0",
        _ => "UNKNOWN",
    }
}

fn read_len<R: Read>(reader: &mut R, magic: VersionedMagic) -> Result<u64> {
    Ok(match magic {
        VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as u64,
        VersionedMagic::GgufV2 | VersionedMagic::GgufV3 => reader.read_u64::<LittleEndian>()?,
    })
}

fn read_string<R: Read>(reader: &mut R, magic: VersionedMagic) -> Result<String> {
    let len = read_len(reader, magic)?;
    if len > MAX_STRING_LENGTH {
        bail!("gguf: 字符串长度 {len} 超限");
    }
    let mut buf = vec![0u8; len as usize];
    reader.read_exact(&mut buf)?;
    String::from_utf8(buf).map_err(|e| anyhow::anyhow!("gguf: 字符串不是合法 utf8: {e}"))
}

/// 读一个带 type-id 的 metadata value（顶层用）
fn read_value<R: Read + Seek>(
    reader: &mut R,
    magic: VersionedMagic,
    depth: usize,
    key: Option<&str>,
    ranges: &mut HashMap<String, ByteRange>,
) -> Result<Value> {
    let type_id = reader.read_u32::<LittleEndian>()?;
    read_value_of_type(reader, magic, type_id, depth, key, ranges)
}

fn read_value_of_type<R: Read + Seek>(
    reader: &mut R,
    magic: VersionedMagic,
    type_id: u32,
    depth: usize,
    key: Option<&str>,
    ranges: &mut HashMap<String, ByteRange>,
) -> Result<Value> {
    if depth > MAX_VALUE_DEPTH {
        bail!("gguf: metadata 嵌套超过 {MAX_VALUE_DEPTH} 层");
    }
    Ok(match type_id {
        0 => Value::U8(reader.read_u8()?),
        1 => Value::I8(reader.read_i8()?),
        2 => Value::U16(reader.read_u16::<LittleEndian>()?),
        3 => Value::I16(reader.read_i16::<LittleEndian>()?),
        4 => Value::U32(reader.read_u32::<LittleEndian>()?),
        5 => Value::I32(reader.read_i32::<LittleEndian>()?),
        6 => Value::F32(reader.read_f32::<LittleEndian>()?),
        7 => Value::Bool(reader.read_u8()? != 0),
        8 => Value::String(read_string(reader, magic)?),
        T_ARRAY => {
            let elem_type = reader.read_u32::<LittleEndian>()?;
            let len = read_len(reader, magic)?;
            if len > MAX_ARRAY_ELEMENTS {
                bail!("gguf: 数组长度 {len} 超限");
            }
            // 大块字节数组不展开成 Value，只记区间（内嵌 config/tokenizer）
            if elem_type == T_U8 && len > BYTE_ARRAY_INLINE_LIMIT {
                let start = reader.stream_position()?;
                if let Some(key) = key {
                    ranges.insert(key.to_string(), ByteRange { start, len });
                }
                reader.seek(SeekFrom::Start(start + len))?;
                Value::Array(Vec::new())
            } else {
                let mut items = Vec::with_capacity((len as usize).min(4096));
                for _ in 0..len {
                    items.push(read_value_of_type(
                        reader,
                        magic,
                        elem_type,
                        depth + 1,
                        None,
                        ranges,
                    )?);
                }
                Value::Array(items)
            }
        }
        10 => Value::U64(reader.read_u64::<LittleEndian>()?),
        11 => Value::I64(reader.read_i64::<LittleEndian>()?),
        12 => Value::F64(reader.read_f64::<LittleEndian>()?),
        _ => bail!("gguf: 未知 metadata value type {type_id}"),
    })
}

fn alignment_from(metadata: &HashMap<String, Value>) -> Option<u64> {
    match metadata.get("general.alignment") {
        Some(Value::U8(v)) => Some(*v as u64),
        Some(Value::U16(v)) => Some(*v as u64),
        Some(Value::U32(v)) => Some(*v as u64),
        Some(Value::I8(v)) if *v > 0 => Some(*v as u64),
        Some(Value::I16(v)) if *v > 0 => Some(*v as u64),
        Some(Value::I32(v)) if *v > 0 => Some(*v as u64),
        _ => None,
    }
}

fn string_array(value: &Value) -> Option<Vec<String>> {
    match value {
        Value::Array(items) => Some(
            items
                .iter()
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
        ),
        _ => None,
    }
}

fn u64_array(value: &Value) -> Option<Vec<u64>> {
    match value {
        Value::Array(items) => Some(
            items
                .iter()
                .filter_map(|v| match v {
                    Value::U64(n) => Some(*n),
                    Value::U32(n) => Some(*n as u64),
                    Value::U16(n) => Some(*n as u64),
                    Value::U8(n) => Some(*n as u64),
                    _ => None,
                })
                .collect(),
        ),
        _ => None,
    }
}

/// 解析 GGUF 头，产出可被 candle 消费的 `Content`（跳过的 dtype 记在报告里）。
pub fn read<R: Read + Seek>(reader: &mut R) -> Result<LenientContent> {
    let start = reader.stream_position()?;
    let file_size = reader.seek(SeekFrom::End(0))?;
    reader.seek(SeekFrom::Start(start))?;

    let magic_raw = reader.read_u32::<LittleEndian>()?;
    if magic_raw != GGUF_MAGIC_A && magic_raw != GGUF_MAGIC_B {
        bail!("gguf: 未知 magic 0x{magic_raw:08x}");
    }
    let version = reader.read_u32::<LittleEndian>()?;
    let magic = match version {
        1 => VersionedMagic::GgufV1,
        2 => VersionedMagic::GgufV2,
        3 => VersionedMagic::GgufV3,
        v => bail!("gguf: 不支持的版本 {v}"),
    };

    let tensor_count = read_len(reader, magic)?;
    let metadata_count = read_len(reader, magic)?;

    let mut ranges: HashMap<String, ByteRange> = HashMap::new();
    let mut metadata = HashMap::with_capacity(metadata_count.min(4096) as usize);
    for _ in 0..metadata_count {
        let key = read_string(reader, magic)?;
        let value = read_value(reader, magic, 0, Some(&key), &mut ranges)?;
        metadata.insert(key, value);
    }

    // audiocpp 会把一部分名字替换成 `_audiocpp.<idx>`，原名按张量顺序存在这里
    let original_names = metadata.get("audiocpp.tensor_names").and_then(string_array);

    let mut report = LenientReport {
        version,
        tensor_count: tensor_count as usize,
        metadata_count: metadata_count as usize,
        renamed: 0,
        tensor_data_offset: 0,
        skipped: Vec::new(),
    };

    let mut tensor_infos: HashMap<String, TensorInfo> =
        HashMap::with_capacity(tensor_count as usize * 2);

    for idx in 0..tensor_count {
        let raw_name = read_string(reader, magic)?;
        let n_dims = reader.read_u32::<LittleEndian>()?;
        if n_dims > MAX_TENSOR_DIMS {
            bail!("gguf: 张量 {raw_name} 维度数 {n_dims} 超限");
        }
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            let d = match magic {
                VersionedMagic::GgufV1 => reader.read_u32::<LittleEndian>()? as u64,
                _ => reader.read_u64::<LittleEndian>()?,
            };
            dims.push(usize::try_from(d)?);
        }
        // ggml 里维度是反序存的，翻过来才是 PyTorch 的 (out, in)
        dims.reverse();

        let dtype_id = reader.read_u32::<LittleEndian>()?;
        let offset = reader.read_u64::<LittleEndian>()?;

        let Some(ggml_dtype) = ggml_dtype_from_id(dtype_id) else {
            report.skipped.push(SkippedTensor {
                name: raw_name,
                dtype_id,
                dtype: dtype_name(dtype_id),
                shape: dims,
            });
            continue;
        };

        let shape = Shape::from(dims);

        // 真实名优先，压缩名保留成别名，两种 key 都能取到
        let mut names = vec![raw_name.clone()];
        if let Some(orig) = original_names.as_ref().and_then(|v| v.get(idx as usize))
            && !orig.is_empty()
            && orig != &raw_name
        {
            names.insert(0, orig.clone());
            report.renamed += 1;
        }
        for name in names {
            tensor_infos.insert(
                name,
                TensorInfo {
                    ggml_dtype,
                    shape: shape.clone(),
                    offset,
                },
            );
        }
    }

    let position = reader.stream_position()?;
    let alignment = alignment_from(&metadata)
        .unwrap_or(DEFAULT_ALIGNMENT)
        .max(1);
    let tensor_data_offset = position.div_ceil(alignment) * alignment;
    if tensor_data_offset > file_size {
        bail!("gguf: 张量数据区起点 {tensor_data_offset} 超出文件大小 {file_size}");
    }
    report.tensor_data_offset = tensor_data_offset;

    Ok(LenientContent {
        content: Content {
            magic,
            metadata,
            tensor_infos,
            tensor_data_offset,
        },
        report,
        byte_arrays: ranges,
    })
}

/// 取出全部内嵌文件（`audiocpp.embedded_files.*`），按名字返回。
pub fn embedded_files<R: Read + Seek>(
    reader: &mut R,
    lc: &LenientContent,
) -> Result<Vec<(String, Vec<u8>)>> {
    let names = lc
        .content
        .metadata
        .get("audiocpp.embedded_files.names")
        .and_then(string_array)
        .unwrap_or_default();
    let offsets = lc
        .content
        .metadata
        .get("audiocpp.embedded_files.offsets")
        .and_then(u64_array)
        .unwrap_or_default();
    let data = range_of(lc, "audiocpp.embedded_files.data");

    let mut out = Vec::with_capacity(names.len());
    for (idx, name) in names.iter().enumerate() {
        let start = match offsets.get(idx) {
            Some(v) => *v,
            None => break,
        };
        let end = offsets.get(idx + 1).copied().unwrap_or(match data {
            Some(r) => r.len,
            None => start,
        });
        let bytes = match data {
            Some(r) if end >= start && end - start <= r.len => {
                read_range(reader, r.start + start, end - start)?
            }
            _ => Vec::new(),
        };
        out.push((name.clone(), bytes));
    }
    Ok(out)
}

/// 按名字取单个内嵌文件
pub fn embedded_file<R: Read + Seek>(
    reader: &mut R,
    lc: &LenientContent,
    name: &str,
) -> Result<Option<Vec<u8>>> {
    Ok(embedded_files(reader, lc)?
        .into_iter()
        .find(|(n, _)| n == name)
        .map(|(_, b)| b))
}

fn range_of(lc: &LenientContent, key: &str) -> Option<ByteRange> {
    lc.byte_arrays.get(key).copied()
}

fn read_range<R: Read + Seek>(reader: &mut R, start: u64, len: u64) -> Result<Vec<u8>> {
    let mut buf = vec![0u8; len as usize];
    reader.seek(SeekFrom::Start(start))?;
    reader.read_exact(&mut buf)?;
    Ok(buf)
}

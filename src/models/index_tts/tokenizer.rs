//! IndexTTS 2.5 的文本分词器（tiktoken BPE）。
//!
//! 完全按上游 `indextts/utils/tokenizer.py` 复刻：
//! * 词表来自 GGUF 内嵌的 `multilingual_zh_ja_yue_char_del.tiktoken`（58836 个 mergeable ranks）；
//! * 特殊 token 从 `n_vocab` 开始按固定顺序追加（1673 个），`<|en|>`=58838、`<|zh|>`=58839 …；
//! * 预切分正则和上游**逐字一致**：
//!   `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`
//!   （这条正则里没有 `(?i)`，所以是大小写敏感的；`\s+(?!\S)` 和 `\s+` 的切分结果等价，
//!   这里只实现一次）。
//!
//! 没有引入 regex / fancy-regex 依赖，预切分是手写扫描器：按正则的备选顺序逐个尝试，
//! 用 `char::is_alphabetic / is_numeric / is_whitespace` 近似 `\p{L} / \p{N} / \s`。

use std::collections::HashMap;

use anyhow::{Result, bail};
use base64::Engine;

/// 和上游 `LANGUAGES` 顺序一致，用于生成 `<|xx|>` 特殊 token 的 id
const LANGUAGES: &[&str] = &[
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh", "common",
];

const AUDIO_EVENT: &[&str] = &[
    "ASR",
    "AED",
    "SER",
    "Speech",
    "/Speech",
    "BGM",
    "/BGM",
    "Laughter",
    "/Laughter",
    "Applause",
    "/Applause",
];

const EMOTION: &[&str] = &["HAPPY", "SAD", "ANGRY", "NEUTRAL"];

const TTS_VOCAL: &[&str] = &[
    "TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H", "TTS/SP01", "TTS/SP02",
    "TTS/SP03", "TTS/SP04", "TTS/SP05", "TTS/SP06", "TTS/SP07", "TTS/SP08", "TTS/SP09", "TTS/SP10",
    "TTS/SP11", "TTS/SP12", "TTS/SP13",
];

/// 语言 -> `lang_embedding` 的下标（上游 `lang_to_token`）
pub fn lang_to_token(lang: &str) -> u32 {
    let lang = lang.to_lowercase();
    LANGUAGES
        .iter()
        .position(|l| *l == lang)
        .unwrap_or_else(|| LANGUAGES.iter().position(|l| *l == "common").unwrap()) as u32
}

/// 特殊 token 表：`<|xx|>` -> id（id 从 n_vocab 开始按上游顺序连续分配）
fn build_specials(n_vocab: u32) -> HashMap<String, u32> {
    let mut out = HashMap::new();
    let mut next = n_vocab;
    let push = |s: String, next: &mut u32, out: &mut HashMap<String, u32>| {
        out.insert(s, *next);
        *next += 1;
    };
    push("<|endoftext|>".to_string(), &mut next, &mut out);
    push("<|startoftranscript|>".to_string(), &mut next, &mut out);
    for lang in LANGUAGES.iter().take(99) {
        push(format!("<|{lang}|>"), &mut next, &mut out);
    }
    for e in AUDIO_EVENT {
        push(format!("<|{e}|>"), &mut next, &mut out);
    }
    for e in EMOTION {
        push(format!("<|{e}|>"), &mut next, &mut out);
    }
    for s in [
        "<|translate|>",
        "<|transcribe|>",
        "<|startoflm|>",
        "<|startofprev|>",
        "<|nospeech|>",
        "<|notimestamps|>",
    ] {
        push(s.to_string(), &mut next, &mut out);
    }
    for i in 1..=30 {
        push(format!("<|SPECIAL_TOKEN_{i}|>"), &mut next, &mut out);
    }
    for t in TTS_VOCAL {
        push(format!("<|{t}|>"), &mut next, &mut out);
    }
    for i in 0..1501 {
        push(format!("<|{:.2}|>", i as f64 * 0.02), &mut next, &mut out);
    }
    out
}

pub struct TextTokenizer {
    ranks: HashMap<Vec<u8>, u32>,
    id_to_bytes: HashMap<u32, Vec<u8>>,
    specials: HashMap<String, u32>,
    special_ids: HashMap<u32, String>,
    /// mergeable 词表大小（特殊 token 从这里开始）
    pub n_vocab: u32,
}

impl TextTokenizer {
    /// 从 `.tiktoken` 文件内容构造（每行：base64(token) + 空格 + rank）
    pub fn from_tiktoken_bytes(data: &[u8]) -> Result<Self> {
        let text = std::str::from_utf8(data)?;
        let mut ranks = HashMap::new();
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let mut it = line.split_whitespace();
            let tok = it
                .next()
                .ok_or_else(|| anyhow::anyhow!("tiktoken 行格式错误"))?;
            let rank = it
                .next()
                .ok_or_else(|| anyhow::anyhow!("tiktoken 行缺少 rank"))?;
            let bytes = match base64::engine::general_purpose::STANDARD.decode(tok) {
                Ok(b) => b,
                // 词表里有一行是 `= 48474`（Python 的 base64.b64decode("=") 得到空串），
                // 这里保持同样行为，避免直接报错
                Err(_) if tok.chars().all(|c| c == '=') => Vec::new(),
                Err(e) => return Err(e.into()),
            };
            ranks.insert(bytes, rank.parse::<u32>()?);
        }
        if ranks.is_empty() {
            bail!("tiktoken 词表为空");
        }
        let n_vocab = ranks.len() as u32;
        let id_to_bytes = ranks.iter().map(|(k, v)| (*v, k.clone())).collect();
        let specials = build_specials(n_vocab);
        let special_ids = specials.iter().map(|(k, v)| (*v, k.clone())).collect();
        Ok(Self {
            ranks,
            id_to_bytes,
            specials,
            special_ids,
            n_vocab,
        })
    }

    /// `allowed_special = all`（和上游一致）
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            // 先看特殊 token
            if rest.starts_with("<|")
                && let Some(end) = rest.find("|>")
            {
                let cand = &rest[..end + 2];
                if let Some(id) = self.specials.get(cand) {
                    ids.push(*id);
                    rest = &rest[end + 2..];
                    continue;
                }
            }
            let (piece, consumed) = next_piece(rest);
            if consumed == 0 {
                bail!("分词器无法推进，停在: {rest:?}");
            }
            self.bpe(piece.as_bytes(), &mut ids)?;
            rest = &rest[consumed..];
        }
        Ok(ids)
    }

    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        let mut out: Vec<u8> = Vec::new();
        for id in ids {
            if let Some(s) = self.special_ids.get(id) {
                out.extend_from_slice(s.as_bytes());
            } else if let Some(b) = self.id_to_bytes.get(id) {
                out.extend_from_slice(b);
            } else {
                bail!("未知 token id {id}");
            }
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    /// 字节级 BPE：每次合并 rank 最小的相邻对
    fn bpe(&self, piece: &[u8], out: &mut Vec<u32>) -> Result<()> {
        if piece.is_empty() {
            return Ok(());
        }
        let mut parts: Vec<Vec<u8>> = piece.iter().map(|b| vec![*b]).collect();
        while parts.len() > 1 {
            let mut best: Option<(usize, u32)> = None;
            for i in 0..parts.len() - 1 {
                let mut key = parts[i].clone();
                key.extend_from_slice(&parts[i + 1]);
                if let Some(r) = self.ranks.get(&key)
                    && best.is_none_or(|(_, br)| *r < br)
                {
                    best = Some((i, *r));
                }
            }
            let Some((i, _)) = best else { break };
            let right = parts.remove(i + 1);
            parts[i].extend_from_slice(&right);
        }
        for p in parts {
            match self.ranks.get(&p) {
                Some(id) => out.push(*id),
                None => bail!("词表里没有这个字节序列: {:?}", String::from_utf8_lossy(&p)),
            }
        }
        Ok(())
    }
}

/// 按上游正则的备选顺序取下一个片段，返回 (片段, 消耗的字节数)
fn next_piece(s: &str) -> (&str, usize) {
    // 1) 英文缩写
    for p in ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"] {
        if s.starts_with(p) {
            return (&s[..p.len()], p.len());
        }
    }
    // 2) ` ?\p{L}+` / ` ?\p{N}+` / ` ?[^\s\p{L}\p{N}]+`
    let mut j = 0;
    if s.as_bytes().first() == Some(&b' ') {
        j = 1;
    }
    if j < s.len() {
        let c = s[j..].chars().next().unwrap();
        let class = classify(c);
        if class != 0 {
            let mut k = j;
            while k < s.len() {
                let c = s[k..].chars().next().unwrap();
                if classify(c) != class {
                    break;
                }
                k += c.len_utf8();
            }
            return (&s[..k], k);
        }
    }
    // 3) 空白串
    let mut k = 0;
    while k < s.len() {
        let c = s[k..].chars().next().unwrap();
        if !c.is_whitespace() {
            break;
        }
        k += c.len_utf8();
    }
    (&s[..k], k)
}

/// 1 = 字母, 2 = 数字, 3 = 其它非空白, 0 = 空白
fn classify(c: char) -> u8 {
    if c.is_alphabetic() {
        1
    } else if c.is_numeric() {
        2
    } else if c.is_whitespace() {
        0
    } else {
        3
    }
}

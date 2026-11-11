//! BERT WordPiece 分词（读 HuggingFace 的 `tokenizer.json`）。
//!
//! 只实现 GPT-SoVITS 用到的部分：`BertNormalizer` → `BertPreTokenizer` → `WordPiece`，
//! 外加 `[CLS]`/`[SEP]` 包装。没有 NFD 去重音（中文链路用不上），需要时可再加。

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct TokenizerFile {
    model: WordPieceModel,
}

#[derive(Debug, Deserialize)]
struct WordPieceModel {
    vocab: HashMap<String, u32>,
    #[serde(default)]
    continuing_subword_prefix: Option<String>,
    #[serde(default)]
    unk_token: Option<String>,
    #[serde(default)]
    max_input_chars_per_word: Option<usize>,
}

pub struct WordPieceTokenizer {
    vocab: HashMap<String, u32>,
    prefix: String,
    unk_id: u32,
    cls_id: u32,
    sep_id: u32,
    max_chars: usize,
}

impl WordPieceTokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file: TokenizerFile = serde_json::from_reader(std::io::BufReader::new(
            std::fs::File::open(path).with_context(|| format!("打开 {}", path.display()))?,
        ))
        .with_context(|| format!("解析 {}", path.display()))?;

        let model = file.model;
        let prefix = model
            .continuing_subword_prefix
            .unwrap_or_else(|| "##".to_string());
        let unk = model.unk_token.unwrap_or_else(|| "[UNK]".to_string());
        let lookup = |token: &str| -> Result<u32> {
            model
                .vocab
                .get(token)
                .copied()
                .ok_or_else(|| anyhow::anyhow!("词表里没有 {token:?}"))
        };
        let unk_id = lookup(&unk)?;
        let cls_id = lookup("[CLS]")?;
        let sep_id = lookup("[SEP]")?;
        let max_chars = model.max_input_chars_per_word.unwrap_or(100);

        Ok(Self {
            vocab: model.vocab,
            prefix,
            unk_id,
            cls_id,
            sep_id,
            max_chars,
        })
    }

    pub fn unk_id(&self) -> u32 {
        self.unk_id
    }

    /// `[CLS] + tokens + [SEP]`
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::with_capacity(text.chars().count() + 2);
        ids.push(self.cls_id);
        for token in self.basic_tokens(text) {
            ids.extend(self.word_piece(&token));
        }
        ids.push(self.sep_id);
        ids
    }

    /// 不带头尾特殊符号的 token 数（对应 `len(norm_text)`）。
    pub fn encode_tokens(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for token in self.basic_tokens(text) {
            ids.extend(self.word_piece(&token));
        }
        ids
    }

    /// 直接编码，并检查 token 数与原文字符数一致（GPT-SoVITS 有这个断言）。
    pub fn encode_expect_chars(&self, text: &str) -> Result<Vec<u32>> {
        let ids = self.encode(text);
        let tokens = ids.len().saturating_sub(2);
        if tokens != text.chars().count() {
            bail!(
                "分词结果 {tokens} 个 token 与原文字符数 {} 不一致：{text:?}",
                text.chars().count()
            );
        }
        Ok(ids)
    }

    fn basic_tokens(&self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        for word in self.normalize(text).split_whitespace() {
            let mut cur = String::new();
            for ch in word.chars() {
                if is_punctuation(ch) {
                    if !cur.is_empty() {
                        out.push(std::mem::take(&mut cur));
                    }
                    out.push(ch.to_string());
                } else {
                    cur.push(ch);
                }
            }
            if !cur.is_empty() {
                out.push(cur);
            }
        }
        out
    }

    /// BertNormalizer(clean_text + handle_chinese_chars + lowercase) + 词间空格
    fn normalize(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len() + 16);
        for ch in text.chars() {
            let ch = match ch {
                '\0' | '\u{fffd}' => continue,
                c if c.is_whitespace() => ' ',
                c if is_control(c) => continue,
                c => c,
            };
            if is_cjk(ch) {
                out.push(' ');
                out.extend(ch.to_lowercase());
                out.push(' ');
            } else {
                out.extend(ch.to_lowercase());
            }
        }
        out
    }

    fn word_piece(&self, word: &str) -> Vec<u32> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() > self.max_chars {
            return vec![self.unk_id];
        }
        let mut ids = Vec::with_capacity(chars.len());
        let mut start = 0;
        while start < chars.len() {
            let mut end = chars.len();
            let mut matched = None;
            while start < end {
                let sub: String = chars[start..end].iter().collect();
                let candidate = if start > 0 {
                    format!("{}{}", self.prefix, sub)
                } else {
                    sub
                };
                if let Some(id) = self.vocab.get(&candidate) {
                    matched = Some(*id);
                    break;
                }
                end -= 1;
            }
            match matched {
                Some(id) => {
                    ids.push(id);
                    start = end;
                }
                // 整词无法切分 → 单个 UNK
                None => return vec![self.unk_id],
            }
        }
        ids
    }
}

fn is_cjk(ch: char) -> bool {
    matches!(ch as u32,
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F
        | 0x2B820..=0x2CEAF
        | 0xF900..=0xFAFF
        | 0x2F800..=0x2FA1F)
}

fn is_control(ch: char) -> bool {
    let c = ch as u32;
    matches!(c, 0x09 | 0x0A | 0x0D) || (0x00..=0x1F).contains(&c) || (0x7F..=0x9F).contains(&c)
}

fn is_punctuation(ch: char) -> bool {
    let c = ch as u32;
    if (33..=47).contains(&c) || (58..=64).contains(&c) || (91..=96).contains(&c) || (123..=126).contains(&c)
    {
        return true;
    }
    matches!(c,
        0x00A1..=0x00BF
        | 0x2000..=0x206F
        | 0x2070..=0x209F
        | 0x20A0..=0x20CF
        | 0x2100..=0x214F
        | 0x2190..=0x21FF
        | 0x2200..=0x22FF
        | 0x2E00..=0x2E7F
        | 0x3000..=0x303F
        | 0xFE10..=0xFE19
        | 0xFE30..=0xFE6F
        | 0xFF00..=0xFF0F
        | 0xFF1A..=0xFF20
        | 0xFF3B..=0xFF40
        | 0xFF5B..=0xFF65
        | 0xFFE0..=0xFFEE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punctuation_and_cjk_helpers() {
        assert!(is_punctuation(','));
        assert!(is_punctuation('。'));
        assert!(is_punctuation('，'));
        assert!(!is_punctuation('a'));
        assert!(is_cjk('好'));
        assert!(!is_cjk('a'));
    }
}

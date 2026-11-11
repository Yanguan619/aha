//! 中文文本前端（GPT-SoVITS v2 系）。
//!
//! 对齐 `GPT_SoVITS/text/chinese2.py`：
//!
//! ```text
//! 归一化(数字/日期/标点) → 按标点切句 → jieba 分词+词性 → 预合并(不/一/叠词/三声/儿化)
//!   → 词库优先取拼音 → 变调(一/不/轻声/三声) → 儿化 → opencpop 音素映射 → phones/word2ph
//! ```
//!
//! 数据来自 `scripts/gpt_sovits/export_frontend_data.py` 导出的
//! `frontend_zh.json` + `dict.txt`（放在模型目录里）。

mod normalize;
mod tone_sandhi;

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result, bail};
use jieba_rs::Jieba;
use serde::Deserialize;

pub use normalize::{normalize_numbers, read_cardinal, read_digit_by_digit, split_sentences};
pub use tone_sandhi::Seg;

/// `v_rep_map`：多音节时韵母缩写。
const V_REP_MAP: [(&str, &str); 3] = [("uei", "ui"), ("iou", "iu"), ("uen", "un")];
/// `pinyin_rep_map`：零声母整音节改写。
const PINYIN_REP_MAP: [(&str, &str); 4] = [("ing", "ying"), ("i", "yi"), ("in", "yin"), ("u", "wu")];
/// `single_rep_map`：零声母单韵母改写。
const SINGLE_REP_MAP: [(char, &str); 4] = [('v', "yu"), ('e', "e"), ('i', "y"), ('u', "w")];

#[derive(Debug, Deserialize)]
struct ToneSandhiTables {
    must_neural: Vec<String>,
    must_not_neural: Vec<String>,
    punc: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct FrontendPack {
    chars: HashMap<String, Vec<String>>,
    phrases: HashMap<String, Vec<String>>,
    syllables: HashMap<String, Vec<String>>,
    opencpop: HashMap<String, Vec<String>>,
    tone_sandhi: ToneSandhiTables,
    rep_map: HashMap<String, String>,
    must_erhua: Vec<String>,
    not_erhua: Vec<String>,
    punctuation: Vec<String>,
}

/// 前端输出。
#[derive(Debug, Clone)]
pub struct G2pOutput {
    /// 归一化后的文本（与 `word2ph` 逐字对应）
    pub norm_text: String,
    /// phone 名字（`symbols.json` 里的符号）
    pub phones: Vec<String>,
    /// 每个「字」对应的 phone 个数
    pub word2ph: Vec<usize>,
}

impl G2pOutput {
    pub fn num_chars(&self) -> usize {
        self.word2ph.len()
    }
}

pub struct ZhFrontend {
    pack: FrontendPack,
    jieba: Jieba,
    symbols: Vec<String>,
    symbol_to_id: HashMap<String, u32>,
    sandhi: tone_sandhi::ToneSandhi,
    normalizer: normalize::Normalizer,
    must_erhua: HashSet<String>,
    not_erhua: HashSet<String>,
    punctuation: HashSet<char>,
}

impl ZhFrontend {
    /// 从模型目录加载（需要 `frontend_zh.json`、`dict.txt`、`symbols.json`）。
    pub fn load(model_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = model_dir.as_ref();

        let pack_path = dir.join("frontend_zh.json");
        let pack: FrontendPack = serde_json::from_reader(BufReader::new(File::open(&pack_path)
            .with_context(|| format!("打开 {}", pack_path.display()))?))
        .with_context(|| format!("解析 {}", pack_path.display()))?;

        let dict_path = dir.join("dict.txt");
        let mut reader = BufReader::new(
            File::open(&dict_path).with_context(|| format!("打开 {}", dict_path.display()))?,
        );
        let jieba = Jieba::with_dict(&mut reader)
            .with_context(|| format!("加载分词词典 {}", dict_path.display()))?;

        let symbols_path = dir.join("symbols.json");
        let symbols: Vec<String> = serde_json::from_reader(BufReader::new(
            File::open(&symbols_path)
                .with_context(|| format!("打开 {}", symbols_path.display()))?,
        ))
        .with_context(|| format!("解析 {}", symbols_path.display()))?;
        let symbol_to_id = symbols
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), i as u32))
            .collect();

        let sandhi = tone_sandhi::ToneSandhi::new(
            pack.tone_sandhi.must_neural.clone(),
            pack.tone_sandhi.must_not_neural.clone(),
            pack.tone_sandhi.punc.clone(),
        );
        let normalizer = normalize::Normalizer::new(&pack.rep_map, &pack.punctuation);

        Ok(Self {
            must_erhua: pack.must_erhua.iter().cloned().collect(),
            not_erhua: pack.not_erhua.iter().cloned().collect(),
            punctuation: pack.punctuation.iter().flat_map(|p| p.chars()).collect(),
            pack,
            jieba,
            symbols,
            symbol_to_id,
            sandhi,
            normalizer,
        })
    }

    pub fn symbols(&self) -> &[String] {
        &self.symbols
    }

    pub fn num_symbols(&self) -> usize {
        self.symbols.len()
    }

    /// 按标点切成适合**流式合成**的片段。
    ///
    /// 和 [`Self::g2p`] 里的切分同一套标点，但多一条约束：太短的片段会并到下一段。
    /// 单字/两三个字送进 AR 会因为上下文不足很快吐 EOS，韵律和音质都差；而流式的
    /// 意义是「尽早出声」，片段切太碎反而得不偿失。
    ///
    /// `min_chars` 是片段的最短长度（按字符数）。末尾剩下的碎片会并到最后一个片段。
    pub fn split_for_streaming(&self, text: &str, min_chars: usize) -> Vec<String> {
        let norm = self.normalizer.normalize(text);
        let parts = split_sentences(&norm, &self.punctuation);
        let mut out: Vec<String> = Vec::new();
        let mut pending = String::new();
        for part in parts {
            pending.push_str(&part);
            if pending.chars().count() >= min_chars {
                out.push(std::mem::take(&mut pending));
            }
        }
        if !pending.is_empty() {
            match out.last_mut() {
                // 尾巴太短：并到最后一段，别单独送一次 AR。
                Some(last) => last.push_str(&pending),
                None => out.push(pending),
            }
        }
        out
    }

    /// 文本 → phones / word2ph。等价于 `chinese2.text_normalize` + `chinese2.g2p`。
    pub fn g2p(&self, text: &str) -> Result<G2pOutput> {
        let norm_text = self.normalizer.normalize(text);
        let sentences = split_sentences(&norm_text, &self.punctuation);

        let mut phones: Vec<String> = Vec::new();
        let mut word2ph: Vec<usize> = Vec::new();

        for sentence in sentences {
            let seg = normalize::strip_ascii_letters(&sentence);
            if seg.is_empty() {
                continue;
            }
            let seg_cut: Vec<Seg> = self
                .jieba
                .tag(&seg, true)
                .into_iter()
                .map(|t| Seg::new(t.word, t.tag))
                .collect();
            let finals_of = |w: &str| self.word_pinyin(w).1;
            let seg_cut = self.sandhi.pre_merge_for_modify(seg_cut, &finals_of);

            let mut initials_all: Vec<String> = Vec::new();
            let mut finals_all: Vec<String> = Vec::new();

            for item in seg_cut {
                if item.pos == "eng" {
                    continue;
                }
                let (initials, finals) = self.word_pinyin(&item.word);
                let finals = self.sandhi.modified_tone(
                    &self.jieba,
                    &item.word,
                    &item.pos,
                    finals,
                );
                let (initials, finals) =
                    self.merge_erhua(initials, finals, &item.word, &item.pos);
                initials_all.extend(initials);
                finals_all.extend(finals);
            }

            for (c, v) in initials_all.iter().zip(finals_all.iter()) {
                self.push_phone(c, v, &mut phones, &mut word2ph)?;
            }
        }

        Ok(G2pOutput {
            norm_text,
            phones,
            word2ph,
        })
    }

    /// phones → symbol id（未知符号落到 `UNK`，与 `cleaner.clean_text` 一致）。
    pub fn to_ids(&self, phones: &[String]) -> Vec<u32> {
        let unk = self.symbol_to_id.get("UNK").copied().unwrap_or(0);
        phones
            .iter()
            .map(|p| self.symbol_to_id.get(p).copied().unwrap_or(unk))
            .collect()
    }

    fn push_phone(
        &self,
        c: &str,
        v: &str,
        phones: &mut Vec<String>,
        word2ph: &mut Vec<usize>,
    ) -> Result<()> {
        if c == v {
            // 标点：声母==韵母
            phones.push(c.to_string());
            word2ph.push(1);
            return Ok(());
        }

        let mut v_chars: Vec<char> = v.chars().collect();
        let tone = match v_chars.pop() {
            Some(t) if ('1'..='5').contains(&t) => t,
            _ => bail!("无法从韵母 {v:?} 取出声调（phone={c}{v}）"),
        };
        let v_without_tone: String = v_chars.iter().collect();

        let mut pinyin = format!("{c}{v_without_tone}");
        if !c.is_empty() {
            if let Some((_, rep)) = V_REP_MAP.iter().find(|(k, _)| *k == v_without_tone) {
                pinyin = format!("{c}{rep}");
            }
        } else if let Some((_, rep)) = PINYIN_REP_MAP.iter().find(|(k, _)| *k == pinyin) {
            pinyin = rep.to_string();
        } else if let Some(first) = pinyin.chars().next()
            && let Some((_, rep)) = SINGLE_REP_MAP.iter().find(|(k, _)| *k == first)
        {
            pinyin = format!("{rep}{}", &pinyin[first.len_utf8()..]);
        }

        let (new_c, new_v) = self
            .pack
            .opencpop
            .get(&pinyin)
            .map(|sym| (sym[0].clone(), sym[1].clone()))
            .ok_or_else(|| anyhow::anyhow!("opencpop 映射里没有拼音 {pinyin:?}"))?;

        phones.push(new_c);
        phones.push(format!("{new_v}{tone}"));
        word2ph.push(2);
        Ok(())
    }

    /// 一个词内部逐字的 (声母, 韵母_tone3)。词内按最长词条匹配，等价 pypinyin 的
    /// `lazy_pinyin(word)`；非汉字原样透传。
    fn word_pinyin(&self, word: &str) -> (Vec<String>, Vec<String>) {
        let chars: Vec<char> = word.chars().collect();
        let mut initials = Vec::with_capacity(chars.len());
        let mut finals = Vec::with_capacity(chars.len());
        let mut i = 0;

        while i < chars.len() {
            let max_len = (chars.len() - i).min(8);
            let mut matched = 0;
            for len in (2..=max_len).rev() {
                let candidate: String = chars[i..i + len].iter().collect();
                if let Some(pys) = self.pack.phrases.get(&candidate)
                    && pys.len() == len
                {
                    for py in pys {
                        let (init, fin) = self.split_syllable(py);
                        initials.push(init);
                        finals.push(fin);
                    }
                    matched = len;
                    break;
                }
            }
            if matched > 0 {
                i += matched;
                continue;
            }

            let key: String = chars[i].to_string();
            let py = self
                .pack
                .chars
                .get(&key)
                .and_then(|v| v.first())
                .cloned()
                .unwrap_or_else(|| key.clone());
            let (init, fin) = self.split_syllable(&py);
            initials.push(init);
            finals.push(fin);
            i += 1;
        }
        (initials, finals)
    }

    /// 带调拼音 → (声母, 韵母_tone3)；查导出表，避免逆向 pypinyin 的零声母规则。
    fn split_syllable(&self, py: &str) -> (String, String) {
        if let Some(pair) = self.pack.syllables.get(py)
            && pair.len() == 2
        {
            return (pair[0].clone(), pair[1].clone());
        }
        // 非拼音（标点、字母等）：声母==韵母，后面按标点处理
        (py.to_string(), py.to_string())
    }

    /// `chinese2._merge_erhua`
    fn merge_erhua(
        &self,
        mut initials: Vec<String>,
        mut finals: Vec<String>,
        word: &str,
        pos: &str,
    ) -> (Vec<String>, Vec<String>) {
        let chars: Vec<char> = word.chars().collect();

        // 词尾「儿」读 er1 的修正成 er2
        if let (Some(last), Some(c)) = (finals.last_mut(), chars.last())
            && *c == '儿'
            && last == "er1"
        {
            *last = "er2".to_string();
        }

        if !self.must_erhua.contains(word)
            && (self.not_erhua.contains(word) || matches!(pos, "a" | "j" | "nr"))
        {
            return (initials, finals);
        }
        if finals.len() != chars.len() {
            return (initials, finals);
        }

        let last_two: String = if chars.len() >= 2 {
            chars[chars.len() - 2..].iter().collect()
        } else {
            word.to_string()
        };
        let not_erhua_tail = self.not_erhua.contains(&last_two);

        let mut new_finals: Vec<String> = Vec::with_capacity(finals.len());
        for (i, phn) in finals.iter().enumerate() {
            let mut phn = phn.clone();
            let is_last = i + 1 == finals.len();
            if is_last
                && chars[i] == '儿'
                && (phn == "er2" || phn == "er5")
                && !not_erhua_tail
                && let Some(prev) = new_finals.last()
                && let Some(prev_tone) = prev.chars().last()
            {
                phn = format!("er{prev_tone}");
            }
            new_finals.push(phn);
        }
        initials.shrink_to_fit();
        (initials, new_finals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rep_maps_match_python() {
        // 零声母：yin2 -> ("", "in2") -> pinyin "in" -> "yin"
        assert_eq!(PINYIN_REP_MAP.iter().find(|(k, _)| *k == "in").unwrap().1, "yin");
        assert_eq!(V_REP_MAP.iter().find(|(k, _)| *k == "uei").unwrap().1, "ui");
    }

    /// 流式切分：按标点切，但太短的片段要并到下一段（不留碎片送 AR）。
    #[test]
    fn streaming_split_merges_short_fragments() {
        let punct: HashSet<char> = ['。', '！', '？', '，', '；', '\n'].into_iter().collect();
        // 直接测纯切分逻辑（不依赖 ZhFrontend 的词典）：
        // "你好。走吧。今天天气真不错，我们出去逛逛吧。"
        let parts = normalize::split_sentences("你好。走吧。今天天气真不错，我们出去逛逛吧。", &punct);
        assert_eq!(
            parts,
            vec!["你好。", "走吧。", "今天天气真不错，", "我们出去逛逛吧。"]
        );

        // 模拟 split_for_streaming 的合并：min_chars=8 时前两句应该并成一段。
        let mut out: Vec<String> = Vec::new();
        let mut pending = String::new();
        for p in parts {
            pending.push_str(&p);
            if pending.chars().count() >= 8 {
                out.push(std::mem::take(&mut pending));
            }
        }
        if !pending.is_empty() {
            match out.last_mut() {
                Some(last) => last.push_str(&pending),
                None => out.push(pending),
            }
        }
        assert_eq!(out, vec!["你好。走吧。今天天气真不错，", "我们出去逛逛吧。"]);
        // 关键性质：没有一段是空的，且拼起来等于原文。
        assert!(out.iter().all(|s| !s.trim().is_empty()));
        assert_eq!(out.concat(), "你好。走吧。今天天气真不错，我们出去逛逛吧。");
    }
}

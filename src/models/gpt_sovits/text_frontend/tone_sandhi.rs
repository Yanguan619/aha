//! PaddleSpeech `ToneSandhi` 的移植（GPT-SoVITS `text/tone_sandhi.py`）。
//!
//! 只在中文链路上使用，作用是在 pypinyin 的原始声调上做：
//! 三声变调、一/不变调、轻声化、儿化前的合并。
//!
//! 与 Python 版本的唯一差别：声母韵母切分由调用方以回调传入（`finals_of`），
//! 因为 Rust 侧是用导出的数据表查的，不调用 pypinyin。

use std::collections::HashSet;

use jieba_rs::Jieba;

/// 一个分词单元（对应 Python 里的 `(word, pos)`）。
#[derive(Debug, Clone)]
pub struct Seg {
    pub word: String,
    pub pos: String,
}

impl Seg {
    pub fn new(word: impl Into<String>, pos: impl Into<String>) -> Self {
        Self {
            word: word.into(),
            pos: pos.into(),
        }
    }
}

pub struct ToneSandhi {
    must_neural: HashSet<String>,
    must_not_neural: HashSet<String>,
    punc: HashSet<char>,
}

fn all_tone_three(finals: &[String]) -> bool {
    !finals.is_empty() && finals.iter().all(|f| f.ends_with('3'))
}

fn is_reduplication(word: &str) -> bool {
    let chars: Vec<char> = word.chars().collect();
    chars.len() == 2 && chars[0] == chars[1]
}

/// 把最后一个音节的声调改成 `tone`（`finals[i][:-1] + tone`）。
fn set_tone(final_: &mut String, tone: char) {
    final_.pop();
    final_.push(tone);
}

fn char_len(word: &str) -> usize {
    word.chars().count()
}

/// Python `str.isnumeric()` 的中文近似：汉字数字也算数字。
fn is_numeric_char(c: char) -> bool {
    c.is_numeric() || "零〇一二三四五六七八九十百千万亿两".contains(c)
}

impl ToneSandhi {
    pub fn new(
        must_neural: Vec<String>,
        must_not_neural: Vec<String>,
        punc: Vec<String>,
    ) -> Self {
        Self {
            must_neural: must_neural.into_iter().collect(),
            must_not_neural: must_not_neural.into_iter().collect(),
            punc: punc.iter().flat_map(|p| p.chars()).collect(),
        }
    }

    fn last_two(&self, word: &str) -> String {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() >= 2 {
            chars[chars.len() - 2..].iter().collect()
        } else {
            word.to_string()
        }
    }

    fn is_must_neural(&self, word: &str) -> bool {
        self.must_neural.contains(word) || self.must_neural.contains(&self.last_two(word))
    }

    fn is_must_not_neural(&self, word: &str) -> bool {
        self.must_not_neural.contains(word) || self.must_not_neural.contains(&self.last_two(word))
    }

    // --- 合并阶段 ---------------------------------------------------------

    fn merge_bu(&self, seg: Vec<Seg>) -> Vec<Seg> {
        let mut new_seg: Vec<Seg> = Vec::with_capacity(seg.len());
        let mut last_word = String::new();
        for mut item in seg {
            if last_word == "不" {
                item.word = format!("不{}", item.word);
            }
            if item.word != "不" {
                new_seg.push(item.clone());
            }
            last_word = item.word;
        }
        if last_word == "不" {
            new_seg.push(Seg::new("不", "d"));
        }
        new_seg
    }

    fn merge_yi(&self, seg: Vec<Seg>) -> Vec<Seg> {
        // 第一遍：听 / 一 / 听 -> 听一听
        let mut first: Vec<Seg> = Vec::with_capacity(seg.len());
        let mut i = 0;
        while i < seg.len() {
            let mut merged = false;
            if i >= 1 && seg[i].word == "一" && i + 1 < seg.len() {
                let last = first.last().cloned().unwrap_or_else(|| seg[i - 1].clone());
                if last.word == seg[i + 1].word && last.pos == "v" && seg[i + 1].pos == "v" {
                    if let Some(entry) = first.last_mut() {
                        entry.word = format!("{}一{}", last.word, seg[i + 1].word);
                    }
                    i += 2;
                    merged = true;
                }
            }
            if !merged {
                first.push(seg[i].clone());
                i += 1;
            }
        }
        // 第二遍：单独的「一」并到后面的词
        let mut second: Vec<Seg> = Vec::with_capacity(first.len());
        for item in first {
            match second.last_mut() {
                Some(last) if last.word == "一" => last.word.push_str(&item.word),
                _ => second.push(item),
            }
        }
        second
    }

    fn merge_reduplication(&self, seg: Vec<Seg>) -> Vec<Seg> {
        let mut new_seg: Vec<Seg> = Vec::with_capacity(seg.len());
        for item in seg {
            match new_seg.last_mut() {
                Some(last) if last.word == item.word => last.word.push_str(&item.word),
                _ => new_seg.push(item),
            }
        }
        new_seg
    }

    fn merge_continuous_three_tones(
        &self,
        seg: Vec<Seg>,
        finals_of: &dyn Fn(&str) -> Vec<String>,
    ) -> Vec<Seg> {
        let sub_finals: Vec<Vec<String>> = seg.iter().map(|s| finals_of(&s.word)).collect();
        let mut merge_last = vec![false; seg.len()];
        let mut new_seg: Vec<Seg> = Vec::with_capacity(seg.len());
        for i in 0..seg.len() {
            let prev_ok = i >= 1
                && all_tone_three(&sub_finals[i - 1])
                && all_tone_three(&sub_finals[i])
                && !merge_last[i - 1];
            if prev_ok {
                let prev_len = char_len(&seg[i - 1].word);
                if !is_reduplication(&seg[i - 1].word) && prev_len + char_len(&seg[i].word) <= 3 {
                    if let Some(last) = new_seg.last_mut() {
                        last.word.push_str(&seg[i].word);
                    }
                    merge_last[i] = true;
                } else {
                    new_seg.push(seg[i].clone());
                }
            } else {
                new_seg.push(seg[i].clone());
            }
        }
        new_seg
    }

    fn merge_continuous_three_tones_2(
        &self,
        seg: Vec<Seg>,
        finals_of: &dyn Fn(&str) -> Vec<String>,
    ) -> Vec<Seg> {
        let sub_finals: Vec<Vec<String>> = seg.iter().map(|s| finals_of(&s.word)).collect();
        let mut merge_last = vec![false; seg.len()];
        let mut new_seg: Vec<Seg> = Vec::with_capacity(seg.len());
        for i in 0..seg.len() {
            let prev_ok = i >= 1
                && sub_finals[i - 1].last().is_some_and(|f| f.ends_with('3'))
                && sub_finals[i].first().is_some_and(|f| f.ends_with('3'))
                && !merge_last[i - 1];
            if prev_ok {
                let prev_len = char_len(&seg[i - 1].word);
                if !is_reduplication(&seg[i - 1].word) && prev_len + char_len(&seg[i].word) <= 3 {
                    if let Some(last) = new_seg.last_mut() {
                        last.word.push_str(&seg[i].word);
                    }
                    merge_last[i] = true;
                } else {
                    new_seg.push(seg[i].clone());
                }
            } else {
                new_seg.push(seg[i].clone());
            }
        }
        new_seg
    }

    fn merge_er(&self, seg: Vec<Seg>) -> Vec<Seg> {
        let mut new_seg: Vec<Seg> = Vec::with_capacity(seg.len());
        for item in seg {
            if item.word == "儿"
                && let Some(last) = new_seg.last_mut()
                && last.word != "#"
            {
                last.word.push('儿');
                continue;
            }
            new_seg.push(item);
        }
        new_seg
    }

    pub fn pre_merge_for_modify(
        &self,
        seg: Vec<Seg>,
        finals_of: &dyn Fn(&str) -> Vec<String>,
    ) -> Vec<Seg> {
        let seg = self.merge_bu(seg);
        let seg = self.merge_yi(seg);
        let seg = self.merge_reduplication(seg);
        let seg = self.merge_continuous_three_tones(seg, finals_of);
        let seg = self.merge_continuous_three_tones_2(seg, finals_of);
        self.merge_er(seg)
    }

    // --- 变调阶段 ---------------------------------------------------------

    fn bu_sandhi(&self, word: &str, mut finals: Vec<String>) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        if chars.len() == 3 && chars[1] == '不' {
            set_tone(&mut finals[1], '5');
        } else {
            for (i, c) in chars.iter().enumerate() {
                if *c == '不' && i + 1 < chars.len() && finals[i + 1].ends_with('4') {
                    set_tone(&mut finals[i], '2');
                }
            }
        }
        finals
    }

    fn yi_sandhi(&self, word: &str, mut finals: Vec<String>) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        let has_yi = word.contains('一');
        if has_yi && chars.iter().filter(|c| **c != '一').all(|c| is_numeric_char(*c)) {
            return finals;
        }
        if chars.len() == 3 && chars[1] == '一' && chars[0] == chars[2] {
            set_tone(&mut finals[1], '5');
        } else if word.starts_with("第一") {
            set_tone(&mut finals[1], '1');
        } else {
            for (i, c) in chars.iter().enumerate() {
                if *c != '一' || i + 1 >= chars.len() {
                    continue;
                }
                if finals[i + 1].ends_with('4') {
                    set_tone(&mut finals[i], '2');
                } else if !self.punc.contains(&chars[i + 1]) {
                    set_tone(&mut finals[i], '4');
                }
            }
        }
        finals
    }

    fn neural_sandhi(
        &self,
        jieba: &Jieba,
        word: &str,
        pos: &str,
        mut finals: Vec<String>,
    ) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        let pos_first = pos.chars().next().unwrap_or(' ');
        let must_not = self.is_must_not_neural(word);

        // 叠词：奶奶 / 试试 / 旺旺
        for j in 1..chars.len() {
            if chars[j] == chars[j - 1]
                && matches!(pos_first, 'n' | 'v' | 'a')
                && !must_not
            {
                set_tone(&mut finals[j], '5');
            }
        }

        let ge_idx = chars.iter().position(|c| *c == '个');
        let last = chars.last().copied();
        let n = chars.len();

        // 以下六种情况都是把最后一个音节读轻声
        let tail_neutral = last.is_some_and(|l| {
            "吧呢哈啊呐噻嘛吖嗨哦哒额滴哩哟喽啰耶喔诶的地得".contains(l)
        }) || (n == 1 && "了着过".contains(chars[0]) && matches!(pos, "ul" | "uz" | "ug"))
            || (n > 1
                && last.is_some_and(|l| "们子".contains(l))
                && matches!(pos_first, 'r' | 'n')
                && !must_not)
            || (n > 1
                && last.is_some_and(|l| "上下里".contains(l))
                && matches!(pos_first, 's' | 'l' | 'f'))
            || (n > 1
                && last.is_some_and(|l| "来去".contains(l))
                && "上下进出回过起开".contains(chars[n - 2]));

        let ge_neutral = ge_idx.is_some_and(|g| {
            g >= 1
                && (is_numeric_char(chars[g - 1])
                    || "几有两半多各整每做是".contains(chars[g - 1]))
        }) || word == "个";

        if tail_neutral {
            set_tone(&mut finals[n - 1], '5');
        } else if ge_neutral {
            let g = ge_idx.unwrap_or(0);
            set_tone(&mut finals[g], '5');
        } else if self.is_must_neural(word) {
            set_tone(&mut finals[n - 1], '5');
        }

        // 按 jieba 子词再扫一遍必读轻声
        let (first, second) = self.split_word(jieba, word);
        let first_len = char_len(&first).min(finals.len());
        let mut list_a: Vec<String> = finals[..first_len].to_vec();
        let mut list_b: Vec<String> = finals[first_len..].to_vec();
        if !first.is_empty()
            && !list_a.is_empty()
            && (self.must_neural.contains(&first) || self.must_neural.contains(&self.last_two(&first)))
        {
            let idx = list_a.len() - 1;
            set_tone(&mut list_a[idx], '5');
        }
        if !second.is_empty()
            && !list_b.is_empty()
            && (self.must_neural.contains(&second)
                || self.must_neural.contains(&self.last_two(&second)))
        {
            let idx = list_b.len() - 1;
            set_tone(&mut list_b[idx], '5');
        }
        list_a.extend(list_b);
        list_a
    }

    fn three_sandhi(
        &self,
        jieba: &Jieba,
        word: &str,
        mut finals: Vec<String>,
    ) -> Vec<String> {
        let n = char_len(word);
        if n == 2 && all_tone_three(&finals) {
            set_tone(&mut finals[0], '2');
        } else if n == 3 {
            let (first, _) = self.split_word(jieba, word);
            let first_len = char_len(&first);
            if all_tone_three(&finals) {
                if first_len == 2 {
                    set_tone(&mut finals[0], '2');
                    set_tone(&mut finals[1], '2');
                } else if first_len == 1 {
                    set_tone(&mut finals[1], '2');
                }
            } else {
                let mut list_a = finals[..first_len.min(finals.len())].to_vec();
                let mut list_b = finals[first_len.min(finals.len())..].to_vec();
                if all_tone_three(&list_a) && list_a.len() == 2 {
                    set_tone(&mut list_a[0], '2');
                }
                if all_tone_three(&list_b) && list_b.len() == 2 {
                    set_tone(&mut list_b[0], '2');
                } else if !all_tone_three(&list_b)
                    && list_b.first().is_some_and(|f| f.ends_with('3'))
                    && list_a.last().is_some_and(|f| f.ends_with('3'))
                {
                    let idx = list_a.len() - 1;
                    set_tone(&mut list_a[idx], '2');
                }
                list_a.extend(list_b);
                finals = list_a;
            }
        } else if n == 4 {
            let mut list_a = finals[..2.min(finals.len())].to_vec();
            let mut list_b = finals[2.min(finals.len())..].to_vec();
            if all_tone_three(&list_a) && !list_a.is_empty() {
                set_tone(&mut list_a[0], '2');
            }
            if all_tone_three(&list_b) && !list_b.is_empty() {
                set_tone(&mut list_b[0], '2');
            }
            list_a.extend(list_b);
            finals = list_a;
        }
        finals
    }

    /// `jieba.cut_for_search` 取最短子词，把词切成两段。
    pub fn split_word(&self, jieba: &Jieba, word: &str) -> (String, String) {
        let mut pieces: Vec<String> = jieba
            .cut_for_search(word, true)
            .into_iter()
            .map(|t| t.word.to_string())
            .collect();
        if pieces.is_empty() {
            return (word.to_string(), String::new());
        }
        pieces.sort_by_key(|p| char_len(p));
        let first = pieces[0].clone();
        let chars: Vec<char> = word.chars().collect();
        // Python 的 str.find 返回字符下标
        let first_chars: Vec<char> = first.chars().collect();
        let idx = chars
            .windows(first_chars.len())
            .position(|w| w == first_chars.as_slice())
            .unwrap_or(0);
        if idx == 0 {
            (
                first,
                chars[first_chars.len().min(chars.len())..].iter().collect(),
            )
        } else {
            (chars[..chars.len() - first_chars.len()].iter().collect(), first)
        }
    }

    pub fn modified_tone(
        &self,
        jieba: &Jieba,
        word: &str,
        pos: &str,
        finals: Vec<String>,
    ) -> Vec<String> {
        let finals = self.bu_sandhi(word, finals);
        let finals = self.yi_sandhi(word, finals);
        let finals = self.neural_sandhi(jieba, word, pos, finals);
        self.three_sandhi(jieba, word, finals)
    }
}

//! 中文文本归一化：数字/日期/百分比 + 标点替换 + 非法字符清理。
//!
//! 对齐 GPT-SoVITS `text/chinese2.py::text_normalize`：
//! `TextNormalizer`（PaddleSpeech 的 zh_normalization）→ `replace_punctuation`
//! → `replace_consecutive_punctuation`。
//!
//! 注意：完整的 `zh_normalization` 还处理电话、单位、分数、朝代等长尾；这里只实现
//! 最常用的「整数 / 小数 / 年月日 / 百分比」，其余原样保留（会被后面的过滤步骤剔除）。

use std::collections::{HashMap, HashSet};

const DIGITS: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
const UNITS: [&str; 4] = ["", "十", "百", "千"];
const BIG_UNITS: [&str; 3] = ["", "万", "亿"];

/// 0..=9999 的四位组读法（内部用，含零占位）。
fn read_group(n: u32) -> String {
    let mut out = String::new();
    let mut pending_zero = false;
    for i in (0..4).rev() {
        let d = (n / 10u32.pow(i as u32)) % 10;
        if d == 0 {
            if !out.is_empty() {
                pending_zero = true;
            }
            continue;
        }
        if pending_zero {
            out.push_str(DIGITS[0]);
            pending_zero = false;
        }
        out.push_str(DIGITS[d as usize]);
        out.push_str(UNITS[i]);
    }
    out
}

/// 整数 → 中文读法（1 → 一，10 → 十，105 → 一百零五，2024 → 二千零二十四）。
pub fn read_cardinal(n: u64) -> String {
    if n == 0 {
        return DIGITS[0].to_string();
    }
    if n < 10 {
        return DIGITS[n as usize].to_string();
    }
    // 10..=19 习惯读「十X」而不是「一十X」
    if n < 20 {
        return if n == 10 {
            UNITS[1].to_string()
        } else {
            format!("{}{}", UNITS[1], DIGITS[(n % 10) as usize])
        };
    }
    let groups = [
        ((n / 100_000_000) % 10_000) as u32,
        ((n / 10_000) % 10_000) as u32,
        (n % 10_000) as u32,
    ];
    let mut out = String::new();
    let mut pending_zero = false;
    for (idx, group) in groups.iter().enumerate() {
        let big = BIG_UNITS[2 - idx];
        if *group == 0 {
            if !out.is_empty() {
                pending_zero = true;
            }
            continue;
        }
        if pending_zero || (!out.is_empty() && *group < 1000) {
            out.push_str(DIGITS[0]);
        }
        pending_zero = false;
        out.push_str(&read_group(*group));
        out.push_str(big);
    }
    out
}

/// 逐位读（年份、手机号等）。
pub fn read_digit_by_digit(s: &str) -> String {
    s.chars()
        .map(|c| c.to_digit(10).map(|d| DIGITS[d as usize]).unwrap_or(""))
        .collect()
}

fn read_decimal(int_part: &str, frac_part: &str) -> String {
    let int_val: u64 = int_part.parse().unwrap_or(0);
    let mut out = read_cardinal(int_val);
    if !frac_part.is_empty() {
        out.push('点');
        out.push_str(&read_digit_by_digit(frac_part));
    }
    out
}

/// 把一串阿拉伯数字按上下文转成中文。
fn read_number(raw: &str, next_char: Option<char>) -> String {
    let (int_part, frac_part) = match raw.split_once('.') {
        Some((a, b)) => (a, b),
        None => (raw, ""),
    };
    // 年份：4 位且后面跟「年」→ 逐位读（二零二四年）
    if frac_part.is_empty() && int_part.len() == 4 && next_char == Some('年') {
        return read_digit_by_digit(int_part);
    }
    read_decimal(int_part, frac_part)
}

/// 数字/日期/百分比归一化。
pub fn normalize_numbers(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len() * 2);
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if !c.is_ascii_digit() {
            out.push(c);
            i += 1;
            continue;
        }
        // 收集数字（允许中间一个小数点，且点后必须还有数字）
        let start = i;
        while i < chars.len() && chars[i].is_ascii_digit() {
            i += 1;
        }
        if i + 1 < chars.len() && chars[i] == '.' && chars[i + 1].is_ascii_digit() {
            i += 1;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
        }
        let raw: String = chars[start..i].iter().collect();
        let next = chars.get(i).copied();

        if next == Some('%') {
            out.push_str("百分之");
            out.push_str(&read_number(&raw, None));
            i += 1; // 吃掉 %
            continue;
        }
        // 年份逐位读（二零二四年），其余（含月/日/号）按基数读
        if next == Some('年') && raw.len() == 4 {
            out.push_str(&read_digit_by_digit(&raw));
        } else {
            out.push_str(&read_number(&raw, next));
        }
    }
    out
}

/// GPT-SoVITS 的 `replace_punctuation` + `replace_consecutive_punctuation`。
pub struct Normalizer {
    /// 按 key 长度降序排列，保证 "..." 先于单字符匹配
    rep_map: Vec<(String, String)>,
    punctuation: HashSet<char>,
}

impl Normalizer {
    pub fn new(rep_map: &HashMap<String, String>, punctuation: &[String]) -> Self {
        let mut pairs: Vec<(String, String)> = rep_map
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        pairs.sort_by_key(|(k, _)| std::cmp::Reverse(k.chars().count()));
        let punct = punctuation
            .iter()
            .flat_map(|p| p.chars())
            .collect::<HashSet<char>>();
        Self {
            rep_map: pairs,
            punctuation: punct,
        }
    }

    pub fn normalize(&self, text: &str) -> String {
        let with_numbers = normalize_numbers(text);
        let replaced = self.replace_punctuation(&with_numbers);
        self.replace_consecutive_punctuation(&replaced)
    }

    fn replace_punctuation(&self, text: &str) -> String {
        let text = text.replace('嗯', "恩").replace('呣', "母");
        let mut out = String::with_capacity(text.len());
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        'outer: while i < chars.len() {
            for (key, value) in &self.rep_map {
                let key_chars: Vec<char> = key.chars().collect();
                if i + key_chars.len() <= chars.len() && chars[i..i + key_chars.len()] == key_chars[..]
                {
                    out.push_str(value);
                    i += key_chars.len();
                    continue 'outer;
                }
            }
            let c = chars[i];
            i += 1;
            if is_cjk(c) || self.punctuation.contains(&c) {
                out.push(c);
            }
        }
        out
    }

    fn replace_consecutive_punctuation(&self, text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        let mut last_is_punct = false;
        for c in text.chars() {
            let is_punct = self.punctuation.contains(&c);
            if is_punct && last_is_punct {
                continue;
            }
            last_is_punct = is_punct;
            out.push(c);
        }
        out
    }
}

fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fa5}').contains(&c)
}

/// `re.split("(?<=[punct])\\s*", text)`：在每个标点之后切开，标点留在前一段末尾。
pub fn split_sentences(text: &str, punctuation: &HashSet<char>) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    for c in text.chars() {
        cur.push(c);
        if punctuation.contains(&c) {
            out.push(std::mem::take(&mut cur));
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.into_iter()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

/// `re.sub("[a-zA-Z]+", "", seg)`
pub fn strip_ascii_letters(text: &str) -> String {
    text.chars().filter(|c| !c.is_ascii_alphabetic()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cardinal_numbers() {
        assert_eq!(read_cardinal(0), "零");
        assert_eq!(read_cardinal(3), "三");
        assert_eq!(read_cardinal(10), "十");
        assert_eq!(read_cardinal(15), "十五");
        assert_eq!(read_cardinal(25), "二十五");
        assert_eq!(read_cardinal(105), "一百零五");
        assert_eq!(read_cardinal(2024), "二千零二十四");
        assert_eq!(read_cardinal(10_000), "一万");
        assert_eq!(read_cardinal(30_005), "三万零五");
    }

    #[test]
    fn digit_by_digit() {
        assert_eq!(read_digit_by_digit("2024"), "二零二四");
    }

    #[test]
    fn number_context() {
        assert_eq!(normalize_numbers("现在是2024年"), "现在是二零二四年");
        assert_eq!(normalize_numbers("气温25度"), "气温二十五度");
        assert_eq!(normalize_numbers("8月15日"), "八月十五日");
        assert_eq!(normalize_numbers("涨了3.5%"), "涨了百分之三点五");
    }
}

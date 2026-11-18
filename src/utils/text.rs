//! 文本与请求体小工具：请求内容抽取、CJK 判断、TTS 文本预处理、ASR 响应清洗。
//!
//! 这里放**跨模型**的部分；某个模型的方言（提示模板、特殊 token）仍留在它自己的模块里。

use anyhow::{Result, anyhow};

use crate::params::chat::{
    ChatCompletionParameters, ChatMessage, ChatMessageContent, ChatMessageContentPart,
};

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

pub fn capitalize_first_letter(input: &str) -> String {
    if input.is_empty() {
        return input.to_string();
    }

    let mut chars = input.chars();
    let first_char = chars.next().unwrap().to_uppercase().collect::<String>();
    let remaining = chars.as_str().to_lowercase();
    format!("{}{}", first_char, remaining)
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

//! 流式 TTS 的句子切分：攒够一句就交给合成器，避免逐字合成。

/// 强停顿：一句话结束。
pub(super) const SENTENCE_ENDS: &[char] = &['。', '！', '？', '!', '?', '；', ';', '\n'];

/// 弱停顿：攒够一定长度再切，否则"你好，"这种会被切得太碎。
pub(super) const SOFT_ENDS: &[char] = &['，', ',', '、', '：', ':'];

/// 弱停顿生效的最短长度（字符数）。
pub(super) const SOFT_MIN_CHARS: usize = 24;

/// 首句的弱停顿门槛：宁可韵律断一下，也要早点开口。
///
/// 取 4 是因为中文短句常是「小柴迎客，」这种 5 字带逗号的结构：
/// 门槛 <=4 就能在第一处逗号切开，TTS 只合成 5 个字。
pub(super) const FIRST_SENTENCE_MIN_CHARS: usize = 4;

/// 一个停顿都没有时硬切的长度（字符数）。
pub(super) const HARD_MIN_CHARS: usize = 80;

/// 从生成缓冲区里捞完整句子；`flush_all` 为真时把残余也吐出来（生成结束）。
pub(super) fn drain_sentences(buffer: &mut String, flush_all: bool) -> Vec<String> {
    drain_sentences_with(buffer, flush_all, SOFT_MIN_CHARS, HARD_MIN_CHARS)
}

/// 同 [`drain_sentences`]，但可以自定义停顿阈值（首句用小阈值抢时间）。
pub(super) fn drain_sentences_with(
    buffer: &mut String,
    flush_all: bool,
    soft_min: usize,
    hard_min: usize,
) -> Vec<String> {
    let mut out = Vec::new();
    loop {
        let mut cut: Option<usize> = None;
        let mut soft: Option<usize> = None;
        for (index, ch) in buffer.char_indices() {
            if SENTENCE_ENDS.contains(&ch) {
                cut = Some(index + ch.len_utf8());
                break;
            }
            if SOFT_ENDS.contains(&ch) {
                soft = Some(index + ch.len_utf8());
            }
        }
        if cut.is_none()
            && let Some(position) = soft
            && buffer[..position].chars().count() >= soft_min
        {
            cut = Some(position);
        }
        if cut.is_none() && buffer.chars().count() >= hard_min {
            cut = buffer.char_indices().nth(hard_min).map(|(index, _)| index);
        }
        let Some(end) = cut else { break };
        let piece: String = buffer.drain(..end).collect();
        let piece = piece.trim().to_owned();
        if !piece.is_empty() {
            out.push(piece);
        }
    }
    if flush_all {
        let rest = buffer.trim().to_owned();
        buffer.clear();
        if !rest.is_empty() {
            out.push(rest);
        }
    }
    out
}

// ---------------------------------------------------------------------------
// 模型调用（feature 开关在这里收口，没编进去就给一句能看懂的错误）
// ---------------------------------------------------------------------------

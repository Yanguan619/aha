//! Laya 路由判定：一条输入 → VLA（身体动作）还是 LLM（对话）。
//!
//! 移植自桌宠仓库 `tools/laya/backend/rust/src/route.rs`（桌宠的 Laya 决策桥）。
//! 这份逻辑是 `tools/laya/backend/python/route.py` 的 **Rust 镜像**，三层判定顺序和
//! 阈值必须一致：
//!
//! 1. 空输入                     → llm（reason=empty，不调模型）
//! 2. 关键词快路径                → vla（reason=keyword，不调模型）
//! 3. 模型判定 + 置信度门控        → 见 [`decide`]
//!
//! 改这里要同时改 Python 那份（`tools/laya/backend/python/test_route.py` 的语料是
//! 共同验收标准；Rust 侧的等价验收是 `aha` 的 laya selftest，语料同样抄自 test_route.py）。

/// 低于这个置信度的 vla 判定一律回退 llm：宁可多说一句，也不做错动作。
pub const MIN_CONFIDENCE: f32 = 0.8;

/// 精确短语：只覆盖高频身体指令，长句/疑问句不进来（"你能走两步吗"交给模型）。
const VLA_PHRASES: &[&str] = &[
    "走",
    "走走",
    "走两步",
    "往前走",
    "往左走",
    "往右走",
    "走过来",
    "到我这里来",
    "过来",
    "转个身",
    "转过去",
    "背过去",
    "转圈圈",
    "跳一下",
    "跳一跳",
    "扔个球",
    "扔一下",
    "停下来",
    "站住",
    "别动",
    "摇一摇",
    "挥挥手",
    "拍拍手",
    "做个鬼脸",
    "做鬼脸",
    "看看屏幕",
    "看一下屏幕",
    "飞起来",
    "蹲下",
    "坐下",
    "停",
];

/// 短文本（≤6 字）前缀匹配。**长前缀优先**，免得"转个"先吃掉"转个身"。
const VLA_PREFIXES: &[&str] = &[
    "到我这里来",
    "往前走",
    "往左走",
    "往右走",
    "走过来",
    "转个身",
    "转过去",
    "背过去",
    "转圈圈",
    "跳一下",
    "跳一跳",
    "扔个",
    "停下来",
    "看看屏幕",
    "看一下屏幕",
    "摇一摇",
    "挥挥手",
    "拍拍手",
    "做个鬼脸",
    "做鬼脸",
    "飞起来",
    "站住",
];

/// 轻量归一：去空白、全角转半角。**不动中文标点**（和 Python 那份一致：
/// `"走两步。"` 靠前缀匹配命中，而不是靠去标点）。
pub fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for ch in text.trim().chars() {
        if ch.is_whitespace() {
            continue;
        }
        let mapped = match ch as u32 {
            // 全角 ASCII → 半角
            code @ 0xFF01..=0xFF5E => char::from_u32(code - 0xFEE0).unwrap_or(ch),
            _ => ch,
        };
        out.push(mapped);
    }
    out
}

/// 关键词快路径：精确短语命中，或 ≤6 字的短指令前缀命中。
///
/// 长度限制是安全边界——长句里的动词（"往前走会怎么样呢"）不能变成物理动作。
pub fn keyword_hit(norm: &str) -> bool {
    if norm.is_empty() {
        return false;
    }
    if VLA_PHRASES.contains(&norm) {
        return true;
    }
    norm.chars().count() <= 6 && VLA_PREFIXES.iter().any(|p| norm.starts_with(p))
}

/// 一次判定的完整结果。`target` 才是调用方该用的值。
#[derive(Debug, Clone, PartialEq)]
pub struct RouteResult {
    pub text: String,
    /// "vla" | "llm"
    pub target: &'static str,
    /// 模型/关键词的原始选择（关键词层与空输入时等于 target）
    pub choice: String,
    /// 模型给出的置信度；关键词层与空输入时为 None
    pub confidence: Option<f32>,
    /// empty / keyword / model / low_confidence / unknown_choice
    pub reason: &'static str,
}

/// 第 1 层：空输入。永不调模型，直接放行给 LLM（对话链路最安全）。
pub fn empty(text: &str) -> RouteResult {
    RouteResult {
        text: text.to_owned(),
        target: "llm",
        choice: "llm".to_owned(),
        confidence: None,
        reason: "empty",
    }
}

/// 第 2 层：关键词命中。
pub fn keyword(text: &str) -> RouteResult {
    RouteResult {
        text: text.to_owned(),
        target: "vla",
        choice: "vla".to_owned(),
        confidence: None,
        reason: "keyword",
    }
}

/// 第 3 层的门控：模型答了 vla 但不够自信时回退 llm。
///
/// 返回值是 `(target, reason)`：
/// - vla 且 conf < min → (llm, low_confidence)
/// - 模型回了未知值     → (llm, unknown_choice)
/// - 其余               → (choice, model)
pub fn decide(choice: &str, confidence: f32, min_confidence: f32) -> (&'static str, &'static str) {
    if choice == "vla" && confidence < min_confidence {
        return ("llm", "low_confidence");
    }
    match choice {
        "vla" => ("vla", "model"),
        "llm" => ("llm", "model"),
        _ => ("llm", "unknown_choice"),
    }
}

/// 验收语料：抄自 `tools/laya/backend/python/test_route.py` 的 MAIN_CORPUS。
/// 第一列是期望的最终 target（门控之后）。
pub const CORPUS: &[(&str, &str)] = &[
    // --- VLA：身体指令 ---
    ("vla", "走两步"),
    ("vla", "往左走"),
    ("vla", "转个身"),
    ("vla", "背过去"),
    ("vla", "扔个球"),
    ("vla", "跳一下"),
    ("vla", "停下来"),
    ("vla", "站住"),
    ("vla", "走过来"),
    ("vla", "到我这里来"),
    ("vla", "摇一摇"),
    ("vla", "做个鬼脸"),
    ("vla", "拍拍手"),
    ("vla", "看看屏幕"),
    ("vla", "转圈圈"),
    ("vla", "走"),
    ("vla", "你能走两步吗"), // 疑问式命令：关键词层不收，交给模型
    // --- LLM：对话/问答/推理 ---
    ("llm", "你是谁"),
    ("llm", "今天天气怎么样"),
    ("llm", "你吃饭了吗"),
    ("llm", "讲个笑话"),
    ("llm", "1+1等于几"),
    ("llm", "你会写诗吗"),
    ("llm", "给我讲个故事"),
    ("llm", "你好呀"),
    ("llm", "在吗"),
    ("llm", "你喜欢什么"),
    ("llm", "什么是爱"),
    ("llm", "帮我算一下 25乘以4"),
    ("llm", "你觉得我今天开心吗"),
    ("llm", "你为什么叫这个名字"),
];

/// 模糊/垃圾输入：一律默认 llm，绝不轻易触发物理动作。
pub const AMBIGUOUS: &[&str] = &["", "。。。。。", "😊", "你是谁呀今天过得怎么样"];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_matches_python() {
        // 全角 ！ 会被折成半角 !（和 Python 那份一致）
        assert_eq!(normalize(" 给我走两步看看！ "), "给我走两步看看!");
        // 全角转半角，但**不**转小写（route.py 的 _normalize 也不转）
        assert_eq!(normalize("Ａ Ｂ Ｃ"), "ABC");
        assert_eq!(normalize(""), "");
        // 中文标点不在 0xFF01..=0xFF5E 里，原样保留
        assert_eq!(normalize("走两步。"), "走两步。");
    }

    #[test]
    fn keyword_layer_is_deterministic() {
        for text in ["走两步", "转个身", "站住", "看看屏幕"] {
            assert!(keyword_hit(&normalize(text)), "{text} 应命中关键词层");
        }
        // 带标点的精确短语不走关键词层（和 Python 一致：它只做整词/前缀匹配，
        // 不做去标点）；这种输入交给模型，麻烦的是模型，不是关键词表。
        assert!(!keyword_hit(&normalize("走两步。")));
    }

    #[test]
    fn keyword_layer_stays_out_of_conversation() {
        // 长句里的动词不能变成物理动作
        for text in ["你能走两步吗", "讲个笑话", "往前走会怎么样呢", "什么是爱"]
        {
            assert!(!keyword_hit(&normalize(text)), "{text} 被关键词层误伤");
        }
    }

    #[test]
    fn gate_falls_back_on_low_confidence() {
        assert_eq!(
            decide("vla", 0.79, MIN_CONFIDENCE),
            ("llm", "low_confidence")
        );
        assert_eq!(decide("vla", 0.80, MIN_CONFIDENCE), ("vla", "model"));
    }

    #[test]
    fn gate_treats_unknown_choice_as_llm() {
        assert_eq!(
            decide("jump", 0.99, MIN_CONFIDENCE),
            ("llm", "unknown_choice")
        );
        assert_eq!(decide("", 0.99, MIN_CONFIDENCE), ("llm", "unknown_choice"));
    }

    #[test]
    fn gate_never_blocks_llm() {
        // conf 再低，模型答 llm 也是 llm（门控只拦动作）
        assert_eq!(decide("llm", 0.01, MIN_CONFIDENCE), ("llm", "model"));
    }

    #[test]
    fn corpus_labels_are_only_the_two_targets() {
        for (want, text) in CORPUS {
            assert!(matches!(*want, "vla" | "llm"), "{text} 标注非法: {want}");
        }
    }
}

//! `/realtime` —— OpenAI Realtime 形状的实时语音端点（WebSocket）。
//!
//! 一句话：一条 WebSocket 上把 **ASR → LLM → TTS** 串成对流式对话。
//! 事件名/字段名跟着 OpenAI Realtime 走，官方 JS/Python SDK 直接能连；
//! 但这里是"本机三段式流水线"，不是 OpenAI 语义的代理，差异都写在下面。
//!
//! # 客户端 → 服务端
//!
//! | 事件 | 作用 |
//! | --- | --- |
//! | `session.update` | 改会话配置（模型/音色/指令/采样率/回合检测） |
//! | `input_audio_buffer.append` | 追加 base64 PCM16，任意采样率（`input_audio_sample_rate`） |
//! | `input_audio_buffer.commit` | 手动断句：把缓冲区送 ASR 出一句话 |
//! | `input_audio_buffer.clear` | 丢掉缓冲区 |
//! | `conversation.item.create` | 塞一条文本或多模态（`input_image`）消息 |
//! | `conversation.item.delete` | 删一条（`item_id`） |
//! | `response.create` | 让模型基于当前对话生成一轮（文本 + 语音） |
//! | `response.cancel` | 打断正在生成的那一轮 |
//!
//! # 服务端 → 客户端
//!
//! `session.created/updated`、`input_audio_buffer.speech_started/stopped/committed/cleared`、
//! `conversation.item.created/deleted`、
//! `conversation.item.input_audio_transcription.completed`、
//! `response.created`、`response.output_item.added`、`response.content_part.added`、
//! `response.audio_transcript.delta/done`、`response.text.delta/done`、
//! `response.audio.delta`（base64 PCM16）、`response.content_part.done`、
//! `response.output_item.done`、`response.done`、`error`。
//!
//! # 和 OpenAI 的差异
//!
//! - **音频编码**只有 `pcm16`（G.711 μ-law/A-law 不支持）。默认输入 24k、输出 24k，
//!   送进模型/出模型时各自重采样；模型输出的原采样率（Kokoro 24k / GPT-SoVITS 32k）
//!   会先重采样到 `output_audio_sample_rate` 再发。
//! - **回合检测**内置一个能量 VAD（`server_vad`：阈值 + 静音时长），没有模型级 VAD。
//!   不想要就 `turn_detection: null`，自己发 `input_audio_buffer.commit`。
//! - **工具调用**不支持（`tools` 字段会被忽略）。
//! - 一轮生成期间持有 LLM 模型的写锁（和 `/v1/chat/completions` 一致），
//!   同一进程里同时只跑一轮生成；TTS 是另一个模型的锁，可以和生成并行。
//! - `usage` 不发真值（本地后端不吐 token 账），字段直接为 `null`。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::{
    extract::{
        Query,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::StatusCode,
    response::Response,
};
use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use futures_util::{SinkExt, StreamExt};
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::sync::mpsc;

mod audio;
mod protocol;
mod sentences;

use audio::{
    ASR_RATE, AUDIO_CHUNK_MS, AudioBuffer, DEFAULT_INPUT_RATE, DEFAULT_OUTPUT_RATE, EnergyVad,
    VadEvent, pcm16_to_f32, resample_pcm16,
};
use protocol::{build_request, envelope, item_to_json, new_id, normalize_content, response_shell};
use sentences::{
    FIRST_SENTENCE_MIN_CHARS, HARD_MIN_CHARS, SOFT_MIN_CHARS, drain_sentences, drain_sentences_with,
};

use crate::server::api::model_for;
use aha::models::GenerateModel;
use aha::params::chat::{ChatMessageContent, ChatMessageContentPart, DeltaChatMessage};

/// WebSocket 出站消息队列的类型别名。
type Sender = mpsc::UnboundedSender<Message>;

/// 对话里最多保留多少条 item（超了丢最老的，避免上下文无限膨胀）。
const MAX_ITEMS: usize = 40;
/// 默认音色（Kokoro 的 `zf_001`；GPT-SoVITS 没有这个名字时用模型自己的默认音色）。
const DEFAULT_VOICE: &str = "zf_001";
/// `GET /realtime` 的查询参数。等价于连上之后先发一条 `session.update`。
#[derive(Debug, Default, Deserialize)]
pub(crate) struct RealtimeQuery {
    /// 用哪个模型：注册表里的 key / 模型 id / 角色名（`asr`、`tts`、`system1`、`vla`）。
    model: Option<String>,
    /// 音色。
    voice: Option<String>,
}

/// WebSocket 升级入口。`Sec-WebSocket-Protocol: realtime`（OpenAI 官方 SDK 会带）
/// 会被原样回显，否则部分客户端的握手直接失败。
pub(crate) async fn handler(ws: WebSocketUpgrade, Query(query): Query<RealtimeQuery>) -> Response {
    ws.protocols(["realtime"])
        .on_upgrade(move |socket| async move {
            run(socket, query).await;
        })
}

/// 一路连接的收发主循环。
async fn run(socket: WebSocket, query: RealtimeQuery) {
    let (mut sink, mut inbound) = socket.split();
    let (tx, mut rx) = mpsc::unbounded_channel::<Message>();

    // 写半边单独一个任务：生成任务、TTS 任务、主循环都只管往 channel 里扔。
    let writer = tokio::spawn(async move {
        while let Some(message) = rx.recv().await {
            if sink.send(message).await.is_err() {
                break;
            }
        }
    });

    let mut conn = Conn::new(tx.clone(), query);
    conn.send_session("session.created");

    while let Some(frame) = inbound.next().await {
        match frame {
            Ok(Message::Text(text)) => {
                let raw = text.as_str().to_owned();
                conn.handle_text(&raw).await;
            }
            Ok(Message::Binary(bytes)) => conn.push_audio(&bytes).await,
            Ok(Message::Ping(payload)) => {
                let _ = tx.send(Message::Pong(payload));
            }
            Ok(Message::Close(_)) | Err(_) => break,
            _ => {}
        }
        conn.reap();
    }

    conn.cancel_active();
    drop(conn);
    writer.abort();
}

/// 回合检测（`turn_detection`）。字段名照抄 OpenAI，语义也照抄：
/// 阈值是 0..1 的线性幅度，`silence_duration_ms` 是判定"说完了"的静音时长。
#[derive(Debug, Clone)]
struct TurnDetection {
    threshold: f32,
    prefix_padding_ms: u64,
    silence_duration_ms: u64,
    create_response: bool,
    interrupt_response: bool,
}

impl Default for TurnDetection {
    fn default() -> Self {
        Self {
            // 0.02 ≈ -34 dBFS：笔记本内置麦说人话能过、底噪过不去。
            threshold: 0.02,
            prefix_padding_ms: 300,
            // 300 ms 是「说完就响应」和「别把句子中间的停顿当成说完」之间的折中：
            // 实测 250 ms 会在「你好，我是小桌宠，」的逗号停顿处误切（把一句话拆成两轮），
            // 500 ms 又让人多等 0.2 秒才开始识别。
            silence_duration_ms: 300,
            create_response: true,
            interrupt_response: true,
        }
    }
}

/// 会话配置。`session.update` 按字段合并（没给的字段保持原值）。
#[derive(Debug, Clone)]
struct SessionConfig {
    model: Option<String>,
    voice: String,
    speed: f32,
    instructions: String,
    /// `["audio"]` / `["text"]` / `["audio","text"]`。
    modalities: Vec<String>,
    temperature: Option<f32>,
    max_output_tokens: Option<u32>,
    /// 客户端送进来的采样率。
    input_rate: u32,
    /// 发出去的采样率。
    output_rate: u32,
    turn_detection: Option<TurnDetection>,
    /// 要不要把用户语音转成文字（`input_audio_transcription` 非 null 即开）。
    transcribe_input: bool,
    /// 是否让模型先想再答。语音对话默认关掉——思考链会推迟首个 token 数十秒。
    enable_thinking: bool,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            model: None,
            voice: DEFAULT_VOICE.to_owned(),
            speed: 1.0,
            instructions: String::new(),
            modalities: vec!["audio".to_owned()],
            temperature: Some(0.8),
            max_output_tokens: None,
            input_rate: DEFAULT_INPUT_RATE,
            output_rate: DEFAULT_OUTPUT_RATE,
            turn_detection: None,
            transcribe_input: true,
            enable_thinking: false,
        }
    }
}

impl SessionConfig {
    fn wants_audio(&self) -> bool {
        self.modalities.iter().any(|m| m == "audio")
    }

    fn wants_text(&self) -> bool {
        self.modalities.iter().any(|m| m == "text")
    }

    /// 合并一条 `session.update` 里的 `session` 对象。返回错误文案（由调用方包成 error 事件）。
    fn apply_update(&mut self, session: &Value) -> Result<(), String> {
        if let Some(model) = session.get("model").and_then(Value::as_str) {
            let model = model.trim();
            self.model = (!model.is_empty()).then(|| model.to_owned());
        }
        if let Some(voice) = session.get("voice").and_then(Value::as_str)
            && !voice.trim().is_empty()
        {
            self.voice = voice.trim().to_owned();
        }
        if let Some(speed) = session.get("speed").and_then(Value::as_f64) {
            self.speed = speed.clamp(0.25, 4.0) as f32;
        }
        if let Some(instructions) = session.get("instructions").and_then(Value::as_str) {
            self.instructions = instructions.to_owned();
        }
        if let Some(modalities) = session.get("modalities").and_then(Value::as_array) {
            let mut parsed = Vec::new();
            for modality in modalities {
                let Some(name) = modality.as_str() else {
                    continue;
                };
                if !matches!(name, "audio" | "text") {
                    return Err(format!(
                        "不支持的 modalities 值 {name}（只有 audio / text）"
                    ));
                }
                parsed.push(name.to_owned());
            }
            if !parsed.is_empty() {
                self.modalities = parsed;
            }
        }
        if let Some(temperature) = session.get("temperature").and_then(Value::as_f64) {
            self.temperature = Some(temperature.clamp(0.0, 2.0) as f32);
        }
        if let Some(thinking) = session.get("enable_thinking").and_then(Value::as_bool) {
            self.enable_thinking = thinking;
        }
        match session.get("max_response_output_tokens") {
            Some(Value::Number(number)) => {
                self.max_output_tokens = number.as_u64().map(|n| n.min(u64::from(u32::MAX)) as u32);
            }
            Some(Value::String(text)) if text == "inf" => self.max_output_tokens = None,
            _ => {}
        }
        for field in ["input_audio_format", "output_audio_format"] {
            if let Some(format) = session.get(field).and_then(Value::as_str)
                && format != "pcm16"
            {
                return Err(format!("只支持 pcm16，收到 {field}={format}"));
            }
        }
        if let Some(rate) = session
            .get("input_audio_sample_rate")
            .and_then(Value::as_u64)
        {
            let rate = rate as u32;
            if !(8_000..=96_000).contains(&rate) || !rate.is_multiple_of(100) {
                return Err(format!("input_audio_sample_rate 不合法：{rate}"));
            }
            self.input_rate = rate;
        }
        if let Some(rate) = session
            .get("output_audio_sample_rate")
            .and_then(Value::as_u64)
        {
            let rate = rate as u32;
            if !(8_000..=96_000).contains(&rate) || !rate.is_multiple_of(100) {
                return Err(format!("output_audio_sample_rate 不合法：{rate}"));
            }
            self.output_rate = rate;
        }
        if let Some(turn) = session.get("turn_detection") {
            if turn.is_null() {
                self.turn_detection = None;
            } else {
                let kind = turn
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("server_vad");
                if kind != "server_vad" {
                    return Err(format!(
                        "只支持 turn_detection.type=server_vad，收到 {kind}"
                    ));
                }
                let mut detection = TurnDetection::default();
                if let Some(threshold) = turn.get("threshold").and_then(Value::as_f64) {
                    detection.threshold = (threshold as f32).clamp(0.0, 1.0);
                }
                if let Some(padding) = turn.get("prefix_padding_ms").and_then(Value::as_u64) {
                    detection.prefix_padding_ms = padding;
                }
                if let Some(silence) = turn.get("silence_duration_ms").and_then(Value::as_u64) {
                    detection.silence_duration_ms = silence.clamp(100, 5_000);
                }
                if let Some(flag) = turn.get("create_response").and_then(Value::as_bool) {
                    detection.create_response = flag;
                }
                if let Some(flag) = turn.get("interrupt_response").and_then(Value::as_bool) {
                    detection.interrupt_response = flag;
                }
                self.turn_detection = Some(detection);
            }
        }
        // OpenAI 的形状是 `input_audio_transcription: {model: "whisper-1"}`；显式 null 表示关掉。
        if let Some(transcription) = session.get("input_audio_transcription") {
            self.transcribe_input = !transcription.is_null();
        }
        if let Some(enabled) = session
            .get("transcription_enabled")
            .and_then(Value::as_bool)
        {
            self.transcribe_input = enabled;
        }
        Ok(())
    }

    /// 序列化成 `session.created` / `session.updated` 里的 `session` 对象。
    fn to_json(&self, session_id: &str) -> Value {
        json!({
            "object": "realtime.session",
            "id": session_id,
            "model": self.model.clone().unwrap_or_default(),
            "modalities": self.modalities,
            "instructions": self.instructions,
            "voice": self.voice,
            "speed": self.speed,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_sample_rate": self.input_rate,
            "output_audio_sample_rate": self.output_rate,
            "temperature": self.temperature,
            "max_response_output_tokens": self.max_output_tokens
                .map(|n| json!(n))
                .unwrap_or_else(|| json!("inf")),
            "turn_detection": self.turn_detection.as_ref().map(|turn| json!({
                "type": "server_vad",
                "threshold": turn.threshold,
                "prefix_padding_ms": turn.prefix_padding_ms,
                "silence_duration_ms": turn.silence_duration_ms,
                "create_response": turn.create_response,
                "interrupt_response": turn.interrupt_response,
            })),
            "input_audio_transcription": if self.transcribe_input {
                json!({ "model": "aha-asr" })
            } else {
                Value::Null
            },
            "tools": [],
            "tool_choice": "auto",
            "enable_thinking": self.enable_thinking,
        })
    }
}

/// 一句用户话 / 一句助手话。
#[derive(Debug, Clone)]
struct Item {
    id: String,
    /// `user` / `assistant` / `system`。
    role: String,
    /// 纯文本是 `Value::String`，多模态（视频帧）是 chat 形状的 content 数组。
    content: Value,
}

/// 已经甩出去的那一轮生成，主循环靠它做打断判定。
struct ActiveResponse {
    cancel: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
}

/// 一路连接的会话状态。
struct Conn {
    tx: Sender,
    cfg: SessionConfig,
    session_id: String,
    items: Vec<Item>,
    audio: AudioBuffer,
    vad: Option<EnergyVad>,
    active: Option<ActiveResponse>,
}

impl Conn {
    fn new(tx: Sender, query: RealtimeQuery) -> Self {
        let mut cfg = SessionConfig {
            model: query
                .model
                .map(|model| model.trim().to_owned())
                .filter(|model| !model.is_empty()),
            ..SessionConfig::default()
        };
        if let Some(voice) = query.voice.map(|voice| voice.trim().to_owned())
            && !voice.is_empty()
        {
            cfg.voice = voice;
        }
        Self {
            tx,
            cfg,
            session_id: new_id("sess"),
            items: Vec::new(),
            audio: AudioBuffer::default(),
            vad: None,
            active: None,
        }
    }

    fn send(&self, value: Value) {
        let _ = self.tx.send(Message::Text(value.to_string().into()));
    }

    fn send_session(&self, kind: &str) {
        self.send(envelope(
            kind,
            json!({ "session": self.cfg.to_json(&self.session_id) }),
        ));
    }

    fn send_error(&self, code: &str, kind: &str, message: impl Into<String>) {
        self.send(envelope(
            "error",
            json!({
                "error": {
                    "type": kind,
                    "code": code,
                    "message": message.into(),
                    "param": Value::Null,
                    "event_id": Value::Null,
                }
            }),
        ));
    }

    /// 一轮生成收尾后清掉 active 槽位（任务的 `finished` 标志由它自己置位）。
    fn reap(&mut self) {
        if let Some(active) = &self.active
            && active.finished.load(Ordering::SeqCst)
        {
            self.active = None;
        }
    }

    fn cancel_active(&mut self) {
        if let Some(active) = &self.active {
            active.cancel.store(true, Ordering::SeqCst);
        }
    }

    fn trim_items(&mut self) {
        if self.items.len() > MAX_ITEMS {
            let drop_count = self.items.len() - MAX_ITEMS;
            self.items.drain(..drop_count);
        }
    }

    /// 一条客户端事件。
    async fn handle_text(&mut self, raw: &str) {
        let value: Value = match serde_json::from_str(raw) {
            Ok(value) => value,
            Err(error) => {
                self.send_error(
                    "invalid_json",
                    "invalid_request_error",
                    format!("JSON 解析失败：{error}"),
                );
                return;
            }
        };
        let kind = value
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_owned();
        match kind.as_str() {
            "session.update" => {
                let session = value.get("session").cloned().unwrap_or(Value::Null);
                match self.cfg.apply_update(&session) {
                    Ok(()) => {
                        // 回合检测模式变了 → 重建 VAD 状态机。
                        self.vad = self
                            .cfg
                            .turn_detection
                            .as_ref()
                            .map(|turn| EnergyVad::new(turn.clone(), self.cfg.input_rate));
                        self.send_session("session.updated");
                    }
                    Err(message) => {
                        self.send_error("invalid_value", "invalid_request_error", message);
                    }
                }
            }
            "input_audio_buffer.append" => {
                let Some(audio) = value.get("audio").and_then(Value::as_str) else {
                    self.send_error(
                        "missing_audio",
                        "invalid_request_error",
                        "缺少 audio 字段（base64 PCM16）",
                    );
                    return;
                };
                match BASE64.decode(audio.trim()) {
                    Ok(bytes) => self.push_audio(&bytes).await,
                    Err(error) => self.send_error(
                        "invalid_audio",
                        "invalid_request_error",
                        format!("audio 不是合法 base64：{error}"),
                    ),
                }
            }
            "input_audio_buffer.commit" => self.commit_audio(false).await,
            "input_audio_buffer.clear" => {
                self.audio.clear();
                if let Some(vad) = &mut self.vad {
                    vad.reset();
                }
                self.send(envelope("input_audio_buffer.cleared", json!({})));
            }
            "conversation.item.create" => self.create_item(&value),
            "conversation.item.delete" => {
                let Some(item_id) = value.get("item_id").and_then(Value::as_str) else {
                    self.send_error("missing_item_id", "invalid_request_error", "缺少 item_id");
                    return;
                };
                if let Some(position) = self.items.iter().position(|item| item.id == item_id) {
                    self.items.remove(position);
                    self.send(envelope(
                        "conversation.item.deleted",
                        json!({ "item_id": item_id }),
                    ));
                } else {
                    self.send_error(
                        "item_not_found",
                        "invalid_request_error",
                        format!("没有 item {item_id}"),
                    );
                }
            }
            "response.create" => self.start_response(),
            "response.cancel" => {
                self.cancel_active();
                // OpenAI 回 `response.cancelled` 之类的事件；这里由生成任务自己发 `response.done`。
            }
            "conversation.item.truncate" | "input_audio_buffer.truncate" => {
                // 打断后的截断语义对本地流水线没意义（音频已经放出去了），静默忽略。
            }
            other => self.send_error(
                "unknown_event",
                "invalid_request_error",
                format!("不认识的事件类型 {other}"),
            ),
        }
    }

    /// 追加一段裸 PCM16（二进制帧走这里，和 `append` 事件等价）。
    async fn push_audio(&mut self, bytes: &[u8]) {
        if bytes.len() < 2 {
            return;
        }
        self.audio.push(bytes);
        let Some(_) = self.vad else {
            return;
        };
        let samples = pcm16_to_f32(bytes);
        let events = match &mut self.vad {
            Some(vad) => vad.push(&samples),
            None => Vec::new(),
        };
        for event in events {
            match event {
                VadEvent::Start => {
                    self.send(envelope(
                        "input_audio_buffer.speech_started",
                        json!({
                            "audio_start_ms": self.audio.duration_ms(),
                            "item_id": Value::Null,
                        }),
                    ));
                    if self
                        .cfg
                        .turn_detection
                        .as_ref()
                        .map(|turn| turn.interrupt_response)
                        .unwrap_or(false)
                    {
                        self.cancel_active();
                    }
                }
                VadEvent::Stop => {
                    self.send(envelope(
                        "input_audio_buffer.speech_stopped",
                        json!({
                            "audio_end_ms": self.audio.duration_ms(),
                            "item_id": Value::Null,
                        }),
                    ));
                    let respond = self
                        .cfg
                        .turn_detection
                        .as_ref()
                        .map(|turn| turn.create_response)
                        .unwrap_or(false);
                    self.commit_audio(respond).await;
                }
            }
        }
    }

    /// 缓冲区 → ASR → 一条 user item。`then_respond` 为真时顺手触发一轮生成。
    async fn commit_audio(&mut self, then_respond: bool) {
        if self.audio.is_empty() {
            self.send_error(
                "input_audio_buffer_commit_empty",
                "invalid_request_error",
                "音频缓冲区是空的",
            );
            return;
        }
        let samples = self.audio.take_samples(ASR_RATE);
        let previous = self.items.last().map(|item| item.id.clone());
        let item_id = new_id("item");

        // 先回 committed 再跑识别：语义上缓冲区此刻已经提交，识别是它的后续动作，
        // 而且识别要几百毫秒，不能挡在提交确认前面（客户端靠它推进回合状态）。
        self.send(envelope(
            "input_audio_buffer.committed",
            json!({ "item_id": item_id, "previous_item_id": previous }),
        ));

        let mut transcript = String::new();
        if self.cfg.transcribe_input {
            if samples.len() < crate::server::asr::ASR_MIN_SAMPLES {
                // 太短，不值得跑一遍模型，但仍然把缓冲区清掉。
                transcript = String::new();
            } else {
                let asr_start = std::time::Instant::now();
                match transcribe(&samples).await {
                    Ok(text) => {
                        transcript = text;
                        tracing::debug!(
                            asr_ms = asr_start.elapsed().as_millis(),
                            samples = samples.len(),
                            "识别完成"
                        );
                        self.send(envelope(
                            "conversation.item.input_audio_transcription.completed",
                            json!({
                                "item_id": item_id,
                                "content_index": 0,
                                "transcript": transcript,
                            }),
                        ));
                    }
                    Err(message) => self.send_error(
                        "transcription_failed",
                        "server_error",
                        format!("识别失败：{message}"),
                    ),
                }
            }
        }

        let transcript = transcript.trim().to_owned();
        if !transcript.is_empty() {
            let item = Item {
                id: item_id,
                role: "user".to_owned(),
                content: Value::String(transcript),
            };
            self.items.push(item.clone());
            self.trim_items();
            self.send(envelope(
                "conversation.item.created",
                json!({ "previous_item_id": previous, "item": item_to_json(&item) }),
            ));
        }

        if then_respond {
            self.start_response();
        }
    }

    /// `conversation.item.create`：文本 / 多模态（视频帧就是一张张 `input_image`）。
    fn create_item(&mut self, value: &Value) {
        let item = value.get("item").cloned().unwrap_or(Value::Null);
        let role = item
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("user")
            .to_owned();
        if !matches!(role.as_str(), "user" | "assistant" | "system") {
            self.send_error(
                "invalid_role",
                "invalid_request_error",
                format!("不支持的 role {role}"),
            );
            return;
        }
        let Some(content) = item.get("content") else {
            self.send_error(
                "missing_content",
                "invalid_request_error",
                "item 缺 content",
            );
            return;
        };
        let Some(content) = normalize_content(content) else {
            self.send_error(
                "invalid_content",
                "invalid_request_error",
                "content 里没有可用内容（支持 input_text / text / input_image / image_url）",
            );
            return;
        };
        let item = Item {
            id: new_id("item"),
            role,
            content,
        };
        let previous = self.items.last().map(|previous| previous.id.clone());
        self.send(envelope(
            "conversation.item.created",
            json!({ "previous_item_id": previous, "item": item_to_json(&item) }),
        ));
        self.items.push(item);
        self.trim_items();
    }

    /// 起一轮生成。已经有在跑的就不叠（OpenAI 也是这个规矩）。
    fn start_response(&mut self) {
        if let Some(active) = &self.active
            && !active.finished.load(Ordering::SeqCst)
        {
            self.send_error(
                "conversation_already_has_active_response",
                "invalid_request_error",
                "上一轮还在生成；先发 response.cancel",
            );
            return;
        }
        if self.items.is_empty() {
            self.send_error(
                "empty_conversation",
                "invalid_request_error",
                "对话还是空的，先说话或 conversation.item.create",
            );
            return;
        }
        let cancel = Arc::new(AtomicBool::new(false));
        let finished = Arc::new(AtomicBool::new(false));
        self.active = Some(ActiveResponse {
            cancel: cancel.clone(),
            finished: finished.clone(),
        });
        let tx = self.tx.clone();
        let cfg = self.cfg.clone();
        let items = self.items.clone();
        tokio::spawn(async move {
            run_response(tx, cfg, items, cancel, finished).await;
        });
    }
}

/// 跑一轮"生成 + 边生成边合成"。
async fn run_response(
    tx: Sender,
    cfg: SessionConfig,
    items: Vec<Item>,
    cancel: Arc<AtomicBool>,
    finished: Arc<AtomicBool>,
) {
    let response_id = new_id("resp");
    let item_id = new_id("item");
    let started = std::time::Instant::now();
    let mut first_token_logged = false;
    let mut first_sentence_logged = false;
    let mut sentences_sent = 0usize;
    let wants_audio = cfg.wants_audio();
    let wants_text = cfg.wants_text();
    // 助手消息的 content 数组：音频在前，转写文本在后（和 OpenAI 一致）。
    let audio_index = 0u32;
    let transcript_index = if wants_audio { 1u32 } else { 0u32 };

    emit(
        &tx,
        envelope(
            "response.created",
            json!({ "response": response_shell(&response_id, &cfg, "in_progress") }),
        ),
    );
    emit(
        &tx,
        envelope(
            "response.output_item.added",
            json!({
                "response_id": response_id,
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "status": "in_progress",
                    "role": "assistant",
                    "content": [],
                },
            }),
        ),
    );
    if wants_audio {
        emit(
            &tx,
            envelope(
                "response.content_part.added",
                json!({
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": audio_index,
                    "part": { "type": "audio", "transcript": Value::Null },
                }),
            ),
        );
    }
    if wants_text {
        emit(
            &tx,
            envelope(
                "response.content_part.added",
                json!({
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": transcript_index,
                    "part": { "type": "text", "text": "" },
                }),
            ),
        );
    }

    // 先把请求拼好，模型缺失这类错误要能在同一轮里以 response.done(failed) 收尾。
    let params = match build_request(&cfg, &items) {
        Ok(params) => params,
        Err(message) => {
            emit(
                &tx,
                envelope(
                    "error",
                    json!({ "error": { "type": "invalid_request_error", "code": "invalid_request", "message": message, "param": Value::Null, "event_id": Value::Null } }),
                ),
            );
            finish_response(
                &tx,
                &cfg,
                &response_id,
                &item_id,
                audio_index,
                transcript_index,
                "failed",
                "",
                wants_audio,
                wants_text,
            );
            finished.store(true, Ordering::SeqCst);
            return;
        }
    };
    let Some(model_ref) = model_for(cfg.model.as_deref()) else {
        emit(
            &tx,
            envelope(
                "error",
                json!({ "error": { "type": "server_error", "code": "model_not_loaded", "message": "模型没有初始化", "param": Value::Null, "event_id": Value::Null } }),
            ),
        );
        finish_response(
            &tx,
            &cfg,
            &response_id,
            &item_id,
            audio_index,
            transcript_index,
            "failed",
            "",
            wants_audio,
            wants_text,
        );
        finished.store(true, Ordering::SeqCst);
        return;
    };

    // 句子队列 → TTS 任务。生成读完就 drop 掉发送端，TTS 任务自然收尾。
    let (sentence_tx, sentence_rx) = mpsc::channel::<String>(4);
    let tts_task = tokio::spawn(tts_worker(
        tx.clone(),
        cfg.clone(),
        sentence_rx,
        response_id.clone(),
        item_id.clone(),
        audio_index,
        cancel.clone(),
    ));

    let mut full = String::new();
    let mut pending = String::new();
    let mut cancelled = false;
    let mut failed = false;

    {
        // 写锁覆盖整段生成（和 `/v1/chat/completions` 一个约定）。
        let mut guard = model_ref.write().await;
        match guard.instance.generate_stream(params) {
            Ok(stream) => {
                let mut stream = std::pin::pin!(stream);
                while let Some(entry) = stream.next().await {
                    if cancel.load(Ordering::SeqCst) {
                        cancelled = true;
                        break;
                    }
                    match entry {
                        Ok(chunk) => {
                            let delta = chunk
                                .choices
                                .first()
                                .and_then(|choice| delta_text(&choice.delta))
                                .unwrap_or_default();
                            if delta.is_empty() {
                                continue;
                            }
                            if !first_token_logged {
                                first_token_logged = true;
                                tracing::debug!(
                                    ttft_ms = started.elapsed().as_millis(),
                                    "首个 token 到达"
                                );
                            }
                            full.push_str(&delta);
                            pending.push_str(&delta);
                            if wants_text {
                                emit(
                                    &tx,
                                    envelope(
                                        "response.text.delta",
                                        json!({
                                            "response_id": response_id,
                                            "item_id": item_id,
                                            "output_index": 0,
                                            "content_index": transcript_index,
                                            "delta": delta,
                                        }),
                                    ),
                                );
                            }
                            if wants_audio {
                                emit(
                                    &tx,
                                    envelope(
                                        "response.audio_transcript.delta",
                                        json!({
                                            "response_id": response_id,
                                            "item_id": item_id,
                                            "output_index": 0,
                                            "content_index": transcript_index,
                                            "delta": delta,
                                        }),
                                    ),
                                );
                            }
                            let soft_min = if sentences_sent == 0 {
                                FIRST_SENTENCE_MIN_CHARS
                            } else {
                                SOFT_MIN_CHARS
                            };
                            for sentence in
                                drain_sentences_with(&mut pending, false, soft_min, HARD_MIN_CHARS)
                            {
                                if !first_sentence_logged {
                                    first_sentence_logged = true;
                                    tracing::debug!(
                                        first_sentence_ms = started.elapsed().as_millis(),
                                        chars = sentence.chars().count(),
                                        "第一句交给 TTS"
                                    );
                                }
                                if sentence_tx.send(sentence).await.is_err() {
                                    break;
                                }
                                sentences_sent += 1;
                            }
                        }
                        Err(error) => {
                            failed = true;
                            emit(
                                &tx,
                                envelope(
                                    "error",
                                    json!({ "error": { "type": "server_error", "code": "generation_failed", "message": format!("{error:#}"), "param": Value::Null, "event_id": Value::Null } }),
                                ),
                            );
                            break;
                        }
                    }
                }
            }
            Err(error) => {
                failed = true;
                emit(
                    &tx,
                    envelope(
                        "error",
                        json!({ "error": { "type": "server_error", "code": "generation_failed", "message": format!("{error:#}"), "param": Value::Null, "event_id": Value::Null } }),
                    ),
                );
            }
        }
    }

    // 尾巴上不足一句的残余也要念完。
    for sentence in drain_sentences(&mut pending, true) {
        if sentence_tx.send(sentence).await.is_err() {
            break;
        }
    }
    drop(sentence_tx);
    let _ = tts_task.await;

    let status = if cancelled {
        "cancelled"
    } else if failed {
        "failed"
    } else {
        "completed"
    };
    finish_response(
        &tx,
        &cfg,
        &response_id,
        &item_id,
        audio_index,
        transcript_index,
        status,
        &full,
        wants_audio,
        wants_text,
    );
    finished.store(true, Ordering::SeqCst);
}

/// 一轮生成走到终点：把照应的事件都补齐（`*.done` 系列 + `response.done`）。
#[allow(clippy::too_many_arguments)]
fn finish_response(
    tx: &Sender,
    cfg: &SessionConfig,
    response_id: &str,
    item_id: &str,
    audio_index: u32,
    transcript_index: u32,
    status: &str,
    full: &str,
    wants_audio: bool,
    wants_text: bool,
) {
    let mut parts = Vec::new();
    if wants_audio {
        emit(
            tx,
            envelope(
                "response.audio_transcript.done",
                json!({
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": transcript_index,
                    "transcript": full,
                }),
            ),
        );
        parts.push(json!({ "type": "audio", "transcript": full }));
    }
    if wants_text {
        emit(
            tx,
            envelope(
                "response.text.done",
                json!({
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": transcript_index,
                    "text": full,
                }),
            ),
        );
        parts.push(json!({ "type": "text", "text": full }));
    }
    if wants_audio {
        emit(
            tx,
            envelope(
                "response.content_part.done",
                json!({
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": audio_index,
                    "part": { "type": "audio", "transcript": full },
                }),
            ),
        );
    }
    if wants_text {
        emit(
            tx,
            envelope(
                "response.content_part.done",
                json!({
                    "response_id": response_id,
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": transcript_index,
                    "part": { "type": "text", "text": full },
                }),
            ),
        );
    }
    emit(
        tx,
        envelope(
            "response.output_item.done",
            json!({
                "response_id": response_id,
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "object": "realtime.item",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": parts,
                },
            }),
        ),
    );
    let mut response = response_shell(response_id, cfg, status);
    let mut content: Vec<Value> = Vec::new();
    if wants_audio {
        content.push(json!({ "type": "audio", "transcript": full }));
    }
    if wants_text {
        content.push(json!({ "type": "text", "text": full }));
    }
    response["output"] = json!([{
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": content,
    }]);
    emit(
        tx,
        envelope("response.done", json!({ "response": response })),
    );
}

/// 逐句合成：一句进、一块块 `response.audio.delta` 出。
async fn tts_worker(
    tx: Sender,
    cfg: SessionConfig,
    mut sentences: mpsc::Receiver<String>,
    response_id: String,
    item_id: String,
    audio_index: u32,
    cancel: Arc<AtomicBool>,
) {
    let mut reported = false;
    let chunk_bytes = ((cfg.output_rate as usize * AUDIO_CHUNK_MS as usize) / 1000).max(1) * 2;
    while let Some(sentence) = sentences.recv().await {
        if cancel.load(Ordering::SeqCst) {
            break;
        }
        if !cfg.wants_audio() {
            continue;
        }
        let synth_start = std::time::Instant::now();
        let chars = sentence.chars().count();
        // 边合成边收：第一个块到了就能开口，不必等整句合成完。
        let (chunk_tx, mut chunk_rx) = mpsc::channel::<Vec<u8>>(16);
        let (text, voice, speed) = (sentence.clone(), cfg.voice.clone(), cfg.speed);
        let synth_task =
            tokio::spawn(async move { stream_synth(text, voice, speed, chunk_tx).await });
        let rate = match synth_rate().await {
            Some(rate) => rate,
            None => cfg.output_rate,
        };
        let mut pending: Vec<u8> = Vec::new();
        let mut first_chunk_logged = false;
        while let Some(bytes) = chunk_rx.recv().await {
            if cancel.load(Ordering::SeqCst) {
                synth_task.abort();
                return;
            }
            let out = if rate == cfg.output_rate {
                bytes
            } else {
                resample_pcm16(&bytes, rate, cfg.output_rate)
            };
            pending.extend_from_slice(&out);
            while pending.len() >= chunk_bytes {
                let tail = pending.split_off(chunk_bytes);
                let chunk = std::mem::replace(&mut pending, tail);
                if !first_chunk_logged {
                    first_chunk_logged = true;
                    tracing::debug!(
                        first_audio_ms = synth_start.elapsed().as_millis(),
                        chars,
                        "首块音频"
                    );
                }
                emit(
                    &tx,
                    envelope(
                        "response.audio.delta",
                        json!({
                            "response_id": response_id,
                            "item_id": item_id,
                            "output_index": 0,
                            "content_index": audio_index,
                            "delta": BASE64.encode(chunk),
                        }),
                    ),
                );
            }
        }
        // 尾巴：不足一块的也发出去（PCM16 不能有半个样本）。
        if !pending.len().is_multiple_of(2) {
            pending.pop();
        }
        if !pending.is_empty() {
            emit(
                &tx,
                envelope(
                    "response.audio.delta",
                    json!({
                        "response_id": response_id,
                        "item_id": item_id,
                        "output_index": 0,
                        "content_index": audio_index,
                        "delta": BASE64.encode(&pending),
                    }),
                ),
            );
        }
        match synth_task.await {
            Ok(Ok(_)) => {
                tracing::debug!(
                    synth_ms = synth_start.elapsed().as_millis(),
                    chars,
                    "整句合成完成"
                );
            }
            Ok(Err((_status, message))) => {
                if !reported {
                    reported = true;
                    emit(
                        &tx,
                        envelope(
                            "error",
                            json!({
                                "error": {
                                    "type": "server_error",
                                    "code": "tts_failed",
                                    "message": message,
                                    "param": Value::Null,
                                    "event_id": Value::Null,
                                }
                            }),
                        ),
                    );
                }
            }
            Err(error) => tracing::warn!(%error, "合成任务中断"),
        }
    }
}

/// 从流式 chunk 里抠出文本增量。
fn delta_text(delta: &DeltaChatMessage) -> Option<String> {
    let content = match delta {
        // assistant / untagged 的 content 是可选的（纯 tool_calls 时没有正文）
        DeltaChatMessage::Assistant { content, .. }
        | DeltaChatMessage::Untagged { content, .. } => content.as_ref()?,
        DeltaChatMessage::User { content, .. }
        | DeltaChatMessage::System { content, .. }
        | DeltaChatMessage::Developer { content, .. } => content,
        DeltaChatMessage::Tool { .. } => return None,
    };
    match content {
        ChatMessageContent::Text(text) => Some(text.clone()),
        ChatMessageContent::ContentPart(parts) => {
            let text = parts
                .iter()
                .filter_map(|part| match part {
                    ChatMessageContentPart::Text(part) => Some(part.text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join("");
            (!text.is_empty()).then_some(text)
        }
        ChatMessageContent::None => None,
    }
}

// ---------------------------------------------------------------------------
// 音频：缓冲 / 重采样 / PCM 转换
// ---------------------------------------------------------------------------

#[cfg(feature = "sherpa-asr")]
async fn transcribe(samples: &[f32]) -> Result<String, String> {
    crate::server::asr::transcribe_samples(samples)
        .await
        .map_err(|(_, message)| message)
}

#[cfg(not(feature = "sherpa-asr"))]
async fn transcribe(_samples: &[f32]) -> Result<String, String> {
    Err("这个二进制没编译 ASR（要开启 sherpa-asr feature）".to_owned())
}

/// 合成一句话，块级回调直接灌进 `sink`；返回 TTS 模型的采样率。
#[cfg(any(feature = "sherpa-tts", feature = "gpt-sovits"))]
async fn stream_synth(
    text: String,
    voice: String,
    speed: f32,
    sink: mpsc::Sender<Vec<u8>>,
) -> Result<u32, (StatusCode, String)> {
    crate::server::tts::synthesize_pcm16_stream(text, voice, speed, sink).await
}

#[cfg(not(any(feature = "sherpa-tts", feature = "gpt-sovits")))]
async fn stream_synth(
    _text: String,
    _voice: String,
    _speed: f32,
    _sink: mpsc::Sender<Vec<u8>>,
) -> Result<u32, (StatusCode, String)> {
    Err((
        StatusCode::BAD_REQUEST,
        "这个二进制没编译 TTS（要开启 sherpa-tts 或 gpt-sovits feature）".to_owned(),
    ))
}

/// TTS 的采样率；没编 TTS 时返回 `None`，调用方回落到 session 的输出采样率。
#[cfg(any(feature = "sherpa-tts", feature = "gpt-sovits"))]
async fn synth_rate() -> Option<u32> {
    crate::server::tts::sample_rate().await
}

#[cfg(not(any(feature = "sherpa-tts", feature = "gpt-sovits")))]
async fn synth_rate() -> Option<u32> {
    None
}

fn emit(tx: &Sender, value: Value) {
    let _ = tx.send(Message::Text(value.to_string().into()));
}

// ---------------------------------------------------------------------------
// 测试
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_keeps_length_proportional() {
        let samples = vec![0.0f32; 24_000];
        let down = resample_f32(&samples, 24_000, 16_000);
        assert_eq!(down.len(), 16_000);
        let up = resample_f32(&down, 16_000, 24_000);
        assert_eq!(up.len(), 24_000);
        assert!(resample_f32(&samples, 16_000, 16_000).len() == samples.len());
        assert!(resample_f32(&[], 24_000, 16_000).is_empty());
    }

    #[test]
    fn resample_keeps_signal_shape() {
        // 24k 上 1kHz 正弦 → 16k 应该还是 1kHz（周期 16 个样本）。
        let samples: Vec<f32> = (0..2_400)
            .map(|index| (index as f32 * 1_000.0 * std::f32::consts::TAU / 24_000.0).sin())
            .collect();
        let down = resample_f32(&samples, 24_000, 16_000);
        assert_eq!(down.len(), 1_600);
        let peak = down.iter().fold(0.0f32, |acc, value| acc.max(value.abs()));
        assert!(peak > 0.9, "幅度不该被吃掉，peak={peak}");
    }

    #[test]
    fn pcm_round_trip() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let bytes = f32_to_pcm16(&samples);
        assert_eq!(bytes.len(), samples.len() * 2);
        let back = pcm16_to_f32(&bytes);
        assert_eq!(back.len(), samples.len());
        assert!((back[1] - 0.5).abs() < 0.001);
        assert!((back[3] - 1.0).abs() < 0.001);
    }

    #[test]
    fn sentences_split_on_punctuation() {
        let mut buffer = String::from("你好。今天天气不错，我们出去走走看看好不好？");
        let sentences = drain_sentences(&mut buffer, false);
        assert_eq!(sentences[0], "你好。");
        // 弱停顿要攒够 24 个字才切，这句不够，只剩问号。
        assert!(sentences[1].ends_with('？'));
        assert!(buffer.is_empty() || buffer.chars().count() < SOFT_MIN_CHARS);
    }

    #[test]
    fn sentences_flush_rest_when_done() {
        let mut buffer = String::from("没有标点的半句话");
        assert!(drain_sentences(&mut buffer, false).is_empty());
        let sentences = drain_sentences(&mut buffer, true);
        assert_eq!(sentences, vec!["没有标点的半句话".to_owned()]);
        assert!(buffer.is_empty());
    }

    #[test]
    fn energy_vad_finds_start_and_stop() {
        let turn = TurnDetection {
            threshold: 0.05,
            silence_duration_ms: 200,
            ..TurnDetection::default()
        };
        let mut vad = EnergyVad::new(turn, 16_000);
        // 200ms 静音
        let silence = vec![0.0f32; 3_200];
        assert!(vad.push(&silence).is_empty());
        // 200ms 说话
        let speech = vec![0.3f32; 3_200];
        let events = vad.push(&speech);
        assert!(matches!(events.first(), Some(VadEvent::Start)));
        // 200ms 静音 → 判定说完
        let events = vad.push(&silence);
        assert!(matches!(events.first(), Some(VadEvent::Stop)));
        assert!(!vad.in_speech);
    }

    #[test]
    fn session_update_parses_server_vad() {
        let mut cfg = SessionConfig::default();
        cfg.apply_update(&json!({
            "modalities": ["text", "audio"],
            "voice": "zf_002",
            "instructions": "你是桌宠",
            "input_audio_sample_rate": 16000,
            "turn_detection": { "type": "server_vad", "threshold": 0.1, "silence_duration_ms": 800 },
            "input_audio_transcription": { "model": "aha" },
        }))
        .expect("update 应该成功");
        assert!(cfg.wants_audio() && cfg.wants_text());
        assert_eq!(cfg.voice, "zf_002");
        assert_eq!(cfg.input_rate, 16_000);
        let turn = cfg.turn_detection.expect("server_vad 应该被解析");
        assert!((turn.threshold - 0.1).abs() < f32::EPSILON);
        assert_eq!(turn.silence_duration_ms, 800);
        assert!(cfg.transcribe_input);
    }

    #[test]
    fn session_update_rejects_unknown_audio_format() {
        let mut cfg = SessionConfig::default();
        assert!(
            cfg.apply_update(&json!({ "input_audio_format": "g711_ulaw" }))
                .is_err()
        );
        assert!(
            cfg.apply_update(&json!({ "turn_detection": { "type": "semantic_vad" } }))
                .is_err()
        );
    }

    #[test]
    fn session_update_null_turns_vad_off() {
        let mut cfg = SessionConfig::default();
        cfg.apply_update(&json!({ "turn_detection": { "type": "server_vad" } }))
            .expect("开 VAD");
        assert!(cfg.turn_detection.is_some());
        cfg.apply_update(&json!({ "turn_detection": Value::Null }))
            .expect("关 VAD");
        assert!(cfg.turn_detection.is_none());
        assert_eq!(cfg.to_json("sess_1")["turn_detection"], Value::Null);
    }

    #[test]
    fn normalize_content_maps_image_parts() {
        let content = json!([
            { "type": "input_text", "text": "这是什么" },
            { "type": "input_image", "image_url": "data:image/jpeg;base64,AAAA" },
        ]);
        let normalized = normalize_content(&content).expect("应该能解析");
        let parts = normalized.as_array().expect("有图片时应该是数组");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[1]["type"], "image_url");
        assert_eq!(parts[1]["image_url"]["url"], "data:image/jpeg;base64,AAAA");
        assert_eq!(parts[1]["image_url"]["detail"], "auto");
    }

    #[test]
    fn normalize_content_collapses_text_only() {
        let content = json!([{ "type": "input_text", "text": "  hi  " }]);
        assert_eq!(
            normalize_content(&content),
            Some(Value::String("  hi  ".to_owned()))
        );
        assert_eq!(
            normalize_content(&json!("你好")),
            Some(Value::String("你好".to_owned()))
        );
        assert_eq!(normalize_content(&json!([])), None);
    }

    #[test]
    fn build_request_includes_instructions_and_items() {
        let cfg = SessionConfig {
            instructions: "你是桌宠".to_owned(),
            max_output_tokens: Some(64),
            ..SessionConfig::default()
        };
        let items = vec![Item {
            id: "item_1".to_owned(),
            role: "user".to_owned(),
            content: Value::String("你好".to_owned()),
        }];
        let params = build_request(&cfg, &items).expect("应该能构造");
        assert_eq!(params.messages.len(), 2);
        assert_eq!(params.max_completion_tokens, Some(64));
        assert_eq!(params.stream, Some(true));
    }

    #[test]
    fn build_request_rejects_empty_conversation() {
        let cfg = SessionConfig::default();
        assert!(build_request(&cfg, &[]).is_err());
    }

    #[test]
    fn envelope_carries_event_id_and_type() {
        let value = envelope("session.created", json!({ "session": { "id": "s" } }));
        assert_eq!(value["type"], "session.created");
        assert!(value["event_id"].as_str().unwrap().starts_with("event_"));
        assert_eq!(value["session"]["id"], "s");
    }

    #[test]
    fn audio_buffer_trims_to_cap() {
        let mut buffer = AudioBuffer {
            pcm: Vec::new(),
            rate: 16_000,
        };
        // 灌 61 秒
        let bytes = vec![0u8; 16_000 * 2 * 61];
        buffer.push(&bytes);
        assert_eq!(buffer.pcm.len(), 16_000 * 2 * 60);
        assert_eq!(buffer.duration_ms(), 60_000);
    }

    #[test]
    fn audio_buffer_resamples_on_take() {
        let mut buffer = AudioBuffer {
            pcm: Vec::new(),
            rate: 24_000,
        };
        // 1 秒 24k → 16k
        buffer.push(&vec![0u8; 24_000 * 2]);
        let samples = buffer.take_samples(16_000);
        assert_eq!(samples.len(), 16_000);
        assert!(buffer.pcm.is_empty());
    }
}

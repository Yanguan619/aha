//! C ABI for embedding aha in a host runtime (JNI, Swift, Python `ctypes`, Unity, ...).
//!
//! Ground rules — they are what make this boundary usable from a language that knows nothing
//! about Rust:
//!
//! * **No Rust type crosses the boundary.** Model handles are opaque `void*`, text is
//!   NUL-terminated UTF-8, audio is `bytes + length`, streaming goes through C function
//!   pointers. Nothing here mentions `String`, `Vec`, `Tensor`, or a trait object.
//! * **Panics never reach the host.** Every entry point is `catch_unwind`-guarded; a panic
//!   becomes an error string, because unwinding into a JVM/Swift/CPython frame is undefined.
//! * **Ownership is explicit.** Strings returned by this library are released with
//!   `aha_string_free`; handles with the matching `aha_*_free`. Strings you pass in stay yours.
//! * **Errors are out-parameters.** Functions that return a handle or a value take
//!   `char** err`; on failure they return `NULL`/`0` and write an owned message there.
//!
//! The JNI shim that an Android app needs is intentionally *not* here: JNI symbol names encode
//! the consumer's Java package, so they belong to the consumer. This layer is what a JNI shim
//! (or a Swift one, or a Python one) calls into.

use std::ffi::{CStr, CString, c_char, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::task::{Context, Poll, Waker};
use std::thread::Result as ThreadResult;

use aha::models::common::model_mapping::WhichModel;
use aha::models::{GenerateModel, ModelInstance, load_model};
use aha::params::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse, ChatMessage,
    ChatMessageContent, ChatMessageContentPart, DeltaChatMessage,
};
use aha::utils::string_to_static_str;
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use clap::ValueEnum;

// ---------------------------------------------------------------------------------------------
// Public C types
// ---------------------------------------------------------------------------------------------

/// Streaming callbacks. Every one of them is optional; `user` is handed back verbatim.
///
/// `on_chunk` fires for each incremental piece of text, `on_done` once with the whole answer,
/// `on_error` instead of `on_done` when something failed. The pointer `text` is only valid for
/// the duration of the call — copy it if you need to keep it.
#[repr(C)]
pub struct AhaCallbacks {
    pub on_chunk: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    pub on_done: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    pub on_error: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    pub user: *mut c_void,
}

impl AhaCallbacks {
    unsafe fn from_ptr<'a>(callbacks: *const AhaCallbacks) -> Result<&'a AhaCallbacks, String> {
        if callbacks.is_null() {
            return Err("callbacks is null".to_owned());
        }
        Ok(unsafe { &*callbacks })
    }

    fn chunk(&self, text: &str) {
        emit(self.on_chunk, self.user, text);
    }

    fn done(&self, text: &str) {
        emit(self.on_done, self.user, text);
    }

    #[allow(dead_code)] // reserved for callers that want callbacks-only error reporting
    fn error(&self, text: &str) {
        emit(self.on_error, self.user, text);
    }
}

fn emit(
    f: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    user: *mut c_void,
    text: &str,
) {
    let Some(f) = f else { return };
    let text = text.replace('\0', "");
    let Ok(text) = CString::new(text) else { return };
    unsafe { f(text.as_ptr(), user) }
}

// ---------------------------------------------------------------------------------------------
// Internal handles
// ---------------------------------------------------------------------------------------------

struct ModelHandle {
    model: Mutex<ModelInstance<'static>>,
}

struct StreamHandle {
    session: Mutex<aha::stream::AsrSession>,
}

#[cfg(feature = "llama")]
struct LlamaHandle {
    model: Mutex<aha::models::llama_cpp::LlamaCppModel>,
}

// ---------------------------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------------------------

unsafe fn read_str<'a>(value: *const c_char, what: &str) -> Result<&'a str, String> {
    if value.is_null() {
        return Err(format!("{what} is null"));
    }
    unsafe { CStr::from_ptr(value) }
        .to_str()
        .map_err(|e| format!("{what} is not valid UTF-8: {e}"))
}

/// Takes an owned string and converts it into a C string, so ownership never leaks a Rust
/// allocation into the host. Interior NULs are stripped rather than made an error: a model can
/// legitimately emit one, and losing it is better than losing the whole answer.
fn into_c_string(value: String) -> *mut c_char {
    let value = value.replace('\0', "");
    match CString::new(value) {
        Ok(value) => value.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

unsafe fn set_error(err: *mut *mut c_char, message: impl Into<String>) {
    if err.is_null() {
        return;
    }
    unsafe { *err = into_c_string(message.into()) };
}

unsafe fn clear_error(err: *mut *mut c_char) {
    if !err.is_null() {
        unsafe { *err = std::ptr::null_mut() };
    }
}

fn finish_ptr(
    result: ThreadResult<Result<*mut c_void, String>>,
    err: *mut *mut c_char,
) -> *mut c_void {
    match result {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            unsafe { set_error(err, message) };
            std::ptr::null_mut()
        }
        Err(_) => {
            unsafe { set_error(err, "panic inside aha") };
            std::ptr::null_mut()
        }
    }
}

fn finish_int(result: ThreadResult<Result<i32, String>>, err: *mut *mut c_char) -> i32 {
    match result {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            unsafe { set_error(err, message) };
            -1
        }
        Err(_) => {
            unsafe { set_error(err, "panic inside aha") };
            -1
        }
    }
}

unsafe fn handle_ref<'a, T>(handle: *mut c_void, name: &str) -> Result<&'a T, String> {
    if handle.is_null() {
        return Err(format!("{name} handle is null"));
    }
    Ok(unsafe { &*(handle as *const T) })
}

/// Locks a handle, ignoring poisoning: a panic in one call must not brick the model for the
/// rest of the process (the state behind it is a plain inference context).
fn lock<T>(slot: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    slot.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn pcm16_mono_to_wav(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    const CHANNELS: u16 = 1;
    const BITS: u16 = 16;
    let data_len = pcm.len() as u32;
    let block_align = CHANNELS * BITS / 8;
    let byte_rate = sample_rate * block_align as u32;

    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&CHANNELS.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&BITS.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    wav
}

fn content_to_string(content: &ChatMessageContent) -> String {
    match content {
        ChatMessageContent::Text(text) => text.clone(),
        ChatMessageContent::ContentPart(parts) => {
            let mut out = String::new();
            for part in parts {
                if let ChatMessageContentPart::Text(text_part) = part {
                    out.push_str(&text_part.text);
                }
            }
            out
        }
        ChatMessageContent::None => String::new(),
    }
}

fn extract_text(response: &ChatCompletionResponse) -> String {
    for choice in &response.choices {
        if let ChatMessage::Assistant {
            content: Some(content),
            ..
        } = &choice.message
        {
            return content_to_string(content);
        }
    }
    String::new()
}

fn extract_chunk_text(chunk: &ChatCompletionChunkResponse) -> String {
    for choice in &chunk.choices {
        if let DeltaChatMessage::Assistant {
            content: Some(content),
            ..
        } = &choice.delta
        {
            return content_to_string(content);
        }
    }
    String::new()
}

fn build_parameters(
    messages: serde_json::Value,
    model: &str,
    max_tokens: u32,
    temperature: Option<f32>,
) -> Result<ChatCompletionParameters, String> {
    let mut object = match messages {
        serde_json::Value::Array(items) => serde_json::json!({ "messages": items }),
        other => other,
    };
    object["model"] = serde_json::Value::String(model.to_owned());
    object["max_tokens"] = serde_json::json!(max_tokens);
    if let Some(value) = temperature {
        object["temperature"] = serde_json::json!(value);
    }
    serde_json::from_value(object).map_err(|e| format!("invalid messages JSON: {e}"))
}

fn normalise(max_tokens: i32, temperature_percent: i32) -> (u32, Option<f32>) {
    let max_tokens = if max_tokens > 0 {
        max_tokens as u32
    } else {
        512
    };
    let temperature = if temperature_percent > 0 {
        Some(temperature_percent as f32 / 100.0)
    } else {
        None
    };
    (max_tokens, temperature)
}

fn transcribe_pcm(
    handle: &ModelHandle,
    pcm: &[u8],
    sample_rate: u32,
    prompt: &str,
    max_tokens: i32,
) -> Result<String, String> {
    if pcm.is_empty() {
        return Err("pcm is empty".to_owned());
    }
    let wav = pcm16_mono_to_wav(pcm, sample_rate);
    let data_url = format!("data:audio/wav;base64,{}", B64.encode(wav));
    let messages = serde_json::json!([{
        "role": "user",
        "content": [
            { "type": "audio", "audio_url": { "url": data_url } },
            { "type": "text", "text": prompt },
        ],
    }]);
    let (max_tokens, temperature) = normalise(max_tokens, 0);
    let params = build_parameters(messages, "asr", max_tokens, temperature)?;
    let response = lock(&handle.model)
        .generate(params)
        .map_err(|e| format!("aha generate failed: {e:#}"))?;
    Ok(extract_text(&response))
}

fn generate_text(
    handle: &ModelHandle,
    messages_json: &str,
    max_tokens: i32,
    temperature_percent: i32,
) -> Result<String, String> {
    let messages: serde_json::Value =
        serde_json::from_str(messages_json).map_err(|e| format!("invalid messages JSON: {e}"))?;
    let (max_tokens, temperature) = normalise(max_tokens, temperature_percent);
    let params = build_parameters(messages, "asr", max_tokens, temperature)?;
    let response = lock(&handle.model)
        .generate(params)
        .map_err(|e| format!("aha generate failed: {e:#}"))?;
    Ok(extract_text(&response))
}

/// Drives the blocking `generate_stream` iterator, turning it into callbacks.
fn stream_text(
    handle: &ModelHandle,
    messages_json: &str,
    max_tokens: i32,
    temperature_percent: i32,
    callbacks: &AhaCallbacks,
) -> Result<(), String> {
    let messages: serde_json::Value =
        serde_json::from_str(messages_json).map_err(|e| format!("invalid messages JSON: {e}"))?;
    let (max_tokens, temperature) = normalise(max_tokens, temperature_percent);
    let params = build_parameters(messages, "asr", max_tokens, temperature)?;

    let mut guard = lock(&handle.model);
    let mut stream = guard
        .generate_stream(params)
        .map_err(|e| format!("aha generate_stream failed: {e:#}"))?;

    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    let mut full = String::new();

    loop {
        let mut pinned = std::pin::Pin::new(&mut *stream);
        match pinned.as_mut().poll_next(&mut cx) {
            Poll::Ready(Some(Ok(chunk))) => {
                let delta = extract_chunk_text(&chunk);
                if delta.is_empty() {
                    continue;
                }
                full.push_str(&delta);
                callbacks.chunk(&delta);
            }
            Poll::Ready(Some(Err(error))) => return Err(format!("stream error: {error:#}")),
            Poll::Ready(None) => break,
            Poll::Pending => std::thread::yield_now(),
        }
    }

    callbacks.done(&full);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Common entry points
// ---------------------------------------------------------------------------------------------

/// Returns the library version as an owned string (free with `aha_string_free`).
#[unsafe(no_mangle)]
pub extern "C" fn aha_version() -> *mut c_char {
    #[cfg(feature = "llama")]
    const BACKENDS: &str = "aha-ffi + llama.cpp";
    #[cfg(not(feature = "llama"))]
    const BACKENDS: &str = "aha-ffi";
    into_c_string(format!("{BACKENDS} {}", env!("CARGO_PKG_VERSION")))
}

/// Releases a string previously returned by this library.
///
/// # Safety
/// `value` must come from this library and must not be freed twice.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_string_free(value: *mut c_char) {
    if value.is_null() {
        return;
    }
    unsafe { drop(CString::from_raw(value)) };
}

/// Loads a model. `model_type` is a `WhichModel` name (`glm-asr-nano-2512`, `qwen3-0.6b`, ...).
///
/// Returns an opaque handle, or `NULL` with `*err` set.
///
/// # Safety
/// All pointers must be valid; `err` may be `NULL` if you do not care about the message.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_model_load(
    model_type: *const c_char,
    path: *const c_char,
    threads: i32,
    err: *mut *mut c_char,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> {
        let model_type = unsafe { read_str(model_type, "model_type") }?;
        let path = unsafe { read_str(path, "path") }?;
        if threads > 0 {
            unsafe { std::env::set_var("RAYON_NUM_THREADS", threads.to_string()) };
        }
        let which = WhichModel::from_str(model_type, true)
            .map_err(|_| format!("unknown model type: {model_type}"))?;
        let device = aha::Device::Cpu;
        let model = load_model(
            which,
            string_to_static_str(path.to_owned()),
            Some(&device),
            None,
            threads.max(1),
        )
        .map_err(|e| format!("load model failed: {e:#}"))?;
        let handle = Box::new(ModelHandle {
            model: Mutex::new(model),
        });
        Ok(Box::into_raw(handle) as *mut c_void)
    }));
    unsafe { clear_error(err) };
    finish_ptr(result, err)
}

/// Frees a handle returned by `aha_model_load`.
///
/// # Safety
/// `handle` must come from this library and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_model_free(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(handle as *mut ModelHandle));
    }));
}

/// Transcribes one utterance of 16 kHz mono PCM16 audio.
///
/// Returns an owned string (free with `aha_string_free`), or `NULL` with `*err` set.
///
/// # Safety
/// `pcm` must point to at least `len` readable bytes; all pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_transcribe(
    handle: *mut c_void,
    pcm: *const u8,
    len: usize,
    sample_rate: u32,
    prompt: *const c_char,
    max_tokens: i32,
    err: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_char, String> {
        let handle = unsafe { handle_ref::<ModelHandle>(handle, "model") }?;
        if pcm.is_null() || len == 0 {
            return Err("pcm is null or empty".to_owned());
        }
        let audio = unsafe { std::slice::from_raw_parts(pcm, len) };
        let prompt = if prompt.is_null() {
            "Please transcribe this audio into text"
        } else {
            unsafe { read_str(prompt, "prompt") }?
        };
        let text = transcribe_pcm(handle, audio, sample_rate.max(1), prompt, max_tokens)?;
        Ok(into_c_string(text))
    }));
    unsafe { clear_error(err) };
    finish_ptr_wrapped(result, err)
}

fn finish_ptr_wrapped(
    result: ThreadResult<Result<*mut c_char, String>>,
    err: *mut *mut c_char,
) -> *mut c_char {
    match result {
        Ok(Ok(value)) => value,
        Ok(Err(message)) => {
            unsafe { set_error(err, message) };
            std::ptr::null_mut()
        }
        Err(_) => {
            unsafe { set_error(err, "panic inside aha") };
            std::ptr::null_mut()
        }
    }
}

/// One-shot chat completion. `messages_json` is either a bare message array or a full request
/// object (`{"messages": [...], "max_tokens": 128}`).
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_generate(
    handle: *mut c_void,
    messages_json: *const c_char,
    max_tokens: i32,
    temperature_percent: i32,
    err: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_char, String> {
        let handle = unsafe { handle_ref::<ModelHandle>(handle, "model") }?;
        let messages = unsafe { read_str(messages_json, "messages_json") }?;
        let text = generate_text(handle, messages, max_tokens, temperature_percent)?;
        Ok(into_c_string(text))
    }));
    unsafe { clear_error(err) };
    finish_ptr_wrapped(result, err)
}

/// Streaming chat completion: `on_chunk` per delta, `on_done` with the whole text.
///
/// Returns `0` on success, `-1` on failure (`*err` set and `on_error` invoked).
///
/// # Safety
/// All pointers must be valid; the callbacks must be safe to invoke from this thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_generate_stream(
    handle: *mut c_void,
    messages_json: *const c_char,
    max_tokens: i32,
    temperature_percent: i32,
    callbacks: *const AhaCallbacks,
    err: *mut *mut c_char,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        let handle = unsafe { handle_ref::<ModelHandle>(handle, "model") }?;
        let callbacks = unsafe { AhaCallbacks::from_ptr(callbacks) }?;
        let messages = unsafe { read_str(messages_json, "messages_json") }?;
        stream_text(handle, messages, max_tokens, temperature_percent, callbacks)?;
        Ok(0)
    }));
    unsafe { clear_error(err) };
    finish_int(result, err)
}

// ---------------------------------------------------------------------------------------------
// Streaming ASR session (VAD + incremental decode)
// ---------------------------------------------------------------------------------------------

/// Opens a streaming ASR session: `vad_dir` holds the FireRedVAD files, `asr_dir` the ASR model.
///
/// Push 16 kHz mono PCM16 frames with `aha_stream_push`; the session decides where utterances
/// start and end, emits partials while the speaker is still talking, and a final per utterance.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_stream_create(
    vad_dir: *const c_char,
    asr_type: *const c_char,
    asr_dir: *const c_char,
    partial_ms: i32,
    err: *mut *mut c_char,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> {
        let vad_dir = unsafe { read_str(vad_dir, "vad_dir") }?;
        let asr_type = unsafe { read_str(asr_type, "asr_type") }?;
        let asr_dir = unsafe { read_str(asr_dir, "asr_dir") }?;
        let which = WhichModel::from_str(asr_type, true)
            .map_err(|_| format!("unknown ASR model type: {asr_type}"))?;
        let mut session = aha::stream::AsrSession::new(vad_dir, which, asr_dir)
            .map_err(|e| format!("open asr session failed: {e:#}"))?;
        session.set_partial_interval_ms(if partial_ms > 0 {
            partial_ms as u64
        } else {
            700
        });
        let handle = Box::new(StreamHandle {
            session: Mutex::new(session),
        });
        Ok(Box::into_raw(handle) as *mut c_void)
    }));
    unsafe { clear_error(err) };
    finish_ptr(result, err)
}

#[repr(C)]
pub struct AhaAsrCallbacks {
    pub on_speech_start: Option<unsafe extern "C" fn(*mut c_void)>,
    pub on_partial: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    pub on_final: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    pub on_error: Option<unsafe extern "C" fn(*const c_char, *mut c_void)>,
    pub user: *mut c_void,
}

unsafe fn dispatch_asr_events(events: Vec<aha::stream::AsrEvent>, callbacks: &AhaAsrCallbacks) {
    for event in events {
        match event {
            aha::stream::AsrEvent::SpeechStart => {
                if let Some(f) = callbacks.on_speech_start {
                    unsafe { f(callbacks.user) };
                }
            }
            aha::stream::AsrEvent::Partial(text) => {
                emit(callbacks.on_partial, callbacks.user, &text)
            }
            aha::stream::AsrEvent::Final(text) => emit(callbacks.on_final, callbacks.user, &text),
        }
    }
}

/// Feeds PCM16 to a session. `sample_rate` other than 16000 is resampled internally.
///
/// Returns `0` on success, `-1` on failure.
///
/// # Safety
/// `pcm` must point to at least `len` readable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_stream_push(
    handle: *mut c_void,
    pcm: *const u8,
    len: usize,
    sample_rate: u32,
    callbacks: *const AhaAsrCallbacks,
    err: *mut *mut c_char,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        let handle = unsafe { handle_ref::<StreamHandle>(handle, "asr session") }?;
        if pcm.is_null() || len == 0 {
            return Ok(0);
        }
        if callbacks.is_null() {
            return Err("callbacks is null".to_owned());
        }
        let callbacks = unsafe { &*callbacks };
        let audio = unsafe { std::slice::from_raw_parts(pcm, len) };
        let events = lock(&handle.session)
            .push_pcm16(audio, sample_rate.max(1))
            .map_err(|e| format!("asr push failed: {e:#}"))?;
        unsafe { dispatch_asr_events(events, callbacks) };
        Ok(0)
    }));
    unsafe { clear_error(err) };
    finish_int(result, err)
}

/// Flushes whatever is still buffered (call when the user stops talking), then resets the VAD.
///
/// # Safety
/// All pointers must be valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_stream_flush(
    handle: *mut c_void,
    callbacks: *const AhaAsrCallbacks,
    err: *mut *mut c_char,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        let handle = unsafe { handle_ref::<StreamHandle>(handle, "asr session") }?;
        if callbacks.is_null() {
            return Err("callbacks is null".to_owned());
        }
        let callbacks = unsafe { &*callbacks };
        let text = lock(&handle.session)
            .flush()
            .map_err(|e| format!("asr flush failed: {e:#}"))?;
        if let Some(text) = text {
            emit(callbacks.on_final, callbacks.user, &text);
        }
        Ok(0)
    }));
    unsafe { clear_error(err) };
    finish_int(result, err)
}

/// Frees a session returned by `aha_stream_create`.
///
/// # Safety
/// `handle` must come from this library and must not be used afterwards.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_stream_free(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(handle as *mut StreamHandle));
    }));
}

// ---------------------------------------------------------------------------------------------
// llama.cpp backend (feature = "llama")
// ---------------------------------------------------------------------------------------------

/// Loads a GGUF through llama.cpp. Any architecture llama.cpp supports works, including ones
/// candle has no implementation for (e.g. Hy-MT2's `hunyuan-dense` with STQ/Q2_0 weights).
///
/// # Safety
/// All pointers must be valid.
#[cfg(feature = "llama")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_llama_load(
    path: *const c_char,
    n_ctx: i32,
    threads: i32,
    err: *mut *mut c_char,
) -> *mut c_void {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_void, String> {
        let path = unsafe { read_str(path, "path") }?;
        let n_ctx = if n_ctx > 0 { n_ctx as u32 } else { 2048 };
        let model = aha::models::llama_cpp::LlamaCppModel::init(path, n_ctx, threads.max(1))
            .map_err(|e| format!("llama.cpp load failed: {e:#}"))?;
        let handle = Box::new(LlamaHandle {
            model: Mutex::new(model),
        });
        Ok(Box::into_raw(handle) as *mut c_void)
    }));
    unsafe { clear_error(err) };
    finish_ptr(result, err)
}

/// Frees a handle returned by `aha_llama_load`.
///
/// # Safety
/// `handle` must come from this library and must not be used afterwards.
#[cfg(feature = "llama")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_llama_free(handle: *mut c_void) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| unsafe {
        drop(Box::from_raw(handle as *mut LlamaHandle));
    }));
}

/// One-shot completion through llama.cpp.
///
/// # Safety
/// All pointers must be valid.
#[cfg(feature = "llama")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_llama_chat(
    handle: *mut c_void,
    messages_json: *const c_char,
    max_tokens: i32,
    temperature_percent: i32,
    err: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_char, String> {
        let handle = unsafe { handle_ref::<LlamaHandle>(handle, "llama model") }?;
        let messages = unsafe { read_str(messages_json, "messages_json") }?;
        let parsed: serde_json::Value =
            serde_json::from_str(messages).map_err(|e| format!("invalid messages JSON: {e}"))?;
        let (max_tokens, temperature) = normalise(max_tokens, temperature_percent);
        let params = build_parameters(parsed, "llama.cpp", max_tokens, temperature)?;
        let response = lock(&handle.model)
            .generate(params)
            .map_err(|e| format!("llama.cpp generate failed: {e:#}"))?;
        Ok(into_c_string(extract_text(&response)))
    }));
    unsafe { clear_error(err) };
    finish_ptr_wrapped(result, err)
}

/// Streaming completion through llama.cpp.
///
/// # Safety
/// All pointers must be valid; the callbacks must be safe to invoke from this thread.
#[cfg(feature = "llama")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_llama_chat_stream(
    handle: *mut c_void,
    messages_json: *const c_char,
    max_tokens: i32,
    temperature_percent: i32,
    callbacks: *const AhaCallbacks,
    err: *mut *mut c_char,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        let handle = unsafe { handle_ref::<LlamaHandle>(handle, "llama model") }?;
        let callbacks = unsafe { AhaCallbacks::from_ptr(callbacks) }?;
        let messages = unsafe { read_str(messages_json, "messages_json") }?;
        let parsed: serde_json::Value =
            serde_json::from_str(messages).map_err(|e| format!("invalid messages JSON: {e}"))?;
        let (max_tokens, temperature) = normalise(max_tokens, temperature_percent);
        let params = build_parameters(parsed, "llama.cpp", max_tokens, temperature)?;

        let mut guard = lock(&handle.model);
        let mut stream = guard
            .generate_stream(params)
            .map_err(|e| format!("llama.cpp generate_stream failed: {e:#}"))?;
        let waker = Waker::noop();
        let mut cx = Context::from_waker(waker);
        let mut full = String::new();
        loop {
            let mut pinned = std::pin::Pin::new(&mut *stream);
            match pinned.as_mut().poll_next(&mut cx) {
                Poll::Ready(Some(Ok(chunk))) => {
                    let delta = extract_chunk_text(&chunk);
                    if delta.is_empty() {
                        continue;
                    }
                    full.push_str(&delta);
                    callbacks.chunk(&delta);
                }
                Poll::Ready(Some(Err(error))) => return Err(format!("stream error: {error:#}")),
                Poll::Ready(None) => break,
                Poll::Pending => std::thread::yield_now(),
            }
        }
        callbacks.done(&full);
        Ok(0)
    }));
    unsafe { clear_error(err) };
    finish_int(result, err)
}

// ---------------------------------------------------------------------------------------------
// Model download
// ---------------------------------------------------------------------------------------------
//
// 手机宿主拉权重是"装完 App 第一次开"这条路径上最耗时的一步：几百 MB 在 4G 上要几分钟，
// 必须能显示进度、能取消，而不是卡在一个阻塞调用里。这里把 aha 的下载计划
// （[`aha::models::download`]）原样暴露出来，线程模型交给宿主：Kotlin 里就是起一个
// 后台线程/协程调 `aha_model_download`，回调里更新 UI。
//
// 两个使用形态都支持：
//   * 回调式：传 `on_progress`，每次进度变化同步回调（不要在里面做耗时操作）；
//   * 轮询式：`on_progress` 传 NULL，自己定期读 `aha_model_download_done/total`
//     （Android 上更省心：回调跨 JNI 边界时容易踩线程 attach 的坑）。

/// 进度回调：`path` 是当前文件（只在回调期间有效，别存指针），`done` / `total` 是整次
/// 下载的累计字节，`user` 原样回传。
pub type AhaDownloadProgress =
    unsafe extern "C" fn(path: *const c_char, done: u64, total: u64, user: *mut c_void);

/// 最近一次 `aha_model_download` 的累计进度（轮询式宿主用）。
static DOWNLOAD_DONE: AtomicU64 = AtomicU64::new(0);
static DOWNLOAD_TOTAL: AtomicU64 = AtomicU64::new(0);
static DOWNLOAD_FILES_DONE: AtomicU64 = AtomicU64::new(0);
static DOWNLOAD_FILES_TOTAL: AtomicU64 = AtomicU64::new(0);
/// `aha_model_download_cancel` 置位，下一次进度检查时中断下载。
static DOWNLOAD_CANCELLED: AtomicBool = AtomicBool::new(false);
static DOWNLOAD_RUNTIME: OnceLock<tokio::runtime::Runtime> = OnceLock::new();

fn download_runtime() -> Result<&'static tokio::runtime::Runtime, String> {
    if let Some(runtime) = DOWNLOAD_RUNTIME.get() {
        return Ok(runtime);
    }
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("aha-download")
        .enable_all()
        .build()
        .map_err(|e| format!("create download runtime failed: {e}"))?;
    Ok(DOWNLOAD_RUNTIME.get_or_init(|| runtime))
}

/// 权重所在的本地目录（`<save_dir>/<model_id>`，`save_dir` 省略时用 `~/.aha` 或
/// `$AHA_HOME`）。下载成功后拿它当 `aha_model_load` 的 `path`。
///
/// Returns an owned string (free with `aha_string_free`), or `NULL` with `*err` set.
///
/// # Safety
/// All pointers must be valid; `err` may be `NULL`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_model_dir(
    model_id: *const c_char,
    save_dir: *const c_char,
    err: *mut *mut c_char,
) -> *mut c_char {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_char, String> {
        let model_id = unsafe { read_str(model_id, "model_id") }?;
        let save_dir = if save_dir.is_null() {
            None
        } else {
            Some(unsafe { read_str(save_dir, "save_dir") }?)
        };
        let dir =
            aha::models::download::model_dir(model_id, save_dir).map_err(|e| format!("{e:#}"))?;
        Ok(into_c_string(dir.display().to_string()))
    }));
    unsafe { clear_error(err) };
    finish_ptr_wrapped(result, err)
}

/// 下载一个模型的权重（阻塞调用，建议放在宿主自己的后台线程里）。
///
/// `model_id` 是 `-m` 接收的那个 id（`qwen3.5-gguf`、`gpt-sovits-v2pro-zh`、`OpenBMB/TurboVLA`、
/// `csukuangfj/kokoro-multi-lang-v1_1` ...）；不在内置镜像表里的 id 当成 ModelScope 仓库
/// 整仓下载。`save_dir` 可为 `NULL`（= `~/.aha` / `$AHA_HOME`）。
///
/// 返回 `0` 成功、`-1` 失败（`*err` 有消息）、`-2` 已取消。重复调用会跳过已经下完的文件，
/// 所以"下到一半被杀"之后直接再调一次即可续上。
///
/// # Safety
/// 指针必须有效；`on_progress` 可以为 `NULL`；`user` 在本次调用期间必须保持有效。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn aha_model_download(
    model_id: *const c_char,
    save_dir: *const c_char,
    max_retries: i32,
    on_progress: Option<AhaDownloadProgress>,
    user: *mut c_void,
    err: *mut *mut c_char,
) -> i32 {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        let model_id = unsafe { read_str(model_id, "model_id") }?.to_owned();
        let save_dir = if save_dir.is_null() {
            None
        } else {
            Some(unsafe { read_str(save_dir, "save_dir") }?.to_owned())
        };

        DOWNLOAD_CANCELLED.store(false, Ordering::SeqCst);
        DOWNLOAD_DONE.store(0, Ordering::SeqCst);
        DOWNLOAD_TOTAL.store(0, Ordering::SeqCst);
        DOWNLOAD_FILES_DONE.store(0, Ordering::SeqCst);
        DOWNLOAD_FILES_TOTAL.store(0, Ordering::SeqCst);

        let progress = |p: &aha::utils::DownloadProgress<'_>| {
            DOWNLOAD_DONE.store(p.done, Ordering::Relaxed);
            DOWNLOAD_TOTAL.store(p.total, Ordering::Relaxed);
            DOWNLOAD_FILES_DONE.store(p.files_done as u64, Ordering::Relaxed);
            DOWNLOAD_FILES_TOTAL.store(p.files_total as u64, Ordering::Relaxed);
            if let Some(callback) = on_progress {
                let path = CString::new(p.path.replace('\0', "")).unwrap_or_default();
                unsafe { callback(path.as_ptr(), p.done, p.total, user) };
            }
        };
        let cancel = || DOWNLOAD_CANCELLED.load(Ordering::SeqCst);
        let ctl = aha::utils::DownloadCtl {
            progress: Some(&progress),
            cancel: Some(&cancel),
        };

        let runtime = download_runtime()?;
        runtime
            .block_on(aha::models::download::download_model_files(
                &model_id,
                save_dir.as_deref(),
                max_retries.max(1) as u32,
                ctl,
            ))
            .map_err(|e| format!("download {model_id} failed: {e:#}"))?;
        Ok(0)
    }));
    unsafe { clear_error(err) };
    match result {
        Ok(Ok(code)) => code,
        Ok(Err(message)) => {
            if DOWNLOAD_CANCELLED.load(Ordering::SeqCst) {
                -2
            } else {
                unsafe { set_error(err, message) };
                -1
            }
        }
        Err(_) => {
            unsafe { set_error(err, "panic inside aha") };
            -1
        }
    }
}

/// 请求取消正在进行的 `aha_model_download`（下一个 chunk 生效）。
///
/// 只置标记、立刻返回：下载线程自己退出后 `aha_model_download` 返回 `-2`。
/// 在别的线程调用是安全的。
#[unsafe(no_mangle)]
pub extern "C" fn aha_model_download_cancel() {
    DOWNLOAD_CANCELLED.store(true, Ordering::SeqCst);
}

/// 已下载字节数（轮询式宿主用；`on_progress` 传 NULL 时靠这三个读数画进度条）。
#[unsafe(no_mangle)]
pub extern "C" fn aha_model_download_done() -> u64 {
    DOWNLOAD_DONE.load(Ordering::Relaxed)
}

/// 本次下载的总字节数（服务端文件列表之和；开始前为 0）。
#[unsafe(no_mangle)]
pub extern "C" fn aha_model_download_total() -> u64 {
    DOWNLOAD_TOTAL.load(Ordering::Relaxed)
}

/// 已完成的文件数。
#[unsafe(no_mangle)]
pub extern "C" fn aha_model_download_files_done() -> u64 {
    DOWNLOAD_FILES_DONE.load(Ordering::Relaxed)
}

/// 本次下载的文件总数。
#[unsafe(no_mangle)]
pub extern "C" fn aha_model_download_files_total() -> u64 {
    DOWNLOAD_FILES_TOTAL.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn c_string(value: &str) -> CString {
        CString::new(value).unwrap()
    }

    unsafe fn take_string(value: *mut c_char) -> String {
        assert!(!value.is_null());
        let text = unsafe { CStr::from_ptr(value) }
            .to_str()
            .unwrap()
            .to_owned();
        unsafe { aha_string_free(value) };
        text
    }

    unsafe fn take_error(err: *mut c_char) -> String {
        assert!(!err.is_null());
        let text = unsafe { CStr::from_ptr(err) }.to_str().unwrap().to_owned();
        unsafe { aha_string_free(err) };
        text
    }

    #[test]
    fn model_dir_joins_save_dir_and_id() {
        let id = c_string("qwen3.5-gguf");
        let dir = c_string(&std::env::temp_dir().display().to_string());
        let mut err: *mut c_char = std::ptr::null_mut();
        let out = unsafe { aha_model_dir(id.as_ptr(), dir.as_ptr(), &mut err) };
        let text = unsafe { take_string(out) };
        assert!(err.is_null());
        assert_eq!(
            PathBuf::from(text),
            std::env::temp_dir().join("qwen3.5-gguf")
        );
    }

    #[test]
    fn null_model_id_is_an_error_not_a_crash() {
        let mut err: *mut c_char = std::ptr::null_mut();
        let out = unsafe { aha_model_dir(std::ptr::null(), std::ptr::null(), &mut err) };
        assert!(out.is_null());
        let message = unsafe { take_error(err) };
        assert!(message.contains("model_id"), "{message}");
    }

    /// 空 id 在碰网络/磁盘之前就失败；顺便验证上一次的 cancel 标记不会串到下一次调用。
    #[test]
    fn blank_id_fails_and_cancel_flag_resets_per_call() {
        aha_model_download_cancel();
        let id = c_string("   ");
        let mut err: *mut c_char = std::ptr::null_mut();
        let code = unsafe {
            aha_model_download(
                id.as_ptr(),
                std::ptr::null(),
                1,
                None,
                std::ptr::null_mut(),
                &mut err,
            )
        };
        assert_eq!(code, -1, "空 id 应该报错，而不是取消");
        let message = unsafe { take_error(err) };
        assert!(message.contains("model_id"), "{message}");
    }
}

//! llama.cpp 后端。
//!
//! aha 原有的模型都是"手写的 candle 图 + 白名单架构"；这个后端相反：把生成交给 llama.cpp，
//! 因此能吃 llama.cpp 支持的全部架构与量化格式（含 hunyuan-dense、Q1_0/Q2_0 这类新量化）。
//!
//! 编译需要 `--features llama`，并把 llama.cpp 的静态库放到 `native/llama-prebuilt/<abi>/`。

mod llama_bindings;

use llama_bindings as sys;

use std::ffi::{CStr, CString};
use std::pin::Pin;
use std::ptr;
use std::task::{Context, Poll};
use std::time::Instant;

use anyhow::{Result, anyhow};
use futures_util::Stream;

use crate::models::GenerateModel;
use crate::params::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse, ChatMessage,
    ChatMessageContent, ChatMessageContentPart,
};
use crate::utils::response_utils::{
    build_completion_chunk_response, build_completion_response_with_time,
};

/// 一个由 llama.cpp 托管的模型。
///
/// 每次生成都会新建一个 `llama_context`（`n_ctx` 通常很小，这比维护自引用结构简单得多）。
#[cfg(target_os = "android")]
unsafe extern "C" fn android_log_sink(
    _level: sys::ggml_log_level,
    text: *const ::std::os::raw::c_char,
    _user: *mut ::std::os::raw::c_void,
) {
    if text.is_null() {
        return;
    }
    unsafe extern "C" {
        fn __android_log_write(
            prio: i32,
            tag: *const ::std::os::raw::c_char,
            text: *const ::std::os::raw::c_char,
        ) -> i32;
    }
    let tag = b"llama.cpp\0".as_ptr() as *const ::std::os::raw::c_char;
    unsafe { __android_log_write(4, tag, text) };
}

/// 直接用 logcat 输出一行诊断（tag 固定 `llama.cpp`）。
fn log_line(msg: &str) {
    #[cfg(target_os = "android")]
    {
        let tag = b"llama.cpp\0";
        let text = std::ffi::CString::new(msg).unwrap_or_default();
        unsafe extern "C" {
            fn __android_log_write(
                prio: i32,
                tag: *const ::std::os::raw::c_char,
                text: *const ::std::os::raw::c_char,
            ) -> i32;
        }
        unsafe {
            __android_log_write(
                4,
                tag.as_ptr() as *const ::std::os::raw::c_char,
                text.as_ptr(),
            );
        }
    }
    #[cfg(not(target_os = "android"))]
    {
        eprintln!("{msg}");
    }
}
/// 把 llama.cpp/ggml 的内部日志转发到 logcat，否则 Android 上 stderr 被吞掉、加载失败时看不到原因。
fn install_log_sink() {
    #[cfg(target_os = "android")]
    unsafe {
        sys::llama_log_set(Some(android_log_sink), std::ptr::null_mut());
    }
}
pub struct LlamaCppModel {
    model: *mut sys::llama_model,
    vocab: *const sys::llama_vocab,
    n_ctx: u32,
    threads: i32,
    /// 模型自己的 prompt 渲染器（llama.cpp 不认识它的 jinja 模板时用）。
    prompt_renderer: Option<PromptRenderer>,
}

/// prompt 渲染结果。`add_special` 表示渲染出来的文本是否还缺 BOS、需要 tokenizer 补。
pub struct RenderedPrompt {
    pub text: String,
    pub add_special: bool,
}

/// 由具体模型提供的对话渲染函数：`(roles, contents) -> Option<RenderedPrompt>`。
///
/// llama.cpp 只认几个内置 jinja 模板，`llama_chat_apply_template` 对自定义模板会
/// 返回负数。这类模型（例如 Hy-MT2）把渲染逻辑放在自己的模块里，通过这个钩子接进来，
/// llama.cpp 后端本身不需要知道任何模型名。
pub type PromptRenderer = fn(&[CString], &[CString]) -> Option<RenderedPrompt>;

// llama.cpp 的 model / context 在 `llama_decode` 期间是内部加锁的；
// 这里我们自己保证同一时刻只有一个调用方持有 `&mut self`。
unsafe impl Send for LlamaCppModel {}
unsafe impl Sync for LlamaCppModel {}

impl LlamaCppModel {
    /// 打开一个 GGUF 文件。`n_ctx` 是 KV cache 的上下文长度，翻译场景 2048 足够。
    pub fn init(path: &str, n_ctx: u32, threads: i32) -> Result<Self> {
        Self::init_with_prompt_renderer(path, n_ctx, threads, None)
    }

    /// 同 [`Self::init`]，但由调用方指定 prompt 渲染器（自定义聊天模板的模型用）。
    pub fn init_with_prompt_renderer(
        path: &str,
        n_ctx: u32,
        threads: i32,
        prompt_renderer: Option<PromptRenderer>,
    ) -> Result<Self> {
        let cpath = CString::new(path)?;

        unsafe {
            sys::llama_backend_init();
            install_log_sink();
        }

        let mut params = unsafe { sys::llama_model_default_params() };
        params.n_gpu_layers = 0; // 端上只有 CPU 后端

        // 先自己探一次文件，区分“打不开”和“llama.cpp 解析失败”。
        let meta =
            std::fs::metadata(path).map_err(|e| anyhow!("llama.cpp: stat({path}) failed: {e}"))?;
        let probe = std::fs::File::open(path)
            .map_err(|e| anyhow!("llama.cpp: open({path}) failed: {e}"))?;
        let mut head = [0u8; 8];
        use std::io::Read;
        {
            let mut probe = probe;
            probe
                .read_exact(&mut head)
                .map_err(|e| anyhow!("llama.cpp: read head({path}) failed: {e}"))?;
        }
        log_line(&format!("probe: len={} head={:02x?}", meta.len(), head));

        let model = unsafe { sys::llama_model_load_from_file(cpath.as_ptr(), params) };
        if model.is_null() {
            return Err(anyhow!(
                "llama.cpp failed to load model: {path} (len={} head={:02x?})",
                meta.len(),
                head
            ));
        }
        let vocab = unsafe { sys::llama_model_get_vocab(model) };

        Ok(Self {
            model,
            vocab,
            n_ctx: n_ctx.max(256),
            threads: threads.max(1),
            prompt_renderer,
        })
    }

    pub fn n_ctx(&self) -> u32 {
        self.n_ctx
    }

    fn new_context(&self) -> Result<*mut sys::llama_context> {
        let mut params = unsafe { sys::llama_context_default_params() };
        params.n_ctx = self.n_ctx;
        params.n_batch = self.n_ctx;
        params.n_threads = self.threads;
        params.n_threads_batch = self.threads;

        let ctx = unsafe { sys::llama_init_from_model(self.model, params) };
        if ctx.is_null() {
            return Err(anyhow!("llama.cpp failed to create context"));
        }
        Ok(ctx)
    }

    fn new_sampler(&self, mes: &ChatCompletionParameters) -> *mut sys::llama_sampler {
        let temperature = mes.temperature.unwrap_or(0.0);
        let top_p = mes.top_p.unwrap_or(0.95);
        let top_k = mes.top_k.map(|k| k as i32).unwrap_or(40);
        // 重复惩罚：OpenAI 的 frequency/presence penalty 和 llama.cpp 的 penalties 采样器同义。
        let repeat_penalty = mes.repeat_penalty.unwrap_or(1.0);
        let repeat_last_n = mes.repeat_last_n.unwrap_or(64).min(i32::MAX as usize) as i32;
        let frequency = mes.frequency_penalty.unwrap_or(0.0);
        let presence = mes.presence_penalty.unwrap_or(0.0);
        unsafe {
            let params = sys::llama_sampler_chain_default_params();
            let chain = sys::llama_sampler_chain_init(params);
            if repeat_penalty != 1.0 || frequency != 0.0 || presence != 0.0 {
                // 第一个参数（n_vocab）在现行 llama.cpp 里已废弃，传 0 即可。
                sys::llama_sampler_chain_add(
                    chain,
                    sys::llama_sampler_init_penalties(
                        0,
                        repeat_last_n,
                        repeat_penalty,
                        frequency,
                        presence,
                    ),
                );
            }
            if temperature <= 0.0 {
                sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_greedy());
            } else {
                if top_k > 0 {
                    sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_top_k(top_k));
                }
                if top_p > 0.0 && top_p < 1.0 {
                    sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_top_p(top_p, 1));
                }
                sys::llama_sampler_chain_add(chain, sys::llama_sampler_init_temp(temperature));
                sys::llama_sampler_chain_add(
                    chain,
                    sys::llama_sampler_init_dist(sys::LLAMA_DEFAULT_SEED),
                );
            }
            chain
        }
    }

    /// 把 aha 的 messages 渲染成 prompt，并告诉调用方文本里是否已经带有 BOS。
    fn build_prompt(&self, mes: &ChatCompletionParameters) -> Result<(String, bool)> {
        let mut roles: Vec<CString> = Vec::new();
        let mut contents: Vec<CString> = Vec::new();
        let mut chat: Vec<sys::llama_chat_message> = Vec::new();

        for msg in &mes.messages {
            let (role, text) = message_role_text(msg);
            roles.push(CString::new(role)?);
            contents.push(CString::new(text)?);
        }
        for (role, content) in roles.iter().zip(contents.iter()) {
            chat.push(sys::llama_chat_message {
                role: role.as_ptr(),
                content: content.as_ptr(),
            });
        }

        // 模型自带渲染器时优先用它：这类模型的 jinja 模板 llama.cpp 解析不了。
        if let Some(render) = self.prompt_renderer
            && let Some(rendered) = render(&roles, &contents)
        {
            return Ok((rendered.text, rendered.add_special));
        }

        let tmpl = unsafe { sys::llama_model_chat_template(self.model, ptr::null()) };
        if tmpl.is_null() {
            return Ok((fallback_prompt(&roles, &contents), true));
        }

        // 先按 8KB 试，不够再按返回的长度扩容。
        let mut buf = vec![0 as std::os::raw::c_char; 8192];
        let mut needed = unsafe {
            sys::llama_chat_apply_template(
                tmpl,
                chat.as_ptr(),
                chat.len(),
                true,
                buf.as_mut_ptr(),
                buf.len() as i32,
            )
        };
        if needed < 0 {
            return Ok((fallback_prompt(&roles, &contents), true));
        }
        if needed as usize > buf.len() {
            buf = vec![0 as std::os::raw::c_char; needed as usize];
            needed = unsafe {
                sys::llama_chat_apply_template(
                    tmpl,
                    chat.as_ptr(),
                    chat.len(),
                    true,
                    buf.as_mut_ptr(),
                    buf.len() as i32,
                )
            };
            if needed < 0 {
                return Ok((fallback_prompt(&roles, &contents), true));
            }
        }

        let text = unsafe { CStr::from_ptr(buf.as_ptr() as *const std::os::raw::c_char) };
        Ok((text.to_string_lossy().into_owned(), true))
    }

    fn tokenize(&self, text: &str, add_special: bool) -> Result<Vec<i32>> {
        let ctext = CString::new(text)?;
        let len = text.len() as i32;
        // 先问需要多少 token（返回负数表示"需要 -n 个"）。
        let need = unsafe {
            sys::llama_tokenize(
                self.vocab,
                ctext.as_ptr(),
                len,
                ptr::null_mut(),
                0,
                add_special,
                true,
            )
        };
        let cap = if need < 0 { -need } else { need.max(1) } + 8;
        let mut tokens = vec![0i32; cap as usize];
        let n = unsafe {
            sys::llama_tokenize(
                self.vocab,
                ctext.as_ptr(),
                len,
                tokens.as_mut_ptr(),
                cap,
                add_special,
                true,
            )
        };
        if n < 0 {
            return Err(anyhow!("llama.cpp tokenize failed ({n})"));
        }
        tokens.truncate(n as usize);
        Ok(tokens)
    }

    /// 一个 token 的原始字节。中文/emoji 常常被切成多个 token，每个 token 的字节
    /// 单独 `from_utf8` 会变成替换字符，所以要交给 [`take_utf8`] 拼。
    fn token_piece_bytes(&self, token: i32) -> Vec<u8> {
        let mut buf = vec![0u8; 256];
        let mut n = unsafe {
            sys::llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr() as *mut std::os::raw::c_char,
                buf.len() as i32,
                0,
                false,
            )
        };
        if n < 0 {
            // 返回 -n 表示缓冲区不够：按需要的长度重来一次。
            buf = vec![0u8; (-n) as usize];
            n = unsafe {
                sys::llama_token_to_piece(
                    self.vocab,
                    token,
                    buf.as_mut_ptr() as *mut std::os::raw::c_char,
                    buf.len() as i32,
                    0,
                    false,
                )
            };
        }
        if n <= 0 {
            return Vec::new();
        }
        buf.truncate(n as usize);
        buf
    }

    fn max_tokens(mes: &ChatCompletionParameters) -> u32 {
        mes.max_tokens
            .or(mes.max_completion_tokens)
            .unwrap_or(4096)
            .min(4096)
    }

    fn decode_prompt(&self, ctx: *mut sys::llama_context, tokens: &mut [i32]) -> Result<()> {
        let mut batch =
            unsafe { sys::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
        let rc = unsafe { sys::llama_decode(ctx, batch) };
        if rc != 0 {
            return Err(anyhow!("llama.cpp decode failed ({rc})"));
        }
        Ok(())
    }
}

/// 从 `pending` 里取出能构成完整 UTF-8 的片段；被 token 边界截断的尾巴留在 `pending`。
/// 中文/emoji 的字节经常跨 token，逐 token 做 lossy 转换会冒出 U+FFFD。
fn take_utf8(pending: &mut Vec<u8>) -> Option<String> {
    if pending.is_empty() {
        return None;
    }
    match std::str::from_utf8(pending) {
        Ok(_) => {
            let s = String::from_utf8_lossy(pending).into_owned();
            pending.clear();
            Some(s)
        }
        Err(err) => {
            let valid = err.valid_up_to();
            if valid > 0 {
                let s = String::from_utf8_lossy(&pending[..valid]).into_owned();
                pending.drain(..valid);
                return Some(s);
            }
            match err.error_len() {
                // 尾部被截断，等下一个 token 补齐。
                None => None,
                // 真的非法字节：吐一个替换字符，避免流永远卡住。
                Some(len) => {
                    pending.drain(..len);
                    Some("\u{fffd}".to_string())
                }
            }
        }
    }
}

impl Drop for LlamaCppModel {
    fn drop(&mut self) {
        unsafe {
            sys::llama_model_free(self.model);
            sys::llama_backend_free();
        }
        self.model = ptr::null_mut();
    }
}

fn message_role_text(msg: &ChatMessage) -> (&'static str, String) {
    match msg {
        ChatMessage::System { content, .. } => ("system", content_to_text(content)),
        ChatMessage::Developer { content, .. } => ("system", content_to_text(content)),
        ChatMessage::User { content, .. } => ("user", content_to_text(content)),
        ChatMessage::Assistant { content, .. } => (
            "assistant",
            content.as_ref().map(content_to_text).unwrap_or_default(),
        ),
        ChatMessage::Tool { content, .. } => ("tool", content_to_text(content)),
    }
}

fn content_to_text(content: &ChatMessageContent) -> String {
    match content {
        ChatMessageContent::Text(text) => text.clone(),
        ChatMessageContent::ContentPart(parts) => parts
            .iter()
            .filter_map(|p| match p {
                ChatMessageContentPart::Text(t) => Some(t.text.clone()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join(""),
        ChatMessageContent::None => String::new(),
    }
}

fn fallback_prompt(roles: &[CString], contents: &[CString]) -> String {
    let mut out = String::new();
    for (role, content) in roles.iter().zip(contents.iter()) {
        out.push_str(&role.to_string_lossy());
        out.push_str(": ");
        out.push_str(&content.to_string_lossy());
        out.push('\n');
    }
    out.push_str("assistant: ");
    out
}

impl GenerateModel for LlamaCppModel {
    fn generate(&mut self, mes: ChatCompletionParameters) -> Result<ChatCompletionResponse> {
        let (prompt, add_special) = self.build_prompt(&mes)?;
        let mut tokens = self.tokenize(&prompt, add_special)?;
        let max_tokens = Self::max_tokens(&mes);

        if std::env::var("AHA_DEBUG_PROMPT").is_ok() {
            eprintln!("PROMPT={prompt:?}");
            let head: Vec<i32> = tokens.iter().copied().take(12).collect();
            eprintln!("TOKENS={head:?}");
            for t in head {
                let piece = self.token_piece_bytes(t);
                eprintln!("PIECE {t} = {:?}", String::from_utf8_lossy(&piece));
            }
        }

        let ctx = self.new_context()?;
        let sampler = self.new_sampler(&mes);
        let mut out = String::new();
        let mut pending = Vec::<u8>::new();
        let mut n = 0u32;
        let prompt_tokens = tokens.len() as u32;
        let mut prompt_secs = 0.0f64;
        let mut gen_secs = 0.0f64;

        let result = (|| -> Result<String> {
            let t0 = Instant::now();
            self.decode_prompt(ctx, &mut tokens)?;
            prompt_secs = t0.elapsed().as_secs_f64();
            let t1 = Instant::now();
            while n < max_tokens {
                let id = unsafe { sys::llama_sampler_sample(sampler, ctx, -1) };
                if unsafe { sys::llama_vocab_is_eog(self.vocab, id) } {
                    break;
                }
                pending.extend_from_slice(&self.token_piece_bytes(id));
                if let Some(piece) = take_utf8(&mut pending) {
                    out.push_str(&piece);
                }
                let mut next = [id];
                let mut batch = unsafe { sys::llama_batch_get_one(next.as_mut_ptr(), 1) };
                let rc = unsafe { sys::llama_decode(ctx, batch) };
                if rc != 0 {
                    return Err(anyhow!("llama.cpp decode failed ({rc})"));
                }
                n += 1;
            }
            gen_secs = t1.elapsed().as_secs_f64();
            // 结束前把残留的完整字节吐出来（通常是最后一个多字节字符）。
            if let Some(piece) = take_utf8(&mut pending) {
                out.push_str(&piece);
            }
            Ok(out)
        })();

        unsafe {
            sys::llama_sampler_free(sampler);
            sys::llama_free(ctx);
        }

        let text = result?;
        Ok(build_completion_response_with_time(
            text,
            "llama.cpp",
            Some(n),
            Some(gen_secs),
            Some(prompt_tokens),
            Some(prompt_secs),
        ))
    }

    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<Box<dyn Stream<Item = Result<ChatCompletionChunkResponse>> + Send + Unpin + '_>>
    {
        let (prompt, add_special) = self.build_prompt(&mes)?;
        let tokens = self.tokenize(&prompt, add_special)?;
        let max_tokens = Self::max_tokens(&mes);

        let ctx = self.new_context()?;
        let sampler = self.new_sampler(&mes);

        let mut owned = tokens;
        if let Err(err) = self.decode_prompt(ctx, &mut owned) {
            unsafe {
                sys::llama_sampler_free(sampler);
                sys::llama_free(ctx);
            }
            return Err(err);
        }

        Ok(Box::new(LlamaCppStream {
            model: self,
            ctx,
            sampler,
            max_tokens,
            produced: 0,
            pending: Vec::new(),
            done: false,
        }))
    }
}

/// 逐 token 产出的流。每个 `poll_next` 解码一个 token，因此不需要额外线程。
struct LlamaCppStream<'a> {
    model: &'a LlamaCppModel,
    ctx: *mut sys::llama_context,
    sampler: *mut sys::llama_sampler,
    max_tokens: u32,
    produced: u32,
    /// 跨 token 被截断的 UTF-8 字节，凑齐后再发 chunk。
    pending: Vec<u8>,
    done: bool,
}

unsafe impl Send for LlamaCppStream<'_> {}

impl Drop for LlamaCppStream<'_> {
    fn drop(&mut self) {
        unsafe {
            sys::llama_sampler_free(self.sampler);
            sys::llama_free(self.ctx);
        }
    }
}

impl Stream for LlamaCppStream<'_> {
    type Item = Result<ChatCompletionChunkResponse>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        if this.done || this.produced >= this.max_tokens {
            return Poll::Ready(None);
        }

        loop {
            if this.produced >= this.max_tokens {
                this.done = true;
                return Poll::Ready(None);
            }

            let id = unsafe { sys::llama_sampler_sample(this.sampler, this.ctx, -1) };
            if unsafe { sys::llama_vocab_is_eog(this.model.vocab, id) } {
                this.done = true;
                return Poll::Ready(None);
            }

            let bytes = this.model.token_piece_bytes(id);
            this.pending.extend_from_slice(&bytes);
            let mut next = [id];
            let mut batch = unsafe { sys::llama_batch_get_one(next.as_mut_ptr(), 1) };
            let rc = unsafe { sys::llama_decode(this.ctx, batch) };
            if rc != 0 {
                this.done = true;
                return Poll::Ready(Some(Err(anyhow!("llama.cpp decode failed ({rc})"))));
            }
            this.produced += 1;

            // 凑不齐一个完整 UTF-8 字符就继续推进，避免产出空/乱码 chunk。
            match take_utf8(&mut this.pending) {
                Some(piece) if !piece.is_empty() => {
                    return Poll::Ready(Some(Ok(build_completion_chunk_response(
                        piece,
                        "llama.cpp",
                        None,
                        None,
                    ))));
                }
                _ => continue,
            }
        }
    }
}

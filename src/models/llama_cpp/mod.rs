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

use anyhow::{anyhow, Result};
use futures_util::Stream;

use crate::models::GenerateModel;
use crate::params::chat::{
    ChatCompletionChunkResponse, ChatCompletionParameters, ChatCompletionResponse, ChatMessage,
    ChatMessageContent, ChatMessageContentPart,
};
use crate::utils::response_utils::{build_completion_chunk_response, build_completion_response};

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
        fn __android_log_write(prio: i32, tag: *const ::std::os::raw::c_char, text: *const ::std::os::raw::c_char) -> i32;
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
            fn __android_log_write(prio: i32, tag: *const ::std::os::raw::c_char, text: *const ::std::os::raw::c_char) -> i32;
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
}

// llama.cpp 的 model / context 在 `llama_decode` 期间是内部加锁的；
// 这里我们自己保证同一时刻只有一个调用方持有 `&mut self`。
unsafe impl Send for LlamaCppModel {}
unsafe impl Sync for LlamaCppModel {}

impl LlamaCppModel {
    /// 打开一个 GGUF 文件。`n_ctx` 是 KV cache 的上下文长度，翻译场景 2048 足够。
    pub fn init(path: &str, n_ctx: u32, threads: i32) -> Result<Self> {
        let cpath = CString::new(path)?;

        unsafe {
            sys::llama_backend_init();
            install_log_sink();
        }

        let mut params = unsafe { sys::llama_model_default_params() };
        params.n_gpu_layers = 0; // 端上只有 CPU 后端

        // 先自己探一次文件，区分“打不开”和“llama.cpp 解析失败”。
        let meta = std::fs::metadata(path)
            .map_err(|e| anyhow!("llama.cpp: stat({path}) failed: {e}"))?;
        let probe = std::fs::File::open(path)
            .map_err(|e| anyhow!("llama.cpp: open({path}) failed: {e}"))?;
        let mut head = [0u8; 8];
        use std::io::Read;
        {
            let mut probe = probe;
            probe.read_exact(&mut head)
                .map_err(|e| anyhow!("llama.cpp: read head({path}) failed: {e}"))?;
        }
        log_line(&format!(
            "probe: len={} head={:02x?}",
            meta.len(),
            head
        ));

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

    fn new_sampler(&self, temperature: f32, top_p: f32, top_k: i32) -> *mut sys::llama_sampler {
        unsafe {
            let params = sys::llama_sampler_chain_default_params();
            let chain = sys::llama_sampler_chain_init(params);
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

    /// 把 aha 的 messages 渲染成模型自带的 chat template。
    fn build_prompt(&self, mes: &ChatCompletionParameters) -> Result<String> {
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

        let tmpl = unsafe { sys::llama_model_chat_template(self.model, ptr::null()) };
        if tmpl.is_null() {
            return Ok(fallback_prompt(&roles, &contents));
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
            return Ok(fallback_prompt(&roles, &contents));
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
                return Ok(fallback_prompt(&roles, &contents));
            }
        }

        let text = unsafe { CStr::from_ptr(buf.as_ptr() as *const std::os::raw::c_char) };
        Ok(text.to_string_lossy().into_owned())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
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
                true,
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
                true,
                true,
            )
        };
        if n < 0 {
            return Err(anyhow!("llama.cpp tokenize failed ({n})"));
        }
        tokens.truncate(n as usize);
        Ok(tokens)
    }

    fn token_piece(&self, token: i32) -> String {
        let mut buf = vec![0 as std::os::raw::c_char; 256];
        let n = unsafe {
            sys::llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr(),
                buf.len() as i32,
                0,
                false,
            )
        };
        if n <= 0 {
            return String::new();
        }
        let bytes: Vec<u8> = buf[..n as usize].iter().map(|b| *b as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn max_tokens(mes: &ChatCompletionParameters) -> u32 {
        mes.max_tokens.unwrap_or(512).min(4096)
    }

    fn decode_prompt(
        &self,
        ctx: *mut sys::llama_context,
        tokens: &mut [i32],
    ) -> Result<()> {
        let mut batch = unsafe { sys::llama_batch_get_one(tokens.as_mut_ptr(), tokens.len() as i32) };
        let rc = unsafe { sys::llama_decode(ctx, batch) };
        if rc != 0 {
            return Err(anyhow!("llama.cpp decode failed ({rc})"));
        }
        Ok(())
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
        ChatMessage::Assistant { content, .. } => {
            ("assistant", content.as_ref().map(content_to_text).unwrap_or_default())
        }
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
        let prompt = self.build_prompt(&mes)?;
        let mut tokens = self.tokenize(&prompt)?;
        let max_tokens = Self::max_tokens(&mes);
        let temperature = mes.temperature.unwrap_or(0.0);
        let top_p = mes.top_p.unwrap_or(0.95);
        let top_k = 40; // ChatCompletionParameters has no top_k; 40 is llama.cpp's usual default

        let ctx = self.new_context()?;
        let sampler = self.new_sampler(temperature, top_p, top_k);
        let mut out = String::new();
        let mut n = 0u32;

        let result = (|| -> Result<String> {
            self.decode_prompt(ctx, &mut tokens)?;
            while n < max_tokens {
                let id = unsafe { sys::llama_sampler_sample(sampler, ctx, -1) };
                if unsafe { sys::llama_vocab_is_eog(self.vocab, id) } {
                    break;
                }
                out.push_str(&self.token_piece(id));
                let mut next = [id];
                let mut batch =
                    unsafe { sys::llama_batch_get_one(next.as_mut_ptr(), 1) };
                let rc = unsafe { sys::llama_decode(ctx, batch) };
                if rc != 0 {
                    return Err(anyhow!("llama.cpp decode failed ({rc})"));
                }
                n += 1;
            }
            Ok(out)
        })();

        unsafe {
            sys::llama_sampler_free(sampler);
            sys::llama_free(ctx);
        }

        let text = result?;
        Ok(build_completion_response(text, "llama.cpp", Some(n), None))
    }

    fn generate_stream(
        &mut self,
        mes: ChatCompletionParameters,
    ) -> Result<Box<dyn Stream<Item = Result<ChatCompletionChunkResponse>> + Send + Unpin + '_>> {
        let prompt = self.build_prompt(&mes)?;
        let tokens = self.tokenize(&prompt)?;
        let max_tokens = Self::max_tokens(&mes);
        let temperature = mes.temperature.unwrap_or(0.0);
        let top_p = mes.top_p.unwrap_or(0.95);
        let top_k = 40; // ChatCompletionParameters has no top_k; 40 is llama.cpp's usual default

        let ctx = self.new_context()?;
        let sampler = self.new_sampler(temperature, top_p, top_k);

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

            let piece = this.model.token_piece(id);
            let mut next = [id];
            let mut batch = unsafe { sys::llama_batch_get_one(next.as_mut_ptr(), 1) };
            let rc = unsafe { sys::llama_decode(this.ctx, batch) };
            if rc != 0 {
                this.done = true;
                return Poll::Ready(Some(Err(anyhow!("llama.cpp decode failed ({rc})"))));
            }
            this.produced += 1;

            if piece.is_empty() {
                // 空片段（多字节字符的前半段）继续推进，避免产出空 chunk。
                continue;
            }

            return Poll::Ready(Some(Ok(build_completion_chunk_response(
                piece,
                "llama.cpp",
                None,
                None,
            ))));
        }
    }
}

//! llama.cpp GGUF exec（candle 不支持的架构 / STQ 这类新量化，如 Hy-MT2）。

use std::time::Instant;

use anyhow::Result;

use crate::models::GenerateModel;
use crate::models::common::runtime::llama_cpp::LlamaCppModel;
use crate::models::hy_mt2;
use crate::utils::get_file_path;

pub struct LlamaCppExec;

impl LlamaCppExec {
    pub fn run_gguf(
        input: &[String],
        output: Option<&str>,
        gguf_path: &str,
        ctx_size: u32,
        threads: i32,
    ) -> Result<()> {
        // CLI 的 `-i` 用空格分词（RunArgs::input 带 value_delimiter ' '），
        // 这里要把词拼回整句，否则只翻译第一个单词。
        let input_text = input.join(" ");
        let target_text = if input_text.starts_with("file://") {
            std::fs::read_to_string(get_file_path(&input_text)?)?
        } else {
            input_text
        };

        let start = Instant::now();
        let mut model = LlamaCppModel::init_with_prompt_renderer(
            gguf_path,
            ctx_size,
            threads,
            Some(hy_mt2::render_prompt),
        )?;
        println!("Time elapsed in load model is: {:?}", start.elapsed());

        // 参数按 Hy-MT2 官方推荐值：temperature 0.7 / top_p 0.6 / top_k 20 / repeat 1.05。
        let message = serde_json::json!({
            "model": "hy-mt2",
            "messages": [{ "role": "user", "content": target_text }],
            "temperature": hy_mt2::DEFAULT_TEMPERATURE,
            "top_p": hy_mt2::DEFAULT_TOP_P,
            "top_k": hy_mt2::DEFAULT_TOP_K,
            "repeat_penalty": hy_mt2::DEFAULT_REPEAT_PENALTY,
            "max_tokens": 4096
        });
        let mes = serde_json::from_value(message)?;

        let start = Instant::now();
        let result = model.generate(mes)?;
        println!("Time elapsed in generate is: {:?}", start.elapsed());
        println!("Result: {:?}", result);

        if let Some(out) = output {
            std::fs::write(out, format!("{result:?}"))?;
            println!("Output saved to: {}", out);
        }

        Ok(())
    }
}

//! GPT-SoVITS exec implementation for CLI `run` subcommand
//!
//! ```bash
//! aha run -m gpt-sovits-v2pro-zh \
//!   --weight-path D:/model_download/gpt-sovits-v2pro-zh \
//!   -i "你好，今天天气不错。"
//! ```

use std::time::Instant;

use anyhow::Result;

use crate::exec::ExecModel;
use crate::models::gpt_sovits::GptSovitsModel;
use crate::utils::get_file_path;

pub struct GptSovitsExec;

impl GptSovitsExec {
    /// `device` 取自 `--device`（auto / cpu / cuda）
    pub fn run_with_device(
        input: &[String],
        output: Option<&str>,
        weight_path: &str,
        device: &str,
    ) -> Result<()> {
        let device = crate::utils::resolve_device(device)?;
        Self::run_inner(input, output, weight_path, device.as_ref())
    }
}

impl ExecModel for GptSovitsExec {
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        Self::run_inner(input, output, weight_path, None)
    }
}

impl GptSovitsExec {
    fn run_inner(
        input: &[String],
        output: Option<&str>,
        weight_path: &str,
        device: Option<&candle_core::Device>,
    ) -> Result<()> {
        let raw = input
            .first()
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("需要文本输入，例如 `-i \"你好\"`"))?;
        // 支持 file:// 从文件读文本
        let text = if raw.starts_with("file://") {
            let path = get_file_path(&raw)?;
            std::fs::read_to_string(path)?
        } else {
            raw
        };
        let text = text.trim();
        if text.is_empty() {
            anyhow::bail!("输入文本为空");
        }

        let load_start = Instant::now();
        let model = GptSovitsModel::init(weight_path, device, None)?;
        println!("加载模型耗时: {:?}", load_start.elapsed());
        println!(
            "音色: {:?}，采样率: {} Hz",
            model.voices(),
            model.sample_rate()
        );

        let gen_start = Instant::now();
        let audio = model.synthesize(text, None, 1.0)?;
        let elapsed = gen_start.elapsed();
        let samples = audio.dim(2)?;
        let seconds = samples as f32 / model.sample_rate() as f32;
        println!(
            "合成耗时: {elapsed:?}（{seconds:.2} s 音频，RTF {:.2}）",
            elapsed.as_secs_f32() / seconds.max(1e-6)
        );

        let output_path = match output {
            Some(path) => path.to_string(),
            None => {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)?
                    .as_secs();
                format!("gpt_sovits_{ts}.wav")
            }
        };
        crate::utils::audio_utils::save_wav_mono(
            &audio.squeeze(0)?,
            &output_path,
            model.sample_rate() as u32,
        )?;
        println!("输出已保存: {output_path}");
        Ok(())
    }
}

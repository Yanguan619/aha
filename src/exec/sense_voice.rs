//! SenseVoice exec implementation for the CLI `run` subcommand.

use std::time::Instant;

use anyhow::{Ok, Result};

use crate::exec::ExecModel;
use crate::models::GenerateModel;
use crate::models::sense_voice::SenseVoiceModel;

pub struct SenseVoiceExec;

impl ExecModel for SenseVoiceExec {
    /// Usage: `aha run -m <sense-voice> --weight-path <dir> -- <audio path|url>`
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let url = input
            .first()
            .ok_or_else(|| anyhow::anyhow!("sense-voice requires an audio path or URL"))?;
        let input_url = if url.starts_with("http://") || url.starts_with("https://") {
            url.clone()
        } else {
            // `file://relative/path` is parsed as a network share on Windows, so
            // always hand the audio loader an absolute path.
            let raw = url.strip_prefix("file://").unwrap_or(url);
            let absolute = std::fs::canonicalize(raw).unwrap_or_else(|_| raw.into());
            format!("file://{}", absolute.display())
        };

        let i_start = Instant::now();
        let mut model = SenseVoiceModel::init_from_dir(weight_path, 4)?;
        println!("Time elapsed in load model is: {:?}", i_start.elapsed());

        let message = serde_json::json!({
            "model": "sense-voice",
            "messages": [{
                "role": "user",
                "content": [{
                    "type": "audio",
                    "audio_url": { "url": input_url }
                }]
            }]
        });
        let mes = serde_json::from_value(message)?;

        let i_start = Instant::now();
        let res = model.generate(mes)?;
        println!("Time elapsed in generate is: {:?}", i_start.elapsed());

        println!("Result: {:?}", res);
        if let Some(out) = output {
            std::fs::write(out, format!("{:?}", res))?;
            println!("Output saved to: {}", out);
        }
        Ok(())
    }
}

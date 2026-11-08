//! Kokoro exec implementation for the CLI `run` subcommand.

use std::time::Instant;

use anyhow::{Ok, Result};

use crate::exec::ExecModel;
use crate::models::GenerateModel;
use crate::models::kokoro::KokoroTtsModel;
use crate::utils::audio_utils::extract_and_save_audio_from_response;

pub struct KokoroExec;

impl ExecModel for KokoroExec {
    /// Usage: `aha run -m <kokoro> --weight-path <dir> -- "要合成的文本"`
    fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
        let text = input
            .first()
            .ok_or_else(|| anyhow::anyhow!("kokoro requires the text to synthesize"))?
            .clone();

        let i_start = Instant::now();
        let mut model = KokoroTtsModel::init(weight_path, 4, crate::models::kokoro::DEFAULT_SID)?;
        println!("Time elapsed in load model is: {:?}", i_start.elapsed());

        let message = serde_json::json!({
            "model": "kokoro",
            "messages": [{"role": "user", "content": text}],
        });
        let mes = serde_json::from_value(message)?;

        let i_start = Instant::now();
        let res = model.generate(mes)?;
        println!("Time elapsed in generate is: {:?}", i_start.elapsed());

        let dir = output.unwrap_or("./");
        let paths = extract_and_save_audio_from_response(&res, dir)?;
        for path in paths {
            println!("saved audio: {}", path);
        }
        Ok(())
    }
}

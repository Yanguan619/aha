//! VoxCPM exec implementation for CLI `run` subcommand

use std::time::Instant;

use anyhow::{Ok, Result};

use aha::models::voxcpm::generate::VoxCPMGenerate;
use aha::utils::get_file_path;

pub fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
    let input_text = &input[0];
    let target_text = if input_text.starts_with("file://") {
        // let path = &input[7..];
        let path = get_file_path(input_text)?;
        std::fs::read_to_string(path)?
    } else {
        input_text.clone()
    };

    let i_start = Instant::now();
    let mut voxcpm_generate = VoxCPMGenerate::init(weight_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let i_start = Instant::now();
    let audio = voxcpm_generate.inference(
        target_text,
        // Some("啥子小师叔，打狗还要看主人，你再要继续，我就是你的对手".to_string()), // todo args
        // Some("file://./assets/audio/voice_01.wav".to_string()),                     // todo args
        None,
        None,
        2,
        100, // max_len (voxcpm uses 100 vs OpenBMB/VoxCPM1.5's 4096)
        10,
        2.0,
        // retry_badcase：目标时长不够时按比例放宽 max_len（与 GenerateModel 的默认一致）
        true,
        6.0,
    )?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);

    let output_path = if let Some(out) = output {
        out.to_string()
    } else {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        format!("voxcpm_{}.wav", timestamp)
    };

    let sample_rate = voxcpm_generate.sample_rate();
    aha::utils::audio_utils::save_wav_mono(&audio, &output_path, sample_rate as u32)?;

    println!("Output saved to: {}", output_path);

    Ok(())
}

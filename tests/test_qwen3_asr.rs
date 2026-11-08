use std::{pin::pin, time::Instant};

use aha::models::{GenerateModel, qwen3_asr::generate::Qwen3AsrGenerateModel};
use aha::params::chat::ChatCompletionParameters;
use anyhow::Result;
use futures_util::StreamExt;
#[test]
fn qwen3_asr_generate() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda --test test_qwen3_asr qwen3_asr_generate -r -- --nocapture
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Qwen/Qwen3-ASR-0.6B/", save_dir); //Qwen/Qwen3-ASR-1.7B
    let message = r#"
    {
        "model": "qwen3-asr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": 
                        {
                            "url": "file://./assets/audio/voice_01.wav"
                        }
                    }
                ]
            }
        ]
    }
    "#;
    // "metadata": {"language": "Chinese"}
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut model = Qwen3AsrGenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let res = model.generate(mes)?;
    println!("generate: \n {:?}", res);
    if let Some(usage) = &res.usage {
        println!("usage: \n {:?}", usage);
    }
    Ok(())
}

#[tokio::test]
async fn qwen3_asr_stream() -> Result<()> {
    // RUST_BACKTRACE=1 cargo test -F cuda qwen3_asr_stream -r -- --nocapture
    let save_dir =
        aha::utils::get_default_save_dir().ok_or(anyhow::anyhow!("Failed to get save dir"))?;
    let model_path = format!("{}/Qwen/Qwen3-ASR-0.6B/", save_dir);
    let message = r#"
    {
        "model": "qwen3-asr",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "audio_url": 
                        {
                            "url": "https://package-release.coderbox.cn/aiway/test/other/%E5%93%AA%E5%90%92.wav"
                        }
                    }
                ]
            }
        ]
    }
    "#;
    let mes: ChatCompletionParameters = serde_json::from_str(message)?;
    let i_start = Instant::now();
    let mut fun_asr_model = Qwen3AsrGenerateModel::init(&model_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);
    let i_start = Instant::now();
    let mut stream = pin!(fun_asr_model.generate_stream(mes)?);
    while let Some(item) = stream.next().await {
        println!("generate: \n {:?}", item);
    }
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);
    Ok(())
}

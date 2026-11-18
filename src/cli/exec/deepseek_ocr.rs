//! DeepSeek-OCR exec implementation for CLI `run` subcommand

use std::time::Instant;

use anyhow::{Ok, Result};

use aha::models::{GenerateModel, deepseek_ocr::generate::DeepseekOCRGenerateModel};

pub fn run(input: &[String], output: Option<&str>, weight_path: &str) -> Result<()> {
    let url = &input[0];
    let input_url = if url.starts_with("http://")
        || url.starts_with("https://")
        || url.starts_with("file://")
    {
        url.clone()
    } else {
        format!("file://{}", url)
    };

    let i_start = Instant::now();
    let mut model = DeepseekOCRGenerateModel::init(weight_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let message = format!(
        r#"{{
        "model": "deepseek-ai/DeepSeek-OCR",
        "messages": [
            {{
                "role": "user",
                "content": [
                    {{
                        "type": "image",
                        "image_url": {{
                            "url": "{}"
                        }}
                    }},
                    {{
                        "type": "text", 
                        "text": "<image>\nConvert the document to markdown. "
                    }}
                ]
            }}
        ]
    }}"#,
        input_url
    );
    let mes = serde_json::from_str(&message)?;

    let i_start = Instant::now();
    let result = model.generate(mes)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);

    println!("Result: {:?}", result);

    if let Some(out) = output {
        std::fs::write(out, format!("{:?}", result))?;
        println!("Output saved to: {}", out);
    }

    Ok(())
}

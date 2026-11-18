//! PaddleOCR-VL exec implementation for CLI `run` subcommand

use std::time::Instant;

use anyhow::{Ok, Result};

use aha::models::{GenerateModel, paddleocr_vl::generate::PaddleOCRVLGenerateModel};

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
    let mut model = PaddleOCRVLGenerateModel::init(weight_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    let message = format!(
        r#"{{
        "model": "PaddlePaddle/PaddleOCR-VL",
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
                        "text": "OCR:"
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

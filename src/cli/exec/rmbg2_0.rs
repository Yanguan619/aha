//! RMBG2.0 exec implementation for CLI `run` subcommand

use std::time::Instant;

use anyhow::{Ok, Result};

use aha::models::rmbg2_0::generate::RMBG2_0Model;

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
    let model = RMBG2_0Model::init(weight_path, None, None)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in load model is: {:?}", i_duration);

    // Create ChatCompletionParameters for image background removal
    let message = format!(
        r#"{{
        "model": "AI-ModelScope/RMBG-2.0",
        "messages": [
            {{
                "role": "user",
                "content": [
                    {{
                        "type": "image",
                        "image_url": {{
                            "url": "{}"
                        }}
                    }}
                ]
            }}
        ]
    }}"#,
        input_url
    );
    let mes = serde_json::from_str(&message)?;

    let i_start = Instant::now();
    let result = model.inference(mes)?;
    let i_duration = i_start.elapsed();
    println!("Time elapsed in generate is: {:?}", i_duration);

    let output_path = if let Some(out) = output {
        out.to_string()
    } else {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        format!("rmbg_{}.png", timestamp)
    };

    // Save all result images
    for (i, img) in result.iter().enumerate() {
        let path = if result.len() == 1 {
            output_path.clone()
        } else {
            format!("{}_{}.png", output_path.trim_end_matches(".png"), i)
        };
        img.save(&path)?;
        println!("Output saved to: {}", path);
    }

    Ok(())
}

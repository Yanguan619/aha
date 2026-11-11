//! VLA 的抓图路径 vs 逐算子路径，同一份输入下动作必须一致。
//!
//! 抓图机制在 `models/common/cuda_graph.rs`（GPT-SoVITS 的 AR 也用它），
//! 这个测试是 VLA 那侧的守门员。设 `AHA_VLA_DIR` 才跑（要加载两次模型）。

use aha::models::turbovla::TurboVlaModel;
use anyhow::Result;
use candle_core::{DType, Device};

fn image() -> Result<Vec<u8>> {
    let mut img = image::RgbImage::new(256, 256);
    for (x, y, px) in img.enumerate_pixels_mut() {
        *px = image::Rgb([(x % 256) as u8, (y % 256) as u8, 128]);
    }
    let mut png = Vec::new();
    image::DynamicImage::ImageRgb8(img).write_to(
        &mut std::io::Cursor::new(&mut png),
        image::ImageFormat::Png,
    )?;
    Ok(png)
}

#[test]
fn vla_graph_matches_eager() -> Result<()> {
    let Ok(dir) = std::env::var("AHA_VLA_DIR") else {
        println!("跳过：未设置 AHA_VLA_DIR");
        return Ok(());
    };
    let Ok(device) = Device::new_cuda(0) else {
        println!("跳过：没有 CUDA");
        return Ok(());
    };
    let png = image()?;
    let views: [&[u8]; 2] = [&png, &png];
    let instruction = "put the bowl on the plate";

    let mut results = Vec::new();
    for graph in ["0", "1"] {
        unsafe { std::env::set_var("TURBOVLA_GRAPH", graph) };
        let model = TurboVlaModel::load(&dir, &device, DType::BF16)?;
        let state = vec![0f32; model.state_dim()];
        let _ = model.predict(&views, instruction, &state)?; // 抓图/预热
        let t = std::time::Instant::now();
        let actions = model.predict(&views, instruction, &state)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let rows = actions.squeeze(0)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
        println!("graph={graph}: {ms:.0}ms horizon={}", rows.len());
        results.push(rows);
        drop(model);
    }

    let (a, b) = (&results[0], &results[1]);
    assert_eq!(a.len(), b.len());
    let mut max_diff = 0f32;
    for (ra, rb) in a.iter().zip(b.iter()) {
        for (x, y) in ra.iter().zip(rb.iter()) {
            max_diff = max_diff.max((x - y).abs());
        }
    }
    println!("max_diff={max_diff:.6}");
    assert!(max_diff < 1e-2, "两条路径动作差太多：{max_diff}");
    Ok(())
}

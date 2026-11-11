//! TurboVLA 前向基准：同一张图跑 N 次，报单次耗时 + 首个动作行（跨版本对比用）。
//!
//! ```powershell
//! cargo run --release --features vla        --example turbovla_bench -- <模型目录> 20 --cpu
//! cargo run --release --features vla,cuda   --example turbovla_bench -- <模型目录> 50
//! ```

use aha::models::turbovla::TurboVlaModel;
use candle_core::{DType, Device};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .unwrap_or_else(|| r"E:\models\turbovla\turbovla-libero".to_string());
    let mut iters = 10usize;
    let mut cpu = false;
    let mut f32 = false;
    let mut verify = false;
    let mut no_stages = false;
    let mut interleave = false;
    for a in args {
        match a.as_str() {
            "--cpu" => cpu = true,
            "--f32" => f32 = true,
            "--verify" => verify = true,
            "--no-stages" => no_stages = true,
            "--interleave" => interleave = true,
            s => iters = s.parse().unwrap_or(iters),
        }
    }

    // CUDA 走 bf16（f16 会在掩码/极值上溢出成 NaN，见 doc/VLA-AHA.md）。
    #[cfg(feature = "cuda")]
    let (device, dtype) = if cpu {
        (Device::Cpu, DType::F32)
    } else if f32 {
        (Device::new_cuda(0)?, DType::F32)
    } else {
        (Device::new_cuda(0)?, DType::BF16)
    };
    #[cfg(not(feature = "cuda"))]
    let (device, dtype) = {
        let _ = (cpu, f32);
        (Device::Cpu, DType::F32)
    };

    let model = TurboVlaModel::load(&dir, &device, dtype)?;

    // 确定性图案，两个视角共用
    let make_png = |seed: u8| -> anyhow::Result<Vec<u8>> {
        let mut img = image::RgbImage::new(256, 256);
        for (x, y, px) in img.enumerate_pixels_mut() {
            *px = image::Rgb([
                ((x as u8).wrapping_add(seed)),
                ((y as u8).wrapping_mul(3).wrapping_add(seed)),
                128u8.wrapping_add(seed),
            ]);
        }
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img).write_to(
            &mut std::io::Cursor::new(&mut png),
            image::ImageFormat::Png,
        )?;
        Ok(png)
    };
    let png = make_png(0)?;
    let state = vec![0f32; model.state_dim()];
    let images: Vec<&[u8]> = vec![&png; model.num_views()];

    let instruction = "put the bowl on the plate";

    // 校验：同一帧、以及换帧/换 state 后，图和逐算子结果必须一致
    if verify {
        // SAFETY: 单线程 bench，没有其他线程在读环境变量。
        unsafe {
            std::env::set_var("TURBOVLA_GRAPH", "0");
        }
        let plain = TurboVlaModel::load(&dir, &device, dtype)?;
        unsafe {
            std::env::set_var("TURBOVLA_GRAPH", "1");
        }
        let mut worst = 0f32;
        for k in 0..3u8 {
            let p = make_png(k * 7)?;
            let imgs: Vec<&[u8]> = vec![&p; model.num_views()];
            let mut st = vec![0f32; model.state_dim()];
            if k > 0 {
                st[0] = k as f32 * 0.25;
                st[3] = -(k as f32);
            }
            let a = model.predict(&imgs, instruction, &st)?;
            let b = plain.predict(&imgs, instruction, &st)?;
            let d = (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            println!("  帧 {k}: 图 vs 逐算子 最大差 {d:.3e}");
            worst = worst.max(d);
        }
        println!("校验：最大差 {worst:.3e}");
    }

    // 预热：第一次会触发 kernel 编译/权重驻留
    let actions = model.predict(&images, instruction, &state)?;

    // 分段耗时：跑 5 次取每段最小值（这台机器 GPU 频率抖动大，均值没意义）
    if !cpu && !no_stages {
        let mut best: Vec<(&'static str, f64)> = Vec::new();
        for _ in 0..5 {
            let (_, laps) = model.stages_timed(&images, instruction, &state)?;
            if best.is_empty() {
                best = laps;
            } else {
                for (i, (_, ms)) in laps.iter().enumerate() {
                    if *ms < best[i].1 {
                        best[i].1 = *ms;
                    }
                }
            }
        }
        let line: Vec<String> = best
            .iter()
            .map(|(n, ms)| format!("{n} {ms:.2}ms"))
            .collect();
        println!("min×5  {}", line.join(" | "));
    }

    let mut times = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        let out = model.predict(&images, instruction, &state)?;
        // 必须同步：图模式下 launch 是异步的，不同步量到的是入队时间。
        out.to_dtype(DType::F32)?.to_vec3::<f32>()?;
        device.synchronize()?;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = times.iter().sum::<f64>() / times.len() as f64;
    println!(
        "{} / {}：中位 {:.2}ms，均值 {:.2}ms，最小 {:.2}ms，最大 {:.2}ms（{} 次）",
        if cpu { "cpu" } else { "gpu" },
        match dtype {
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            _ => "f32",
        },
        times[times.len() / 2],
        mean,
        times[0],
        times[times.len() - 1],
        times.len()
    );
    let first = actions.squeeze(0)?.narrow(0, 0, 1)?.to_dtype(DType::F32)?.to_vec2::<f32>()?;
    println!("action[0] = {:?}", first[0]);

    // 交错指令：验证多槽 graph 缓存（桌宠的指令会来回切换）。
    //
    // 单槽时「指令 A → 指令 B → 指令 A」每次都重抓（~100ms）；
    // 多槽时第二次回到 A 应该命中，延迟回到稳态。
    if interleave {
        println!("\n--- 交错指令（3 条轮换，各 3 轮）---");
        let instrs = [
            "put the bowl on the plate",
            "open the top drawer",
            "move the mug to the left",
        ];
        for round in 0..3 {
            let mut row = Vec::with_capacity(instrs.len());
            for ins in instrs {
                let t = Instant::now();
                let out = model.predict(&images, ins, &state)?;
                out.to_dtype(DType::F32)?.to_vec3::<f32>()?;
                device.synchronize()?;
                row.push(format!("{:.1}", t.elapsed().as_secs_f64() * 1e3));
            }
            println!("  第 {} 轮：{} ms", round + 1, row.join(" / "));
        }
        println!("  （第 1 轮是各自首次捕获；第 2 轮起应回到稳态说明多槽命中）");
    }
    Ok(())
}

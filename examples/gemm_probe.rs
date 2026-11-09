//! 临时探针：同一批形状下 candle 的 GEMM 有多快（和 tools/vla/probe_gemm.py 对照）。
//!
//! cargo run --release --features vla,cuda --example gemm_probe

use candle_core::{DType, Device, Tensor};

fn bench(name: &str, f: impl Fn() -> anyhow::Result<Tensor>) -> anyhow::Result<()> {
    for _ in 0..20 {
        f()?;
    }
    let dev = Device::new_cuda(0)?;
    dev.synchronize()?;
    let n = 200;
    let t = std::time::Instant::now();
    for _ in 0..n {
        f()?;
    }
    dev.synchronize()?;
    println!("{name}  {:.3} ms", t.elapsed().as_secs_f64() * 1e3 / n as f64);
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let dt = DType::BF16;
    for (m, k, n) in [
        (522, 768, 3072), // ViT MLP up
        (522, 768, 768),  // ViT QKV 的单个投影
        (522, 3072, 768), // ViT MLP down
        (1024, 256, 512), // 融合层 vision 侧投影
        (2048, 768, 768), // 322 tokens × 4 视角的等效大 batch
    ] {
        let a = Tensor::rand(-1f32, 1f32, (m, k), &dev)?.to_dtype(dt)?;
        let b = Tensor::rand(-1f32, 1f32, (n, k), &dev)?.to_dtype(dt)?;
        let bt = b.t()?.contiguous()?;
        bench(&format!("[{m},{k}]x[{k},{n}]"), || Ok(a.matmul(&bt)?))?;
    }
    Ok(())
}

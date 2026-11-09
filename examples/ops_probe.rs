//! 临时探针：把 DINOv3 一层里「GEMM 之外」的开销逐项量出来。
//! 形状按实际：2 视角合批 = 522 token，12 头，head_dim 64，261 个 key。
//!
//! cargo run --release --features vla,cuda --example ops_probe

use candle_core::{DType, Device, Tensor};

const B: usize = 2;
const S: usize = 261;
const H: usize = 12;
const D: usize = 64;
const E: usize = 768;

fn rnd(shape: impl Into<candle_core::Shape>, dev: &Device, dt: DType) -> Tensor {
    Tensor::rand(-1f32, 1f32, shape, dev)
        .unwrap()
        .to_dtype(dt)
        .unwrap()
}

fn bench(dev: &Device, name: &str, f: impl Fn() -> anyhow::Result<Tensor>) -> anyhow::Result<f64> {
    for _ in 0..20 {
        f()?;
    }
    dev.synchronize()?;
    let n = 200;
    let t = std::time::Instant::now();
    for _ in 0..n {
        f()?;
    }
    dev.synchronize()?;
    let ms = t.elapsed().as_secs_f64() * 1e3 / n as f64;
    println!("{name:<36} {ms:7.4} ms");
    Ok(ms)
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let dt = DType::BF16;

    let x = rnd((B * S, E), &dev, dt); // (522,768) 一层输入
    let qkv = rnd((B * H, S, D), &dev, dt); // 拆头后的 q
    let k = rnd((B * H, S, D), &dev, dt);
    let lnw = rnd(E, &dev, dt);
    let lnb = rnd(E, &dev, dt);
    let y = rnd((B * S, E), &dev, dt);
    let g = rnd(E, &dev, dt);

    // 1. 拆头：reshape+transpose+contiguous（每层 q/k/v 各一次）
    let t_split = bench(&dev, "split_heads 1x (contiguous)", || {
        Ok(x.reshape((B, S, H, D))?.transpose(1, 2)?.contiguous()?)
    })?;

    // 2. qk^T：24 个 261x261x64 的 batched GEMM
    let t_qk = bench(&dev, "qk^T (24,261,64)x(24,64,261)", || {
        Ok(qkv.matmul(&k.transpose(1, 2)?.contiguous()?)?)
    })?;

    // 3. bf16 -> f32 往返（视觉层 softmax 在 f32 里做）
    let l32 = qkv.matmul(&k.transpose(1, 2)?.contiguous()?)?.to_dtype(DType::F32)?;
    let t_cast = bench(&dev, "to_dtype F32 再回 bf16", || {
        Ok(l32.to_dtype(dt)?.to_dtype(DType::F32)?)
    })?;

    // 4. 融合 softmax（cuda_fwd）
    let t_sm = bench(&dev, "softmax_last_dim (f32)", || {
        Ok(candle_nn::ops::softmax_last_dim(&l32)?)
    })?;

    // 5. pv
    let p = candle_nn::ops::softmax_last_dim(&l32)?;
    let pb = p.to_dtype(dt)?;
    let t_pv = bench(&dev, "pv + 回 bf16", || Ok(pb.matmul(&qkv)?))?;

    // 6. LayerNorm（bf16，融合版）
    let t_ln = bench(&dev, "layer_norm (522,768) bf16", || {
        Ok(candle_nn::ops::layer_norm(&x, &lnw, &lnb, 1e-5)?)
    })?;

    // 7. 残差 + LayerScale + GELU
    let t_el = bench(&dev, "add / mul / gelu 各一次", || {
        let a = (&x + &y)?;
        let b = a.broadcast_mul(&g)?;
        Ok(b.gelu()?)
    })?;

    let glue = t_split * 3.0 + t_cast + t_sm + t_pv;
    let per_layer = glue + t_ln * 2.0 + t_el * 3.0;
    println!("\n一层（非 GEMM）≈ {per_layer:.3} ms  → 12 层 ≈ {:.2} ms", per_layer * 12.0);
    println!("其中 qk^T {t_qk:.4} / pv {t_pv:.4}（小 GEMM），其余全是搬运");
    Ok(())
}

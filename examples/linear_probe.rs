//! 临时探针：融合 Linear（cublasLt bias epilogue）vs candle 的 matmul+add。
//!
//! cargo run --release --features vla,cuda --example linear_probe

use aha::models::turbovla::linear::Linear;
use candle_core::{DType, Device, Module, Tensor};

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let dt = DType::F32;
    for (m, k, n) in [(522usize, 768usize, 3072usize), (12, 256, 1024), (21, 768, 768)] {
        let x = Tensor::rand(-1f32, 1f32, (m, k), &dev)?.to_dtype(dt)?;
        let w = Tensor::rand(-1f32, 1f32, (n, k), &dev)?.to_dtype(dt)?;
        let b = Tensor::rand(-1f32, 1f32, n, &dev)?.to_dtype(dt)?;

        let got = Linear::new(w.clone(), Some(b.clone())).forward(&x)?;
        let want = x.matmul(&w.t()?)?.broadcast_add(&b)?;

        let a = want.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let g = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let mut worst = 0f32;
        let mut at = 0;
        for i in 0..a.len() {
            let d = (a[i] - g[i]).abs();
            if d > worst {
                worst = d;
                at = i;
            }
        }
        println!("(m={m},k={k},n={n})  max diff {worst:.3e} @{at}  want={} got={}", a[at], g[at]);
        println!("   want[0..4] {:?}", &a[..4]);
        println!("   got [0..4] {:?}", &g[..4]);
        println!("   want[{}..{}] {:?}", n, n+3, &a[n..n+3]);
        println!("   got [{}..{}] {:?}", n, n+3, &g[n..n+3]);
    }
    Ok(())
}

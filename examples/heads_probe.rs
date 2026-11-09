//! 临时探针：拆头/合头专用 kernel vs candle 的 transpose+contiguous。
//!
//! cargo run --release --features vla,cuda --example heads_probe

use aha::models::turbovla::heads_cuda;
use candle_core::{DType, Device, Tensor};

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let dt = DType::BF16;
    let (b, s, h, d) = (2usize, 7usize, 3usize, 4usize);

    let x = Tensor::rand(-1f32, 1f32, (b, s, h * d), &dev)?.to_dtype(dt)?;
    let want = x.reshape((b, s, h, d))?.transpose(1, 2)?.contiguous()?;
    let x4 = x.reshape((b, s, h, d))?;
    let got = heads_cuda::split_heads(&x4, h)?;
    let diff = (got.to_dtype(DType::F32)? - want.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    println!("split  dims {:?}/{:?}  max diff {diff:.3e}", got.dims(), want.dims());

    let back = heads_cuda::merge_heads(&want)?;
    let diff2 = (&back.to_dtype(DType::F32)? - &x.to_dtype(DType::F32)?)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    println!("merge  dims {:?}  max diff vs 原始 {diff2:.3e}", back.dims());
    Ok(())
}

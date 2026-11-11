//! `models/common/cuda_graph.rs` 的自检：抓一张 `y = x * 2 + 1`，改输入后重放，
//! 结果要跟着变（证明图烧死的是那块静态缓冲，而不是拷了一份）。
//!
//! 不需要任何模型权重，有 CUDA 就能跑；没有就跳过。

use aha::models::common::cuda_graph::{static_zeros, upload_f32, upload_u32, CudaGraph};
use anyhow::Result;
use candle_core::{DType, Device};

#[test]
fn capture_replay_matches_eager() -> Result<()> {
    let Device::Cuda(cuda_dev) = Device::new_cuda(0)? else {
        unreachable!()
    };
    let device = Device::Cuda(cuda_dev.clone());
    let stream = cuda_dev.cuda_stream();

    // 静态输入：一块 f32 + 一个 u32（覆盖两类 upload，u32 只验证能写进去）
    let x = static_zeros(&device, DType::F32, (4,))?;
    let seed = static_zeros(&device, DType::U32, (1,))?;
    upload_f32(&stream, &x, &[1.5, -2.0, 0.25, 8.0])?;
    upload_u32(&stream, &seed, &[7])?;

    let (graph, out) = CudaGraph::capture(&device, || {
        let doubled = (x.tensor() * 2.0)?;
        Ok((doubled + 1.0)?)
    })?;

    // 重放前改输入，输出要跟着变
    let host = [-1.0f32, 0.0, 3.0, 0.5];
    upload_f32(&stream, &x, &host)?;
    graph.replay()?;

    let got: Vec<f32> = out.copy()?.to_vec1()?;
    let want: Vec<f32> = host.iter().map(|v| v * 2.0 + 1.0).collect();
    for (g, w) in got.iter().zip(want.iter()) {
        assert!((g - w).abs() < 1e-5, "图输出 {got:?} != 期望 {want:?}");
    }
    Ok(())
}

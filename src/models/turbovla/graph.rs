//! TurboVLA 的 CUDA graph 封装。
//!
//! 目的：一帧里大半时间花在「几百个小 kernel 的启动 + 逐算子分配」上，图能把这段摊掉。
//!
//! 抓图/重放/静态缓冲那套通用机制在 [`crate::models::common::cuda_graph`]（GPT-SoVITS 的
//! AR 也用它）。这里只剩 VLA 特有的部分：两个固定输入（像素 `(1, V, 3, H, W)`、状态
//! `(1, state_dim)`），每帧把新数据写进原地缓冲再重放。
//!
//! 图内外的分工是**类型**说了算：`compute` 只拿得到 `&StaticTensor`（图烧死了地址），
//! 图外的图像解码、文本特征、动作后处理都是普通 `Tensor`。

use anyhow::{bail, Result};
use candle_core::{DType, Device, Tensor};

use crate::models::common::cuda_graph::{static_zeros, upload_f32, CudaGraph, StaticTensor};

/// 一个已被捕获的图 + 它的静态输入/输出。
pub struct GraphRunner {
    graph: CudaGraph,
    /// 图输入（地址烧死，每帧原地写）
    pixel: StaticTensor,
    state: StaticTensor,
    /// 捕获那次返回的动作张量，重放时图会原地写它指向的显存。
    out: Tensor,
}

impl GraphRunner {
    /// 捕获一次 `compute(pixel, state)`。
    ///
    /// `pixel_shape` 是 `(1, V, 3, H, W)`，`state_shape` 是 `(1, state_dim)`；
    /// 传入的主机数据会先写进静态缓冲，再拿它做预热和捕获。
    pub fn try_capture(
        device: &Device,
        dtype: DType,
        pixel_shape: (usize, usize, usize, usize, usize),
        pixel_host: &[f32],
        state_shape: (usize, usize),
        state_host: &[f32],
        compute: impl Fn(&StaticTensor, &StaticTensor) -> Result<Tensor>,
    ) -> Result<Self> {
        let pixel = static_zeros(device, dtype, pixel_shape)?;
        let state = static_zeros(device, dtype, state_shape)?;
        // 捕获前先写真实数据：图烧死的是这两个地址，预热/捕获跑的就是它们的内容。
        let stream = match device {
            Device::Cuda(dev) => dev.cuda_stream(),
            _ => bail!("CUDA graph 只能在 cuda 设备上用"),
        };
        upload_f32(&stream, &pixel, pixel_host)?;
        upload_f32(&stream, &state, state_host)?;

        let (graph, out) = CudaGraph::capture(device, || compute(&pixel, &state))?;

        Ok(Self {
            graph,
            pixel,
            state,
            out,
        })
    }

    /// 把新一帧的数据写进静态缓冲（dtype 转换在主机做，然后 H2D）。
    pub fn upload(&self, pixel: &[f32], state: &[f32]) -> Result<()> {
        upload_f32(self.graph.stream(), &self.pixel, pixel)?;
        upload_f32(self.graph.stream(), &self.state, state)
    }

    /// 重放一次，返回动作的**深拷贝**。
    ///
    /// 必须拷贝：`out` 指向图私有内存池里的块，下一轮 launch 会覆写它。
    pub fn replay(&self) -> Result<Tensor> {
        self.graph.replay()?;
        Ok(self.out.copy()?)
    }
}

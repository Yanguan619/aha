//! AR 逐 token 解码的 CUDA graph（对齐官方 `Accel` 的 static cuda graph 后端）。
//!
//! 官方 `GPT_SoVITS/Accel/PyTorch/AR/backends/torch_static_cuda_graph.py` 的做法：
//! 只把「24 层 transformer 一步」抓进图里，head / 采样 / 位置编码仍在图外逐 token 跑；
//! kv cache 定长、每步原位写一个位置，注意力对整段算、靠 mask 屏蔽未来。
//!
//! 抓图/重放那套通用机制在 [`crate::models::common::cuda_graph`]，这里只做 AR 特有的编排：
//!
//! 1. candle 没有 `index_copy_`，用 [`Tensor::scatter_set`]（同样是原位写）代替。
//!    注意它的索引是**写到哪一格**——整块拷贝要给 `0..H-1`，全 0 会把所有值挤到第 0 格。
//! 2. k 存成 `(1, heads, d, max_seq)`（而不是 `(1, heads, max_seq, d)`），注意力就不用
//!    每层每步转置整段 cache。
//! 3. mask 在图内由 `pos` 生成，必须用 `arange > pos`：刚写进去的那一格要可见。
//! 4. 每步图外只做两件事：把这个 token 的音频嵌入整块拷进静态 `x`、写 4 字节的位置。
//!
//! 和官方唯一的差别是**分桶**：官方所有请求都按 `max_seq_length`（默认 1024）padding，
//! 于是短句也要算 1024 个位置，端到端反而更慢（见 `doc/TTS.md` §3.6）。这里按句长
//! 选最小的桶，机制没变、只是桶大小不同。

use std::sync::Arc;

use anyhow::{bail, Result};
use candle_core::cuda_backend::cudarc::driver::CudaStream;
use candle_core::{DType, Device, Tensor};

use crate::models::common::cuda_graph::{static_zeros, upload_u32, CudaGraph, StaticTensor};

/// 一个已捕获的「AR 一步」+ 它的定长缓冲。
pub(crate) struct ArGraph {
    graph: CudaGraph,
    pub(crate) max_seq: usize,
    /// 图输入：这一步的音频嵌入 `(1, 1, hidden)`
    x: StaticTensor,
    /// 图输入：写入位置 `(1,)` u32
    pos: StaticTensor,
    /// 图输入：`(max_seq,)` u32，图里由它和 `pos` 生成 mask（建好后只占住显存）
    _arange: StaticTensor,
    /// 把 `x` 整块原地覆盖用的索引（沿 hidden 轴 0..H-1，形状同 x），只读，不是图输入
    copy_idx: Tensor,
    /// 定长 kv（图输入）：k `(1, heads, d, max_seq)`、v `(1, heads, max_seq, d)`
    k: Vec<StaticTensor>,
    v: Vec<StaticTensor>,
    /// 图输出 `(1, 1, hidden)`，在图的私有内存池里，launch 后必须立刻 copy
    out: Tensor,
    heads: usize,
    head_dim: usize,
    n_layer: usize,
}

impl ArGraph {
    /// 抓一张图。`compute(x, pos, arange, k, v)` 要跑满 24 层、返回 `(1, 1, hidden)`。
    pub(crate) fn capture<F>(
        device: &Device,
        n_layer: usize,
        heads: usize,
        head_dim: usize,
        max_seq: usize,
        compute: F,
    ) -> Result<Self>
    where
        F: Fn(&StaticTensor, &StaticTensor, &StaticTensor, &[StaticTensor], &[StaticTensor]) -> Result<Tensor>,
    {
        let stream = cuda_stream(device)?;
        let hidden = heads * head_dim;
        let x = static_zeros(device, DType::F32, (1, 1, hidden))?;
        let pos = static_zeros(device, DType::U32, (1,))?;
        // scatter_set 的索引是"写到哪一格"，整块拷贝要给 0..H-1
        let copy_idx = Tensor::arange(0u32, hidden as u32, device)?.reshape((1, 1, hidden))?;
        let arange = static_zeros(device, DType::U32, (max_seq,))?;
        // 0..max_seq 一次写死（图里拿它和 pos 比大小生成 mask）
        let positions: Vec<u32> = (0..max_seq as u32).collect();
        upload_u32(&stream, &arange, &positions)?;
        let mut k = Vec::with_capacity(n_layer);
        let mut v = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            k.push(static_zeros(device, DType::F32, (1, heads, head_dim, max_seq))?);
            v.push(static_zeros(device, DType::F32, (1, heads, max_seq, head_dim))?);
        }

        let (graph, out) = CudaGraph::capture(device, || compute(&x, &pos, &arange, &k, &v))?;

        tracing::info!("AR CUDA graph 已捕获（max_seq={max_seq}）");
        Ok(Self {
            graph,
            max_seq,
            x,
            pos,
            _arange: arange,
            copy_idx,
            k,
            v,
            out,
            heads,
            head_dim,
            n_layer,
        })
    }

    /// 把整段 prefill 出来的 kv 灌进定长缓冲（每句一次）。
    pub(crate) fn seed(&self, caches: &[(Tensor, Tensor)], len: usize) -> Result<()> {
        if caches.len() != self.n_layer {
            bail!("层数不一致：{} != {}", caches.len(), self.n_layer);
        }
        if len > self.max_seq {
            bail!("prefill 长度 {len} 超过图容量 {}", self.max_seq);
        }
        let (heads, d) = (self.heads, self.head_dim);
        // 索引：k 需要 (1, heads, d, len)、v 需要 (1, heads, len, d)，值都是 0..len
        let idx_v = Tensor::arange(0u32, len as u32, self.k[0].tensor().device())?
            .reshape((1, 1, len, 1))?
            .broadcast_as((1, heads, len, d))?
            .contiguous()?;
        let idx_k = idx_v.transpose(2, 3)?.contiguous()?;
        for (i, (kc, vc)) in caches.iter().enumerate() {
            // eager 的 cache 是 (1, len, hidden)，先拆回 (1, len, heads, d)
            // k 目标 (1, heads, d, len)，v 目标 (1, heads, len, d)
            let k_src = kc
                .reshape((1, len, heads, d))?
                .transpose(1, 2)?
                .transpose(2, 3)?
                .contiguous()?;
            let v_src = vc.reshape((1, len, heads, d))?.transpose(1, 2)?.contiguous()?;
            self.k[i].tensor().scatter_set(&idx_k, &k_src, 3)?;
            self.v[i].tensor().scatter_set(&idx_v, &v_src, 2)?;
        }
        Ok(())
    }

    /// 重放一步：写入这一步的输入（k/v 由图内原位更新），返回 `(1, 1, hidden)`。
    pub(crate) fn step(&self, xy: &Tensor, pos: usize) -> Result<Tensor> {
        if pos >= self.max_seq {
            bail!("位置 {pos} 超过图容量 {}", self.max_seq);
        }
        self.x.tensor().scatter_set(&self.copy_idx, xy, 2)?;
        upload_u32(self.graph.stream(), &self.pos, &[pos as u32])?;
        self.graph.replay()?;
        // out 在图私有池里，下一轮 launch 会覆写，必须拷出来
        Ok(self.out.copy()?)
    }
}

/// 抓图前要把常量灌进静态缓冲，需要设备 stream。
fn cuda_stream(device: &Device) -> Result<Arc<CudaStream>> {
    match device {
        Device::Cuda(dev) => Ok(dev.cuda_stream()),
        _ => bail!("CUDA graph 只能用在 cuda 设备上"),
    }
}

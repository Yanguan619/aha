//! TurboVLA 的 CUDA graph 封装。
//!
//! 目的：一帧里大半时间花在「几百个小 kernel 的启动 + 逐算子分配」上，图能把这段摊掉。
//!
//! # 为什么这么绕
//!
//! candle 没有 in-place 拷贝 API（`slice_assign` 是函数式的，返回新张量），也没有公开
//! `Tensor::storage()`。所以静态输入缓冲只能自己造：
//!
//! 1. `stream.alloc_zeros` 拿一块 `CudaSlice`，先取它的裸 `CUdeviceptr`；
//! 2. 再把 slice 交给 `CudaStorage::wrap_cuda_slice` + `Tensor::from_storage` 包成张量
//!    （张量持有所有权，保证这块显存不会被回收）；
//! 3. 每帧用 `cuMemcpyHtoDAsync` 直接往那个裸指针写新数据。
//!
//! 另外两个坑（都在 `try_capture` 里）：
//!
//! - **捕获前必须先预热一次**：`attn * scale`、LayerNorm 的 eps 这类标量会走 H2D，
//!   candle 要求先正常跑一遍把它们灌进参数缓存（`enable_cuda_graph_htod_cache`）。
//!   否则捕获时报 `h2d param cache miss ... warmup did not populate ...`。
//! - **图里会有 alloc 节点**：cudarc 在支持 memory pool 的设备上走 `cuMemAllocAsync`，
//!   分配/释放都会被录进图里，所以 instantiate 必须带 `AUTO_FREE_ON_LAUNCH`，
//!   否则第二次 launch 直接 `CUDA_ERROR_INVALID_VALUE`。

use anyhow::{anyhow, bail, Result};
use candle_core::cuda_backend::cudarc::driver::{sys, CudaGraph, CudaStream, DevicePtr};
use candle_core::cuda_backend::{CudaDevice, CudaStorage};
use candle_core::{DType, Device, Storage, Tensor};
use std::sync::Arc;

/// 一个已被捕获的图 + 它的静态输入/输出。
///
/// `CudaGraph` 本身不是 `Send`（CUDA 要求同一个 graph 对象不能并发访问）。
/// 这里靠调用方的 `Mutex` 串行化；cudarc 每次调用都会 `bind_to_thread`，
/// 所以跨线程重放是安全的。
pub struct GraphRunner {
    graph: CudaGraph,
    stream: Arc<CudaStream>,
    pixel_ptr: u64,
    state_ptr: u64,
    pixel_elems: usize,
    state_elems: usize,
    dtype: DType,
    /// 捕获那次返回的动作张量，重放时图会写它指向的显存。
    out: Tensor,
    /// 静态输入张量，用来**占住**显存（图里烧死的是它们的地址）。
    _pixel: Tensor,
    _state: Tensor,
}

unsafe impl Send for GraphRunner {}

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
        compute: impl Fn(&Tensor, &Tensor) -> Result<Tensor>,
    ) -> Result<Self> {
        let Device::Cuda(cuda_dev) = device else {
            bail!("CUDA graph 只能在 cuda 设备上用");
        };
        let stream = cuda_dev.cuda_stream();

        let (pixel, pixel_ptr) = static_tensor(cuda_dev, &stream, dtype, pixel_shape, pixel_host)?;
        let (state, state_ptr) = static_tensor(cuda_dev, &stream, dtype, state_shape, state_host)?;
        // 捕获前先写真实数据：图烧死的是这两个地址，预热/捕获跑的就是它们的内容。
        upload_one(&stream, dtype, pixel_ptr, pixel_host)?;
        upload_one(&stream, dtype, state_ptr, state_host)?;

        // 捕获期间：小参数 H2D 走 candle 的参数缓存；关掉 event 追踪，省掉跨 stream 的等待。
        let guard = cuda_dev.enable_cuda_graph_htod_cache();
        unsafe { cuda_dev.disable_event_tracking() };

        // 预热：把 compute 里那些标量 H2D 灌进缓存，捕获时必须命中。
        drop(compute(&pixel, &state)?);
        stream
            .synchronize()
            .map_err(|e| anyhow!("预热后同步失败：{e}"))?;

        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(|e| anyhow!("begin_capture 失败：{e}"))?;

        let captured = compute(&pixel, &state);

        // 无论成功失败都要结束捕获，否则 stream 会卡在 capture 状态。
        let ended = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        drop(guard);

        let out = captured?;
        let graph = ended
            .map_err(|e| anyhow!("end_capture 失败：{e}"))?
            .ok_or_else(|| anyhow!("end_capture 返回空图"))?;

        Ok(Self {
            graph,
            stream,
            pixel_ptr,
            state_ptr,
            pixel_elems: pixel_host.len(),
            state_elems: state_host.len(),
            dtype,
            out,
            _pixel: pixel,
            _state: state,
        })
    }

    /// 把新一帧的数据写进静态缓冲（dtype 转换在主机做，然后 H2D）。
    pub fn upload(&self, pixel: &[f32], state: &[f32]) -> Result<()> {
        if pixel.len() != self.pixel_elems || state.len() != self.state_elems {
            bail!(
                "输入长度变了（pixel {} != {} / state {} != {}），图不能复用",
                pixel.len(),
                self.pixel_elems,
                state.len(),
                self.state_elems
            );
        }
        upload_one(&self.stream, self.dtype, self.pixel_ptr, pixel)?;
        upload_one(&self.stream, self.dtype, self.state_ptr, state)
    }

    /// 重放一次，返回动作的**深拷贝**。
    ///
    /// 必须拷贝：`out` 指向图私有内存池里的块，下一轮 launch 会覆写它。
    pub fn replay(&self) -> Result<Tensor> {
        self.graph
            .launch()
            .map_err(|e| anyhow!("graph launch 失败：{e}"))?;
        self.out.copy().map_err(anyhow::Error::from)
    }
}

/// 把 f32 主机数据写进设备缓冲（图上烧死的那个地址）。
fn upload_one(stream: &Arc<CudaStream>, dtype: DType, dst: u64, src: &[f32]) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::result::memcpy_htod_async;
    let st = stream.cu_stream();
    match dtype {
        DType::BF16 => {
            let host: Vec<half::bf16> = src.iter().map(|v| half::bf16::from_f32(*v)).collect();
            unsafe { memcpy_htod_async(dst, &host, st) }
                .map_err(|e| anyhow!("memcpy_htod_async 失败：{e}"))
        }
        DType::F32 => unsafe { memcpy_htod_async(dst, src, st) }
            .map_err(|e| anyhow!("memcpy_htod_async 失败：{e}")),
        dt => bail!("图模式不支持 {dt:?}"),
    }
}

/// 分配一块静态缓冲，返回持有所有权的张量 + 裸设备指针。
fn static_tensor(
    dev: &CudaDevice,
    stream: &Arc<CudaStream>,
    dtype: DType,
    shape: impl Into<candle_core::Shape>,
    host: &[f32],
) -> Result<(Tensor, u64)> {
    let shape = shape.into();
    let n = shape.elem_count();
    if n != host.len() {
        bail!("静态缓冲大小不一致：{n} != {}", host.len());
    }
    let (storage, ptr) = match dtype {
        DType::BF16 => {
            let slice = stream
                .alloc_zeros::<half::bf16>(n)
                .map_err(|e| anyhow!("alloc_zeros 失败：{e}"))?;
            let (ptr, guard) = slice.device_ptr(stream);
            drop(guard);
            (CudaStorage::wrap_cuda_slice(slice, dev.clone()), ptr)
        }
        DType::F32 => {
            let slice = stream
                .alloc_zeros::<f32>(n)
                .map_err(|e| anyhow!("alloc_zeros 失败：{e}"))?;
            let (ptr, guard) = slice.device_ptr(stream);
            drop(guard);
            (CudaStorage::wrap_cuda_slice(slice, dev.clone()), ptr)
        }
        dt => bail!("图模式不支持 {dt:?}"),
    };
    let tensor = Tensor::from_storage(
        Storage::Cuda(storage),
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    );
    Ok((tensor, ptr))
}

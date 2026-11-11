//! CUDA graph 的最小公共封装。
//!
//! 用途是「同一套 kernel 图形状、跑很多次」的推理：VLA 一帧一次（每帧上百个小 kernel）、
//! GPT-SoVITS 的 AR 每个语义 token 一次。一次性的整段前向（ASR / 整句 TTS）用不上——抓图
//! 本身有开销，而每次输入形状都变，padding 的算力比省下的 launch 还多。
//!
//! candle 没有 capture/launch 的 API（也没有 in-place 拷贝、没有公开 storage），所以这里
//! 把三件必须一起做的事收在一处：
//!
//! 1. **静态缓冲**：`alloc_zeros` 拿裸地址，再用 `CudaStorage::wrap_cuda_slice` +
//!    `Tensor::from_storage` 包成张量（张量持有所有权，保证显存不会被回收）。图里烧死的是
//!    这个地址，所以之后每轮只能往原地写。
//! 2. **抓图**：先预热一次（`attn * scale`、LayerNorm eps 这类标量会走 H2D，candle 要求把
//!    它们灌进参数缓存，否则捕获时报 `h2d param cache miss`），再 begin_capture → compute →
//!    end_capture。instantiate 必须带 `AUTO_FREE_ON_LAUNCH`：图里有 alloc 节点（cudarc 在支持
//!    memory pool 的设备上走 `cuMemAllocAsync`），否则第二次 launch 直接 `CUDA_ERROR_INVALID_VALUE`。
//! 3. **重放**：launch 后把输出**深拷贝**出来——输出在图私有池里，下一轮会被覆写。
//!
//! 抓图期间不能有新的分配、也不能有 D2H（candle 会直接报错），所以采样、判停这类要回主机的
//! 动作都得留在图外。

use anyhow::{anyhow, bail, Result};
use std::sync::Arc;

use candle_core::cuda_backend::cudarc::driver::{sys, CudaGraph as RawGraph, CudaStream, DevicePtr};
use candle_core::cuda_backend::CudaStorage;
use candle_core::{DType, Device, Shape, Storage, Tensor};

/// 图输入/图状态的张量：只可能由 [`static_zeros`] 造出来。
///
/// 它的存在就是为了让"这段在图里"变成**类型**而不是注释：图把地址烧死，之后每轮只能往
/// 这批缓冲原地写；随手 `Tensor::zeros(..)` 出来的新张量地址进不了图，所以图内那一步
/// 只接受 `&StaticTensor`，传普通 `Tensor` 是编译错误。
pub struct StaticTensor {
    tensor: Tensor,
    ptr: u64,
}

impl StaticTensor {
    /// 图内的算子都走这里拿底层张量；地址不变，仍是同一个缓冲。
    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn dtype(&self) -> DType {
        self.tensor.dtype()
    }

    pub fn elems(&self) -> usize {
        self.tensor.elem_count()
    }
}

/// 一张已捕获的图。`RawGraph` 不是 `Send`（CUDA 要求同一个 graph 不能被并发访问），
/// 调用方用 `Mutex` 串行化；cudarc 每次调用都会 `bind_to_thread`，跨线程重放是安全的。
pub struct CudaGraph {
    graph: RawGraph,
    stream: Arc<CudaStream>,
}

unsafe impl Send for CudaGraph {}

impl CudaGraph {
    /// 抓一次图，返回图 + 捕获那次跑出来的输出张量（重放时图会原地写它）。
    ///
    /// `compute` 只能读写调用方事先建好的静态缓冲，**不能有新分配**，也不能把结果拷回主机。
    pub fn capture(
        device: &Device,
        compute: impl Fn() -> Result<Tensor>,
    ) -> Result<(Self, Tensor)> {
        let Device::Cuda(cuda_dev) = device else {
            bail!("CUDA graph 只能用在 cuda 设备上");
        };
        let stream = cuda_dev.cuda_stream();

        // 抓图期间：标量 H2D 走 candle 的参数缓存；关掉 event 追踪，省掉跨 stream 等待。
        let guard = cuda_dev.enable_cuda_graph_htod_cache();
        unsafe { cuda_dev.disable_event_tracking() };

        // 预热：把 compute 里的标量 H2D 灌进缓存，捕获时必须命中。
        drop(compute()?);
        stream
            .synchronize()
            .map_err(|e| anyhow!("预热后同步失败：{e}"))?;

        stream
            .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(|e| anyhow!("begin_capture 失败：{e}"))?;
        let captured = compute();
        // 无论成功失败都要结束捕获，否则 stream 会卡在 capture 状态。
        let ended = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        drop(guard);

        let out = captured?;
        let graph = ended
            .map_err(|e| anyhow!("end_capture 失败：{e}"))?
            .ok_or_else(|| anyhow!("end_capture 返回空图"))?;
        Ok((Self { graph, stream }, out))
    }

    /// 重放一次。输出在图私有池里，下一轮会被覆写，需要就立刻 `out.copy()`。
    pub fn replay(&self) -> Result<()> {
        self.graph
            .launch()
            .map_err(|e| anyhow!("graph launch 失败：{e}"))
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// 分配一块全 0 的静态缓冲，返回张量（持所有权）+ 裸地址（图里烧死的就是它）。
pub fn static_zeros(
    device: &Device,
    dtype: DType,
    shape: impl Into<Shape>,
) -> Result<StaticTensor> {
    let Device::Cuda(cuda_dev) = device else {
        bail!("CUDA graph 只能用在 cuda 设备上");
    };
    let stream = cuda_dev.cuda_stream();
    let shape = shape.into();
    let n = shape.elem_count();

    macro_rules! alloc {
        ($ty:ty) => {{
            let slice = stream
                .alloc_zeros::<$ty>(n)
                .map_err(|e| anyhow!("alloc_zeros 失败：{e}"))?;
            let (ptr, guard) = slice.device_ptr(&stream);
            drop(guard);
            let storage = CudaStorage::wrap_cuda_slice(slice, cuda_dev.clone());
            Ok(StaticTensor {
                tensor: wrap(storage, shape),
                ptr,
            })
        }};
    }

    match dtype {
        DType::F32 => alloc!(f32),
        DType::F16 => alloc!(half::f16),
        DType::BF16 => alloc!(half::bf16),
        DType::U32 => alloc!(u32),
        dt => bail!("图模式暂不支持 {dt:?} 的静态缓冲"),
    }
}

/// 把 f32 主机数据写进静态缓冲（bf16/f16 在主机侧转换）。
pub fn upload_f32(stream: &Arc<CudaStream>, dst: &StaticTensor, src: &[f32]) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::result::memcpy_htod_async;
    check_len(dst, src.len())?;
    bind(stream)?;
    let st = stream.cu_stream();
    let ptr = dst.ptr;
    match dst.dtype() {
        DType::BF16 => {
            let host: Vec<half::bf16> = src.iter().map(|v| half::bf16::from_f32(*v)).collect();
            unsafe { memcpy_htod_async(ptr, &host, st) }.map_err(|e| anyhow!("{e}"))
        }
        DType::F16 => {
            let host: Vec<half::f16> = src.iter().map(|v| half::f16::from_f32(*v)).collect();
            unsafe { memcpy_htod_async(ptr, &host, st) }.map_err(|e| anyhow!("{e}"))
        }
        DType::F32 => unsafe { memcpy_htod_async(ptr, src, st) }.map_err(|e| anyhow!("{e}")),
        dt => bail!("upload_f32 不支持 {dt:?}"),
    }
}

/// 把 u32 主机数据写进静态缓冲（索引/位置这类标量）。
pub fn upload_u32(stream: &Arc<CudaStream>, dst: &StaticTensor, src: &[u32]) -> Result<()> {
    use candle_core::cuda_backend::cudarc::driver::result::memcpy_htod_async;
    check_len(dst, src.len())?;
    bind(stream)?;
    unsafe { memcpy_htod_async(dst.ptr, src, stream.cu_stream()) }.map_err(|e| anyhow!("{e}"))
}

/// 静态缓冲的形状是图的一部分，写超了会踩到别的缓冲（或越过图容量）。
fn check_len(dst: &StaticTensor, len: usize) -> Result<()> {
    if dst.elems() != len {
        bail!("静态缓冲长度不一致：{} != {len}，图不能复用", dst.elems());
    }
    Ok(())
}

/// 服务端可能在 tokio 的任意线程上调：裸 CUDA 调用要求当前线程已绑定 stream 的上下文，
/// 否则 memcpy 报 `CUDA_ERROR_INVALID_CONTEXT`（candle 自己的算子内部会 bind，这里不会）。
fn bind(stream: &Arc<CudaStream>) -> Result<()> {
    stream
        .context()
        .bind_to_thread()
        .map_err(|e| anyhow!("绑定 CUDA 上下文失败：{e}"))
}

fn wrap(storage: CudaStorage, shape: Shape) -> Tensor {
    Tensor::from_storage(
        Storage::Cuda(storage),
        shape,
        candle_core::op::BackpropOp::none(),
        false,
    )
}

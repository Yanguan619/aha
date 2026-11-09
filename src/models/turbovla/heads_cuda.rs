//! 拆头 / 合头的专用 kernel。
//!
//! 为什么需要：`Tensor::reshape(..).transpose(1,2).contiguous()` 走的是 candle 的通用
//! `ucopy` kernel —— **逐元素算 `get_strided_index`**，实测只有 ~47 GB/s。
//! 但这两步其实是「按行块搬连续内存」：
//!
//! - split: `out[b][h][s][:]] = in[b][s][h][:]`，每次都是 `D` 个连续元素（D=64 → 128B）；
//! - merge: 反过来。
//!
//! 所以一个纯拷贝 kernel 就够，不需要做真正的 transpose（`s/d` 坐标不变，只是行距不同）。
//! nsys 上这两步合计每帧 158 次、1.74ms，是 A 之后的最大单项。

use anyhow::{anyhow, bail, Result};
use candle_core::cuda_backend::cudarc::driver::{LaunchConfig, PushKernelArg};
use candle_core::cuda_backend::{CudaStorage, WrapErr};
use candle_core::{CpuStorage, CustomOp1, DType, Layout, Shape, Tensor};
use std::sync::OnceLock;

const SRC: &str = r#"
typedef unsigned short u16;

// out[((b*H+h)*S+s)*D + d] = in[((b*S+s)*H+h)*D + d]
extern "C" __global__ void vla_split_heads(
    const u16* __restrict__ in, u16* __restrict__ out,
    int b, int s, int h, int d)
{
    const long total = (long)b * s * h * d;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)blockDim.x * gridDim.x) {
        long dd = i % d;
        long t = i / d;
        long hh = t % h; t /= h;
        long ss = t % s; long bb = t / s;
        out[(((bb * h + hh) * s + ss) * d) + dd] = in[i];
    }
}

// out[((b*S+s)*H+h)*D + d] = in[((b*H+h)*S+s)*D + d]
extern "C" __global__ void vla_merge_heads(
    const u16* __restrict__ in, u16* __restrict__ out,
    int b, int s, int h, int d)
{
    const long total = (long)b * s * h * d;
    for (long i = (long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long)blockDim.x * gridDim.x) {
        long dd = i % d;
        long t = i / d;
        long ss = t % s; t /= s;
        long hh = t % h; long bb = t / h;
        out[(((bb * s + ss) * h + hh) * d) + dd] = in[i];
    }
}
"#;

static PTX: OnceLock<std::result::Result<String, String>> = OnceLock::new();

fn ptx(device: &candle_core::Device) -> Result<String> {
    match PTX.get_or_init(|| compile(device).map_err(|e| format!("{e:#}"))) {
        Ok(p) => Ok(p.clone()),
        Err(e) => bail!("NVRTC 编译失败：{e}"),
    }
}

fn compile(device: &candle_core::Device) -> Result<String> {
    use candle_core::cuda_backend::cudarc::driver::sys;
    use candle_core::cuda_backend::cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

    let candle_core::Device::Cuda(dev) = device else {
        bail!("只在 cuda 上编译");
    };
    let stream = dev.cuda_stream();
    let ctx = stream.context();
    let major = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(|e| anyhow!("查询 compute capability 失败：{e}"))?;
    let minor = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(|e| anyhow!("查询 compute capability 失败：{e}"))?;
    let arch: &'static str = Box::leak(format!("compute_{major}{minor}").into_boxed_str());
    let opts = CompileOptions {
        arch: Some(arch),
        use_fast_math: Some(false),
        ..Default::default()
    };
    Ok(compile_ptx_with_opts(SRC, opts)
        .map_err(|e| anyhow!("{e}"))?
        .to_src())
}

/// `(B,S,H,D)` → `(B,H,S,D)`。非 cuda / 非 bf16 时退回 candle 的通用实现。
pub fn split_heads(xs: &Tensor, heads: usize) -> candle_core::Result<Tensor> {
    #[cfg(feature = "cuda")]
    if xs.device().is_cuda() && xs.dtype() == DType::BF16 && xs.is_contiguous() {
        let (b, s, h, d) = xs.dims4()?;
        return xs.apply_op1_no_bwd(&HeadsOp { b, s, h, d, merge: false });
    }
    let (b, s, c) = xs.dims3()?;
    Ok(xs
        .reshape((b, s, heads, c / heads))?
        .transpose(1, 2)?
        .contiguous()?)
}

/// `(B,H,S,D)` → `(B,S,H*D)`。
pub fn merge_heads(xs: &Tensor) -> candle_core::Result<Tensor> {
    #[cfg(feature = "cuda")]
    if xs.device().is_cuda() && xs.dtype() == DType::BF16 && xs.is_contiguous() {
        let (b, h, s, d) = xs.dims4()?;
        let out = xs
            .apply_op1_no_bwd(&HeadsOp {
                b,
                s,
                h,
                d,
                merge: true,
            })
            ?;
        return Ok(out.reshape((b, s, h * d))?);
    }
    let (b, h, s, d) = xs.dims4()?;
    Ok(xs.transpose(1, 2)?.contiguous()?.reshape((b, s, h * d))?)
}

struct HeadsOp {
    b: usize,
    s: usize,
    h: usize,
    d: usize,
    merge: bool,
}

impl CustomOp1 for HeadsOp {
    fn name(&self) -> &'static str {
        "vla-heads"
    }

    fn cpu_fwd(
        &self,
        _s: &CpuStorage,
        _l: &Layout,
    ) -> candle_core::Result<(CpuStorage, Shape)> {
        candle_core::bail!("vla-heads 只有 cuda 实现")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        src: &CudaStorage,
        sl: &Layout,
    ) -> candle_core::Result<(CudaStorage, Shape)> {
        use candle_core::cuda_backend::cudarc::driver::DevicePtr;
        if !sl.is_contiguous() {
            candle_core::bail!("vla-heads 需要连续输入");
        }
        let dev = src.device.clone();
        let stream = dev.cuda_stream();
        let ptx = ptx(&candle_core::Device::Cuda(dev.clone()))
            .map_err(|e| candle_core::Error::Msg(format!("{e:#}")))?;
        let name = if self.merge {
            "vla_merge_heads"
        } else {
            "vla_split_heads"
        };
        let func = dev.get_or_load_custom_func(name, "vla_heads", &ptx)?;

        // 注意 start_offset：`is_contiguous()` 为真也可能带偏移（narrow 出来的视图）
        let xs = src.as_cuda_slice::<half::bf16>()?.slice(sl.start_offset()..);
        let (xp, gx) = xs.device_ptr(&stream);
        let n = self.b * self.s * self.h * self.d;
        let out = unsafe { dev.alloc::<half::bf16>(n) }?;
        let op = {
            let (p, g) = out.device_ptr(&stream);
            drop(g);
            p
        };

        let threads = 256u32;
        let blocks = ((n as u64 + threads as u64 - 1) / threads as u64).min(65535) as u32;
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&xp);
        builder.arg(&op);
        candle_core::builder_arg!(
            builder,
            self.b as i32,
            self.s as i32,
            self.h as i32,
            self.d as i32
        );
        unsafe { builder.launch(cfg) }.w()?;
        drop(gx);

        let shape = if self.merge {
            Shape::from_dims(&[self.b, self.s, self.h * self.d])
        } else {
            Shape::from_dims(&[self.b, self.h, self.s, self.d])
        };
        Ok((CudaStorage::wrap_cuda_slice(out, dev.clone()), shape))
    }
}

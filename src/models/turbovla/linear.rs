//! 带 bias epilogue 的 Linear。
//!
//! `candle_nn::Linear::forward` 是 `matmul` + `broadcast_add(bias)` **两个 kernel**；
//! nsys 显示这条路径每帧要花 **2.98ms**（`badd_bf16` 是整帧最大单项），因为每帧有
//! 200+ 个带 bias 的 Linear。
//!
//! torch 的 `addmm` 走 cuBLASLt 的 `CUBLASLT_EPILOGUE_BIAS`，一个 kernel 搞定。
//! 这里用 cudarc 的 cublasLt 封装做同样的事：
//!
//! - CUDA + bf16/f32 + 有 bias + 连续 → 走融合 GEMM；
//! - 其他情况（CPU、无 bias、非连续）→ 退回和 candle 完全一样的实现。
//!
//! 布局换算（cuBLASLt 是列主序）：行主序 `C(M,N) = x(M,K) · w(N,K)ᵀ`
//! 等价于列主序 `Cᵀ(N,M) = w(K,N) · x(K,M)`，于是
//! `m=N, n=M, k=K, transa=true, transb=false, lda=ldb=K, ldc=N`，
//! bias 长度 = C 的行数 = N。

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::cublaslt::{CudaBlasLT, Matmul, MatmulConfig};
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::CudaStream;
#[cfg(feature = "cuda")]
use candle_core::cuda_backend::{CudaStorage, WrapErr};
#[allow(unused_imports)]
use candle_core::{CpuStorage, CustomOp3, DType, Layout, Module, Result, Shape, Tensor};
#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

/// 进程级单例：`CudaBlasLT::new` 会分配 4MB workspace，不能每次调用都建。
#[cfg(feature = "cuda")]
static BLAS: OnceLock<CudaBlasLT> = OnceLock::new();

#[cfg(feature = "cuda")]
fn blas(stream: &Arc<CudaStream>) -> Result<&'static CudaBlasLT> {
    if let Some(b) = BLAS.get() {
        return Ok(b);
    }
    let created = CudaBlasLT::new(stream.clone())
        .map_err(|e| candle_core::Error::Msg(format!("cublasLt 初始化失败：{e}")))?;
    Ok(BLAS.get_or_init(|| created))
}

/// 和 `candle_nn::Linear` 同形状的层，`forward` 会把 bias 融进 GEMM。
pub struct Linear {
    weight: Tensor,
    bias: Option<Tensor>,
}

impl Linear {
    pub fn new(weight: Tensor, bias: Option<Tensor>) -> Self {
        Self { weight, bias }
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }
}

impl Module for Linear {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let dims = xs.dims().to_vec();
        let k = *dims.last().unwrap_or(&0);
        let m: usize = dims[..dims.len() - 1].iter().product();
        let n = self.weight.dim(0)?;
        let mut out_dims = dims[..dims.len() - 1].to_vec();
        out_dims.push(n);

        #[cfg(feature = "cuda")]
        if std::env::var("TURBOVLA_NO_FUSED_LINEAR").is_err()
            && xs.device().is_cuda()
            && xs.dtype() == DType::BF16
            && self.bias.is_some()
        {
            let x2 = xs.reshape((m, k))?;
            let x2 = if x2.is_contiguous() { x2 } else { x2.contiguous()? };
            let y = x2.apply_op3_no_bwd(
                &self.weight,
                self.bias.as_ref().unwrap(),
                &AddmmOp { m, n, k },
            )?;
            return y.reshape(out_dims);
        }

        // 回退路径：和 candle_nn::Linear 一致
        let y = xs.reshape((m, k))?.matmul(&self.weight.t()?)?;
        let y = match &self.bias {
            None => y,
            Some(b) => y.broadcast_add(b)?,
        };
        y.reshape(out_dims)
    }
}

struct AddmmOp {
    m: usize,
    n: usize,
    k: usize,
}

impl CustomOp3 for AddmmOp {
    fn name(&self) -> &'static str {
        "vla-addmm"
    }

    fn cpu_fwd(
        &self,
        _s1: &CpuStorage,
        _l1: &Layout,
        _s2: &CpuStorage,
        _l2: &Layout,
        _s3: &CpuStorage,
        _l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("vla-addmm 只有 cuda 实现")
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        x: &CudaStorage,
        xl: &Layout,
        w: &CudaStorage,
        wl: &Layout,
        b: &CudaStorage,
        _bl: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        use candle_core::backend::BackendStorage;
        if !xl.is_contiguous() || !wl.is_contiguous() {
            candle_core::bail!("vla-addmm 的 x/weight 必须连续");
        }
        let dev = x.device.clone();
        let stream = dev.cuda_stream();
        let blas = blas(&stream)?;

        // cuBLASLt 是列主序：Cᵀ(N,M) = w(K,N) · x(K,M)
        let cfg = MatmulConfig {
            transa: true,
            transb: false,
            transc: false,
            m: self.n as u64,
            n: self.m as u64,
            k: self.k as u64,
            alpha: 1.0,
            lda: self.k as i64,
            ldb: self.k as i64,
            beta: 0.0,
            ldc: self.n as i64,
            stride_a: None,
            stride_b: None,
            stride_c: None,
            stride_bias: None,
            batch_size: None,
        };

        macro_rules! run {
            ($t:ty) => {{
                let xs = x.as_cuda_slice::<$t>()?;
                let ws = w.as_cuda_slice::<$t>()?;
                let bs = b.as_cuda_slice::<$t>()?;
                let mut out = unsafe { dev.alloc::<$t>(self.m * self.n) }?;
                unsafe { blas.matmul::<_, _>(cfg, ws, xs, &mut out, Some(bs), None) }
                    .map_err(|e| candle_core::Error::Msg(format!("cublasLt matmul 失败：{e}")))?;
                Ok::<_, candle_core::Error>(CudaStorage::wrap_cuda_slice(out, dev.clone()))
            }};
        }

        let out = match x.dtype() {
            DType::BF16 => run!(half::bf16)?,
            DType::F32 => run!(f32)?,
            dt => candle_core::bail!("vla-addmm 不支持 {dt:?}"),
        };
        Ok((out, Shape::from_dims(&[self.m, self.n])))
    }
}

/// 和 `candle_nn::linear` 同签名。
pub fn linear(in_dim: usize, out_dim: usize, vb: candle_nn::VarBuilder) -> Result<Linear> {
    let weight = vb.get_with_hints(
        (out_dim, in_dim),
        "weight",
        candle_nn::init::DEFAULT_KAIMING_NORMAL,
    )?;
    let bound = 1. / (in_dim as f64).sqrt();
    let init_b = candle_nn::Init::Uniform {
        lo: -bound,
        up: bound,
    };
    let bias = vb.get_with_hints(out_dim, "bias", init_b)?;
    Ok(Linear::new(weight, Some(bias)))
}

/// 和 `candle_nn::linear_no_bias` 同签名。
pub fn linear_no_bias(in_dim: usize, out_dim: usize, vb: candle_nn::VarBuilder) -> Result<Linear> {
    let weight = vb.get_with_hints(
        (out_dim, in_dim),
        "weight",
        candle_nn::init::DEFAULT_KAIMING_NORMAL,
    )?;
    Ok(Linear::new(weight, None))
}

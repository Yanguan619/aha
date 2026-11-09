//! candle 没有、VLA 又必须要的几个算子：手写实现。
//!
//! 为什么不用 `candle_nn::ops::sdpa`：它只有 Metal 实现（`cpu_fwd` 直接
//! `bail!("SDPA has no cpu impl")`，也没有 `cuda_fwd`），CPU/CUDA 都跑不了；
//! `candle_nn::attention::flash_attn` 只有 CPU 版。所以这里用 candle 的基础算子
//! 拼出**和 Python `attention_backend="manual"` 逐位等价**的实现，同时消掉三处
//! 直译时会白付出的代价：
//!
//! 1. **不把 max 搬回主机**：`max_all().to_scalar::<f32>()` 每次都会同步设备
//!    （融合层每帧 6 次）。这里 max 一直留在设备上做 `broadcast_sub`，语义不变。
//! 2. **语言→视觉方向不转置拷贝**：直接对 `dim=-2` 做 softmax，等价于
//!    `transpose(2,3) → softmax(-1)`，但省掉一次 `(B,H,Sv,Sl)` 的 contiguous。
//! 3. **`qkᵀ` 的转置交给 cublas**：`k.transpose(2,3)` 后面不再接 `.contiguous()`，
//!    cublas 走 `OP_T`（`gemm_config` 认这种 stride），CPU 侧也能直接吃 stride。
//!
//! 注意 `q/k/v` 从 `(B,S,H,D)` 拆头的那次 `.contiguous()` **省不掉**：
//! cublas 的 `gemm_config` 只接受「最后两维是 dense 行」的布局，
//! `reshape(B,S,H,D).transpose(1,2)` 出来的 stride 两条分支都不满足，会
//! 报 `MatMulNonContiguous`。

use candle_core::{DType, Result, Tensor, D};
use candle_nn::ops::softmax;

/// `f32::MIN`：加性 mask 用的"负无穷"。在 bf16/f16 里会溢出成 -inf，
/// 和 Python 侧 `mask * torch.finfo(dtype).min` 的效果一致（softmax 后为 0）。
const NEG: f64 = f32::MIN as f64;

/// `(B,S,C)` → `(B,H,S,D)`。cuda+bf16 时走专用拷贝 kernel（candle 的通用 copy 只有 47GB/s）。
pub(crate) fn split_heads(xs: &Tensor, heads: usize) -> Result<Tensor> {
    let (b, s, c) = xs.dims3()?;
    let x4 = xs.reshape((b, s, heads, c / heads))?; // (B,S,H,D)
    #[cfg(feature = "cuda")]
    if x4.device().is_cuda() && x4.dtype() == DType::BF16 {
        return super::heads_cuda::split_heads(&x4, heads);
    }
    x4.transpose(1, 2)?.contiguous()
}

/// `(B,H,S,D)` → `(B,S,H*D)`。
pub(crate) fn merge_heads(xs: &Tensor) -> Result<Tensor> {
    let (b, h, s, d) = xs.dims4()?;
    #[cfg(feature = "cuda")]
    if xs.device().is_cuda() && xs.dtype() == DType::BF16 {
        return super::heads_cuda::merge_heads(xs);
    }
    xs.transpose(1, 2)?.contiguous()?.reshape((b, s, h * d))
}

/// padding mask `(…,S)`（1 = padding）→ 加性 mask（被遮的位置加 `f32::MIN`）。
///
/// 用 `affine` 而不是 `(1 - m) * MIN`：后者要额外开一个 `ones_like` 的缓冲。
///
/// **必须在 f32 里算完再转回目标 dtype**：`Tensor::affine` 会把常数先转成张量
/// 自己的 dtype，`f32::MIN` 落到 bf16/f16 上是 `-inf`，于是 `0.0 * -inf = NaN`
/// ——整个 mask 会变成 NaN。在 f32 里 `0 * MIN = 0`，转 bf16 后才是 `-inf`
/// （被遮的位置加 `-inf` 正是我们要的）。
pub(crate) fn additive_pad(mask: &Tensor, dtype: DType) -> Result<Tensor> {
    Ok(mask.to_dtype(DType::F32)?.affine(NEG, 0.0)?.to_dtype(dtype)?)
}

/// keep mask `(…,S,S)`（1 = 允许）→ 加性 mask（被遮的位置加 `f32::MIN`）。
///
/// `(1 - m) * MIN == -m * MIN + MIN`，所以一次 `affine` 就够。同样在 f32 里算。
pub(crate) fn additive_keep(mask: &Tensor, dtype: DType) -> Result<Tensor> {
    Ok(mask
        .to_dtype(DType::F32)?
        .affine(-NEG, NEG)?
        .to_dtype(dtype)?)
}

/// 多头自注意力（可带加性 mask）。`q/k/v` 是投影后的 `(B,S,C)`。
///
/// `additive` 形状 `(B,1,S,S)` 或 `(B,1,1,S)`，和 `attn` 广播相加。
pub(crate) fn masked_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    heads: usize,
    scale: f64,
    additive: Option<&Tensor>,
) -> Result<Tensor> {
    let q = split_heads(q, heads)?;
    let k = split_heads(k, heads)?;
    let v = split_heads(v, heads)?;

    // k 的转置不 contiguous：cublas 认 OP_T，CPU 也能按 stride 读。
    let mut attn = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
    if let Some(m) = additive {
        attn = attn.broadcast_add(m)?;
    }
    let attn = softmax(&attn, D::Minus1)?;
    merge_heads(&attn.matmul(&v)?)
}

/// 双向跨模态注意力：vision → language 和 language → vision 共用一次 `qkᵀ`。
///
/// - `q`/`values_v`：`(B,H,Sv,D)`，视觉侧的 query / value
/// - `k`/`values_l`：`(B,H,Sl,D)`，语言侧的 key / value
/// - `pad_l`：`(B,Sl)`，1 = 这个文本 token 是 padding（只遮 vision 看的那一侧）
///
/// 返回 `(out_v, out_l)`，分别是 `(B,Sv,H*D)` 和 `(B,Sl,H*D)`。
pub(crate) fn bi_attention(
    q: &Tensor,
    k: &Tensor,
    values_v: &Tensor,
    values_l: &Tensor,
    scale: f64,
    pad_l: Option<&Tensor>,
) -> Result<(Tensor, Tensor)> {
    let mut logits = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

    // 和 Python manual 后端同序：先减全局 max，再 clamp ±50000。
    // 两步在浮点上是恒等变换，保留是为了和参考实现逐位对齐；
    // 关键是 max 留在设备上，不触发 D2H 同步。
    let gmax = logits.max_all()?.detach();
    logits = logits.broadcast_sub(&gmax)?;
    logits = logits.clamp(-50000f32, 50000f32)?;

    // language → vision：(qkᵀ)ᵀ 的 softmax(-1) == 原式的 softmax(dim=-2)，
    // 少一次转置拷贝。这一侧**不**吃 padding mask（上游同样不遮）。
    let p_l = softmax(&logits, D::Minus2)?;

    // vision → language：只遮语言侧的 padding 列。
    let p_v = match pad_l {
        Some(mask) => {
            let add = additive_pad(mask, logits.dtype())?
                .unsqueeze(1)?
                .unsqueeze(1)?;
            softmax(&logits.broadcast_add(&add)?, D::Minus1)?
        }
        None => softmax(&logits, D::Minus1)?,
    };

    let out_v = merge_heads(&p_v.matmul(values_l)?)?;
    let out_l = merge_heads(&p_l.transpose(2, 3)?.matmul(values_v)?)?;
    Ok((out_v, out_l))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    /// 旧实现（逐算子直译 Python manual 后端），用来做数值对照。
    fn legacy_bi(
        q: &Tensor,
        k: &Tensor,
        vv: &Tensor,
        vl: &Tensor,
        scale: f64,
        pad: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let (b, heads, sv, hd) = q.dims4()?;
        let sl = k.dim(2)?;
        let mut attn = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        let max = attn.to_dtype(DType::F32)?.max_all()?.to_scalar::<f32>()?;
        let max_t = Tensor::new(max, attn.device())?.to_dtype(attn.dtype())?;
        attn = attn.broadcast_sub(&max_t)?;
        attn = attn.clamp(-50000f32, 50000f32)?;

        let mut attn_l = attn.transpose(2, 3)?.contiguous()?;
        let max_l = attn_l.max(D::Minus1)?.unsqueeze(D::Minus1)?;
        attn_l = attn_l.broadcast_sub(&max_l)?;
        attn_l = attn_l.clamp(-50000f32, 50000f32)?;

        if let Some(mask) = pad {
            // 注意：原实现把 mask 转成 F32 再加到 bf16 的 attn 上，bf16 下会 dtype 不匹配；
            // 这里是"修好后的等价参考"，用来做 bf16 对照。
            let add = mask
                .to_dtype(DType::F32)?
                .unsqueeze(1)?
                .unsqueeze(1)?
                .affine(NEG, 0.0)?
                .to_dtype(attn.dtype())?;
            attn = attn.broadcast_add(&add)?;
        }
        let pv = softmax(&attn, D::Minus1)?;
        let pl = softmax(&attn_l, D::Minus1)?;
        let out_v = pv.matmul(&vl)?.transpose(1, 2)?.reshape((b, sv, heads * hd))?;
        let out_l = pl.matmul(&vv)?.transpose(1, 2)?.reshape((b, sl, heads * hd))?;
        Ok((out_v, out_l))
    }

    fn rnd(b: usize, h: usize, s: usize, d: usize, seed: u64, dev: &Device, dt: DType) -> Tensor {
        // 简单 LCG，不引额外依赖
        let mut x = seed;
        let mut v = Vec::with_capacity(b * h * s * d);
        for _ in 0..b * h * s * d {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let f = ((x >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
            v.push(f);
        }
        Tensor::new(v.as_slice(), dev)
            .unwrap()
            .reshape((b, h, s, d))
            .unwrap()
            .to_dtype(dt)
            .unwrap()
    }

    fn maxdiff(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.to_dtype(DType::F32).unwrap();
        let d = (&a - &b.to_dtype(DType::F32).unwrap()).unwrap().abs().unwrap();
        d.max_all().unwrap().to_scalar::<f32>().unwrap()
    }

    #[test]
    fn bi_attention_matches_legacy() -> Result<()> {
        let dev = Device::Cpu;
        let (b, h, sv, sl, d) = (1, 4, 64, 21, 32);
        let q = rnd(b, h, sv, d, 1, &dev, DType::F32);
        let k = rnd(b, h, sl, d, 2, &dev, DType::F32);
        let vv = rnd(b, h, sv, d, 3, &dev, DType::F32);
        let vl = rnd(b, h, sl, d, 4, &dev, DType::F32);
        let pad = Tensor::new(
            &[
                0u8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, //
            ],
            &dev,
        )?
        .reshape((1, 21))
        .unwrap();

        let (a_v, a_l) = legacy_bi(&q, &k, &vv, &vl, 0.125, Some(&pad))?;
        let (b_v, b_l) = bi_attention(&q, &k, &vv, &vl, 0.125, Some(&pad))?;
        assert_eq!(a_v.dims(), b_v.dims());
        assert_eq!(a_l.dims(), b_l.dims());
        assert!(maxdiff(&a_v, &b_v) < 1e-6, "vision 侧偏差 {}", maxdiff(&a_v, &b_v));
        assert!(maxdiff(&a_l, &b_l) < 1e-6, "language 侧偏差 {}", maxdiff(&a_l, &b_l));
        Ok(())
    }

    #[test]
    fn bi_attention_masks_padding_columns() -> Result<()> {
        let dev = Device::Cpu;
        let (b, h, sv, sl, d) = (1, 2, 8, 6, 16);
        let q = rnd(b, h, sv, d, 7, &dev, DType::F32);
        let k = rnd(b, h, sl, d, 8, &dev, DType::F32);
        let vv = rnd(b, h, sv, d, 9, &dev, DType::F32);
        let vl = rnd(b, h, sl, d, 10, &dev, DType::F32);
        // 后 3 个是 padding
        let pad = Tensor::new(&[0u8, 0, 0, 1, 1, 1], &dev)?.reshape((1, 6))?;
        let (b_v, _) = bi_attention(&q, &k, &vv, &vl, 0.25, Some(&pad))?;

        // 把被遮住的 value 改成任意值，输出必须完全不变
        let mut vl2 = vl.flatten_all()?.to_vec1::<f32>()?;
        for i in 3 * d..6 * d {
            vl2[i] = vl2[i] + 10.0;
        }
        let vl2 = Tensor::new(vl2.as_slice(), &dev)?
            .reshape((b, h, sl, d))?
            .to_dtype(DType::F32)?;
        let (c_v, _) = bi_attention(&q, &k, &vv, &vl2, 0.25, Some(&pad))?;
        assert!(maxdiff(&b_v, &c_v) == 0.0, "padding 列没有被完全遮掉");
        Ok(())
    }

    #[test]
    fn bi_attention_matches_legacy_bf16() -> Result<()> {
        // 线上跑的是 bf16：这里确认降精度后两条实现仍然一致（bf16 只有 8 位尾数，
        // 容差按"相对误差"给，绝对值放宽到 2e-2）。
        let dev = Device::Cpu;
        let (b, h, sv, sl, d) = (1, 4, 64, 21, 32);
        let q = rnd(b, h, sv, d, 21, &dev, DType::BF16);
        let k = rnd(b, h, sl, d, 22, &dev, DType::BF16);
        let vv = rnd(b, h, sv, d, 23, &dev, DType::BF16);
        let vl = rnd(b, h, sl, d, 24, &dev, DType::BF16);
        let pad: Vec<u8> = (0..21).map(|i| u8::from(i >= 8)).collect();
        let pad = Tensor::from_vec(pad, (1, 21), &dev)?;

        let (a_v, a_l) = legacy_bi(&q, &k, &vv, &vl, 0.125, Some(&pad))?;
        let (b_v, b_l) = bi_attention(&q, &k, &vv, &vl, 0.125, Some(&pad))?;
        assert!(maxdiff(&a_v, &b_v) < 2e-2, "vision 侧偏差 {}", maxdiff(&a_v, &b_v));
        assert!(maxdiff(&a_l, &b_l) < 2e-2, "language 侧偏差 {}", maxdiff(&a_l, &b_l));
        assert!(b_v.dtype() == DType::BF16);
        Ok(())
    }

    #[test]
    fn masked_attention_matches_manual() -> Result<()> {
        let dev = Device::Cpu;
        let (b, s, heads, d) = (1, 12, 4, 64);
        let q = rnd(b, heads, s, d, 11, &dev, DType::F32);
        let k = rnd(b, heads, s, d, 12, &dev, DType::F32);
        let v = rnd(b, heads, s, d, 13, &dev, DType::F32);
        let q3 = merge_heads(&q)?;
        let k3 = merge_heads(&k)?;
        let v3 = merge_heads(&v)?;

        let scale = (d as f64).powf(-0.5);
        let got = masked_attention(&q3, &k3, &v3, heads, scale, None)?;

        let mut want = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale)?;
        want = softmax(&want, D::Minus1)?;
        let want = merge_heads(&want.matmul(&v)?)?;
        assert!(maxdiff(&got, &want) < 1e-6);
        Ok(())
    }

    #[test]
    fn additive_keep_is_zero_when_allowed() -> Result<()> {
        let dev = Device::Cpu;
        let m = Tensor::new(&[1u8, 0, 1, 0], &dev)?.reshape((1, 2, 2))?;
        for dt in [DType::F32, DType::BF16, DType::F16] {
            let a = additive_keep(&m, dt)?;
            let v = a.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            // 允许的位置必须是精确 0（bf16 下 `0 * -inf` 会变 NaN，这里就是回归点）
            assert_eq!(v[0], 0.0, "{dt:?}");
            assert_eq!(v[2], 0.0, "{dt:?}");
            assert!(v[1] < -1e37 && v[3] < -1e37, "{dt:?}");
            assert!(v.iter().all(|x| !x.is_nan()), "{dt:?} 出现 NaN: {v:?}");
        }
        Ok(())
    }

    #[test]
    fn additive_pad_has_no_nan_in_low_precision() -> Result<()> {
        let dev = Device::Cpu;
        let m = Tensor::new(&[0u8, 1, 0], &dev)?.reshape((1, 3))?;
        for dt in [DType::BF16, DType::F16] {
            let v = additive_pad(&m, dt)?.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            assert_eq!(v[0], 0.0, "{dt:?}");
            assert_eq!(v[2], 0.0, "{dt:?}");
            assert!(v[1] < -1e37, "{dt:?}");
        }
        Ok(())
    }
}

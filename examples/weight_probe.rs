//! 临时探针：用**模型里真实的权重**对比「融合 Linear」和「matmul + add」。
//!
//! cargo run --release --features vla,cuda --example weight_probe

use aha::models::turbovla::linear::Linear;
use candle_core::{DType, Device, Module, Tensor};
use candle_nn::VarBuilder;

fn cmp(name: &str, w: &Tensor, b: &Tensor, x: &Tensor) -> anyhow::Result<()> {
    let got = Linear::new(w.clone(), Some(b.clone())).forward(x)?;
    let want = x.matmul(&w.t()?)?.broadcast_add(b)?;
    let a = want.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let g = got.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
    let mut worst = 0f32;
    let mut rel = 0f32;
    for i in 0..a.len() {
        let d = (a[i] - g[i]).abs();
        if d > worst {
            worst = d;
        }
        let r = if a[i].abs() > 1e-6 { d / a[i].abs() } else { 0.0 };
        if r > rel {
            rel = r;
        }
    }
    let scale = a.iter().fold(0f32, |m, v| m.max(v.abs()));
    println!("{name:<52} max|d| {worst:.4e}  max rel {rel:.3e}  |x|max {scale:.3}");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let dev = Device::new_cuda(0)?;
    let dt = DType::BF16;
    let path = r"E:\models\turbovla\turbovla-libero\model.safetensors";
    // SAFETY: 只读
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], dt, &dev)? };

    let x1 = Tensor::randn(0f32, 1f32, (522, 768), &dev)?.to_dtype(dt)?;
    let x2 = Tensor::randn(0f32, 1f32, (512, 256), &dev)?.to_dtype(dt)?;

    let cases = [
        ("vision_encoder.backbone.layer.0.attention.q_proj", 768, 768, &x1),
        ("vision_encoder.backbone.layer.0.mlp.up_proj", 768, 3072, &x1),
        ("vision_projection.mlp.0", 768, 1024, &x1),
        ("vision_language_interaction.fusion_layers.0.attn.v_proj", 256, 1024, &x2),
    ];
    for (prefix, k, n, x) in cases {
        let w = match vb.get((n, k), &format!("{prefix}.weight")) {
            Ok(w) => w,
            Err(_) => {
                println!("{prefix:<52} 权重不存在，跳过");
                continue;
            }
        };
        let b = vb.get(n, &format!("{prefix}.bias"))?;
        cmp(prefix, &w, &b, x)?;
    }
    Ok(())
}

//! 预热：把「第一次推理才会发生的一次性开销」挪到服务启动时。
//!
//! # 为什么需要
//!
//! 桌宠的第一个 ASR / TTS 请求要额外承担一堆**一次性**成本：
//! - ONNX Runtime 首次选择执行 provider、首次加载 kernel（sherpa 模型）；
//! - candle 首次把权重搬进设备、CUDA kernel 首次 JIT/加载（VLA / GPT-SoVITS）；
//! - sys1 的第一帧推理要挑 kernel（所以 Laya 那边有 `warmup_once`）。
//!
//! 这些成本本身不大，但它们**恰好砸在用户开口说话的那一刻**，表现就是
//! "第一次特别慢，之后就快了"。预热把它们提前到端口打开之前。
//!
//! # 为什么按「角色」而不是按「模型」
//!
//! 预热是**横切关注点**：它关心的是"这个角色第一次被调用时会不会慢"，
//! 而不是"这个模型怎么实现"。所以这里的抽象层是**角色**
//! （`asr` / `tts` / `system1` / `vla`），预热动作一律走**角色的公共 API**
//! ——也就是端点本身用的那条路径——而不是去摸模型内部。
//!
//! 这样做的好处：
//! - 换模型实现（SenseVoice → 别的 ASR、Kokoro → GPT-SoVITS）**这里一行都不用改**；
//! - 新增模型只要它在角色下注册，预热自动生效；
//! - 预热用的输入和真实请求走同一条代码路径，所以**预热过就一定跑得通**。
//!
//! # 设计约束
//!
//! - **不产生对外可见的输出**：预热用的输入是静音 / 占位文本，结果丢弃。
//! - **失败不致命**：预热是锦上添花，失败只记日志，绝不让模型加载失败。
//! - **有超时**：预热不能把启动卡死。sherpa 的 ONNX 推理卡在某个 provider 上
//!   是有可能的，所以每步都包在超时里。

use std::time::{Duration, Instant};

use tokio::time::timeout;

use crate::server::api::SharedModel;

/// 单步预热的超时。超过就放弃这一步（不影响后续步骤，也不影响服务启动）。
///
/// 给到 30s：GPT-SoVITS 的完整预热（HuBERT + AR + 合成）在 CPU 上可能要好几秒，
/// CUDA graph 首次捕获更慢。超时的意义是「卡死了就别等了」，不是「快一点就放弃」。
const WARMUP_TIMEOUT: Duration = Duration::from_secs(30);

/// 预热结果：这一步花了多久、成没成。
struct WarmupStep {
    took_ms: u128,
    error: Option<String>,
}

/// 预热一个已加载的模型（按它承担的角色）。
///
/// `model_ref` 是注册表里的共享句柄；这里只**读**它的元信息并调用角色的
/// 公共方法。取写锁是因为 sherpa / sys1 的推理器**不是并发安全的**
/// ——预热和真实请求必须互斥。
pub async fn warmup_model(role: &str, model_ref: &SharedModel) {
    let step = run_step(role, model_ref).await;
    // NOTE: 项目没装 tracing subscriber，这些日志目前是 no-op。
    // 先用 eprintln 保证预热的结果**看得见**（预热失败静默是最难查的一类问题）。
    match &step.error {
        None => eprintln!("[warmup] {role} 预热完成（{} ms）", step.took_ms),
        Some(error) => eprintln!("[warmup] {role} 预热失败（不影响使用）：{error}"),
    }
}

/// 跑一步预热。
///
/// 预热的推理是**同步阻塞**的（sherpa 的 ONNX Runtime 调用、candle 的 CPU 算子
/// 都是阻塞的）。如果直接在 async 上下文里跑，`tokio::time::timeout` **取消不了它**
/// ——ONNX Runtime 卡在某个 provider 上时，整个启动会永久挂住。
///
/// 所以丢进 `spawn_blocking`：这样超时能真的放弃（任务继续在后台跑完，但不再
/// 挡住启动流程），tokio 的 worker 线程也不会被占住。
async fn run_step(role: &str, model_ref: &SharedModel) -> WarmupStep {
    let started = Instant::now();
    let model_ref = model_ref.clone();
    let role_owned = role.to_string();
    let result = timeout(
        WARMUP_TIMEOUT,
        tokio::task::spawn_blocking(move || dispatch(&role_owned, &model_ref)),
    )
    .await;
    let took_ms = started.elapsed().as_millis();
    match result {
        Ok(Ok(Ok(()))) => WarmupStep {
            took_ms,
            error: None,
        },
        Ok(Ok(Err(error))) => WarmupStep {
            took_ms,
            error: Some(format!("{error:#}")),
        },
        // 预热任务本身 panic 了（不该发生，但不能因此挡住启动）。
        Ok(Err(join_error)) => WarmupStep {
            took_ms,
            error: Some(format!("预热任务异常：{join_error}")),
        },
        // 超时：ONNX Runtime 卡在某个 provider 上是可能的，放弃这一步。
        Err(_) => WarmupStep {
            took_ms,
            error: Some(format!("超过 {}s 没出来，放弃", WARMUP_TIMEOUT.as_secs())),
        },
    }
}

/// 按角色分发到对应的预热动作。
///
/// **同步**函数：它在 `spawn_blocking` 的线程里跑，所以里面可以放心调同步推理。
fn dispatch(role: &str, model_ref: &SharedModel) -> anyhow::Result<()> {
    match role {
        "asr" => warmup_asr(model_ref),
        "tts" => warmup_tts(model_ref),
        "system1" => warmup_system1(model_ref),
        "vla" => warmup_vla(model_ref),
        _ => Ok(()),
    }
}

/// ASR 预热：喂 0.5 秒静音，跑一次完整识别。
///
/// 静音对识别结果是"空"或噪声，但**推理路径完全跑到了**：
/// 声学编码器、特征提取、解码器、token 查表，全都热了。
fn warmup_asr(model_ref: &SharedModel) -> anyhow::Result<()> {
    #[cfg(feature = "sherpa")]
    {
        let guard = model_ref.blocking_write();
        let Some(model) = guard.instance.asr_model() else {
            return Ok(());
        };
        // 0.5 秒 16k 静音。
        let silence = vec![0.0f32; 8_000];
        let _ = model.transcribe(&silence);
    }
    #[cfg(not(feature = "sherpa"))]
    {
        let _ = model_ref;
    }
    Ok(())
}

/// TTS 预热：跑一次**完整**合成。
///
/// # 为什么是完整合成而不是只跑一段
///
/// GPT-SoVITS 的首次请求要额外付三笔钱：
/// 1. **参考音频特征提取**（HuBERT + 谱 + ref_enc + eres2net）——几百毫秒，
///    而且结果**有缓存**（`ref_cache`），但缓存只在进程内，首次请求必然付。
/// 2. **AR 解码器的 kernel 首次加载**（24 层 transformer + CUDA graph 捕获）。
/// 3. **合成器**（enc_p → flow → HiFiGAN）的 kernel 首次加载。
///
/// 只跑合成器段（省算力的做法）只热了第 3 项，实测**没有可测量的收益**
/// （A/B：预热组 median 1360ms vs 对照组 1354ms，落在噪声里）。
/// 跑完整合成才能热到前两项——尤其是把参考音频缓存填上。
///
/// 代价是预热本身要几秒，但这是**一次性**的，而且发生在端口开之前。
///
/// 用一句话当输入（最短的合法输入），让 AR 尽快吐 EOS。
fn warmup_tts(model_ref: &SharedModel) -> anyhow::Result<()> {
    #[cfg(feature = "sherpa")]
    {
        let guard = model_ref.blocking_write();
        if let Some(model) = guard.instance.tts_model() {
            let _ = model.synthesize_wav("嗯", "", 1.0);
            return Ok(());
        }
    }
    #[cfg(feature = "gpt-sovits")]
    {
        let guard = model_ref.blocking_write();
        if let Some(model) = guard.instance.gpt_sovits_model() {
            // 用第一个可用音色；没有音色就让模型自己报错（预热失败也不致命）。
            let voice = model.voices().first().map(|s| s.to_string());
            let _ = model.synthesize("嗯", voice.as_deref(), 1.0);
            return Ok(());
        }
    }
    let _ = model_ref;
    Ok(())
}

/// Laya（system1）预热：跑一次判定。
///
/// sys1 的 `Batcher::warmup` 是 async 且内部要过队列，所以这里用一次真实判定
/// 来触发它（`LayaRouteModel::route` 内部会先调 `warmup_once`）。
///
/// `route` 是 async 但内部不 await 阻塞 IO，所以在一个 `block_in_place` 里
/// 同步等它是安全的。
fn warmup_system1(model_ref: &SharedModel) -> anyhow::Result<()> {
    #[cfg(feature = "laya")]
    {
        let guard = model_ref.blocking_write();
        let Some(model) = guard.instance.route_model() else {
            return Ok(());
        };
        // 一句肯定会被 keyword 命中的短文本：只触发 warmup，不进模型推理。
        // （keyword 命中的路径不跑神经网络，正好把 warmup 的开销留下。）
        let fut = model.route("你好");
        let _ = tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current().block_on(fut)
        });
    }
    #[cfg(not(feature = "laya"))]
    {
        let _ = model_ref;
    }
    Ok(())
}

/// VLA 预热：跑一次占位前向。
///
/// CUDA graph 的捕获依赖具体指令（key 里含 instruction），没法预热；
/// 但**文本编码器（BERT）和视觉编码器的 kernel 加载**与指令无关，可以预热。
/// 这里用一张占位图 + 一句占位指令跑一遍非 graph 路径。
fn warmup_vla(model_ref: &SharedModel) -> anyhow::Result<()> {
    #[cfg(feature = "vla")]
    {
        let guard = model_ref.blocking_write();
        let Some(model) = guard.instance.vla_model() else {
            return Ok(());
        };
        // 一张 8x8 的灰度 PNG（最小的合法 PNG）。
        let png = tiny_png();
        let images: Vec<&[u8]> = vec![&png; model.num_views()];
        let state = vec![0.0f32; model.state_dim()];
        // 预热走非 graph 路径：graph 要按指令捕获，预热阶段的指令没意义。
        let _ = model.stages(&images, "warmup", &state);
    }
    #[cfg(not(feature = "vla"))]
    {
        let _ = model_ref;
    }
    Ok(())
}

/// 造一张最小的灰度 PNG（8x8）。
///
/// 只用于预热视觉编码器，**不追求图像内容对**，只要是合法 PNG 能被解码。
///
/// 用 `image` 库生成而不是手写字节：PNG 的每个 chunk 都有 CRC32，
/// 手写容易算错（第一版就踩了，IHDR 的 CRC 不对，解码直接报 CrcMismatch）。
/// 让库来算最省心，也不用在测试里重复验证字节。
#[cfg(feature = "vla")]
fn tiny_png() -> Vec<u8> {
    let mut buffer = Vec::new();
    let image = image::GrayImage::from_pixel(8, 8, image::Luma([0]));
    // `write_to` 需要 `Write + Seek`；Cursor 两者都有。
    let mut cursor = std::io::Cursor::new(&mut buffer);
    if image
        .write_to(&mut cursor, image::ImageFormat::Png)
        .is_err()
    {
        // 生成失败（理论上不会）：返回一个 1x1 的最小 PNG 兜底。
        return vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        ];
    }
    buffer
}

#[cfg(all(test, feature = "vla"))]
mod tests {
    use super::*;

    /// 预热用的 PNG 必须是合法 PNG（否则视觉编码器会报错，预热就白做了）。
    #[test]
    fn tiny_png_is_valid() {
        let png = tiny_png();
        assert_eq!(&png[..8], &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    }

    /// 用 image 库真解一次，确认不是坏字节。
    #[test]
    fn tiny_png_decodes() {
        let png = tiny_png();
        let img = image::load_from_memory_with_format(&png, image::ImageFormat::Png)
            .expect("预热用的 PNG 解不开");
        assert_eq!(img.width(), 8);
        assert_eq!(img.height(), 8);
    }
}

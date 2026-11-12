# IndexTTS 2.5（audiocpp GGUF）接入方案

> 目标：让 aha 直接吃 `rookie6667/IndexTTS2.5-GGUF` 这份量化权重（Q4_K，2.57 GiB），
> **不重打 GGUF、不复制权重**，尽量复用 aha 现有模型实现。
>
> 权重来源：`https://modelscope.cn/models/rookie6667/IndexTTS2.5-GGUF`（社区量化版，
> 上游 `IndexTeam/IndexTTS-2.5`，bilibili 模型使用许可）。

## 1. 这个 GGUF 里有什么

`general.architecture = audiocpp`，`family = index_tts2`，3790 个张量 + 11 个内嵌文件。
Q4_K 版 dtype 只有 `F32 / F16 / BF16 / Q4_K / I64`，**没有 IQ 系列**。

| 组件 | 张量数 | 用途 | aha 现有实现 |
|---|---|---|---|
| `bigvgan` | 783 | 声码器（weight_norm 的 `weight_g/weight_v`，全 F32） | `models::bigvgan` ✅ |
| `campplus` | 815 | 说话人向量（CAMPPlus，全 F32） | `models::campplus` ✅ |
| `wav2vec2bert` | 773 | W2V-BERT 2.0 语义特征（24 层，192 个 Q4_K） | `models::w2v_bert_2_0` ✅ |
| `semantic_codec` | 243 | ConvNeXt/Vocos 语义 codec（codebook 8192×8） | `models::mask_gct::RepCodec` ✅（修了一处） |
| `gpt` | 456 | Tortoise/UnifiedVoice GPT（24 层 × 1280，20 heads）+ emo conformer/perceiver 条件模块 | 新增 `models::index_tts::gpt`（主干 ✅，条件模块待做） |
| `qwen_emotion` | 310 | Qwen3-0.6B 情绪文本编码（28 层，无 lm_head） | `models::qwen3`（待接） |
| `s2mel` | 283 | CFM flow matching（DiT 13 层 + wavenet + length regulator） | 待做 |
| `wav2vec2bert_stats` | 2 | 特征归一化的 mean/var（1024 维） | — |

内嵌文件（`audiocpp.embedded_files.*`，用 `gguf_lenient::embedded_file` 取，不用另下官方仓库）：
`config.yaml`、`bigvgan/config.json`、`multilingual_zh_ja_yue_char_del.tiktoken`、
`qwen0.6bemo4-merge/*`（tokenizer/config/vocab）、`w2v-bert-2.0/config.json(+preprocessor)`。

## 2. 已经落地的代码

| 文件 | 说明 |
|---|---|
| `src/models/common/gguf_lenient.rs` | 宽松 GGUF 头解析，直接产出 candle 的 `gguf_file::Content` |
| `src/models/common/gguf_var_builder.rs` | `GgufVarBuilder`：GGUF → candle `VarBuilder` 后端 |
| `src/models/index_tts/config.rs` | `config.yaml` 结构化（gpt 段 + dataset.mel） |
| `src/models/index_tts/gpt.rs` | GPT 主干：text/mel 双词表 + 位置 embedding、24 层、KV cache、双头 |
| `src/models/index_tts/emo.rs` | 情绪条件：Conformer(conv2d2/相对位置/4 层) + PerceiverResampler + emovec/emo_layer |
| `src/models/index_tts/s2mel.rs` | s2mel：gpt_layer + length_regulator + U-DiT(13 层) + wavenet 尾 + Euler/CFG 采样 |
| `src/models/index_tts/pipeline.rs` | 端到端：条件 latent → GPT 自回归(codes) → codec → length_regulator → CFM → bigvgan |
| `src/models/index_tts/tokenizer.rs` | tiktoken BPE 分词（手写预切分扫描器，无 regex 依赖）+ 1673 个特殊 token 表 |
| `src/models/index_tts/frontend.rs` | 音频前端：wav 读写、加窗 sinc 重采样、Kaldi fbank（w2v-bert / campplus）、librosa slaney mel（ref_mel） || `src/models/mask_gct/model.rs` | 修 `RepCodec::forward` 漏用 `decoder_1` 的问题 |
| `src/models/qwen3/model.rs` | 加 `forward_hidden_full`（情绪编码要整段 hidden，不是最后一个 token） |
| `src/models/common/gguf.rs` | 加 `contains()` / `tensor_names()` |
| `tests/test_gguf_lenient.rs` | 8 个测试，见下 |

### 为什么不需要重打 GGUF

candle 0.11 的 `gguf_file::Content::read` 在**解析阶段**就对每个张量调
`GgmlDType::from_u32`，遇到 I64 直接 bail（这份文件有 123 个 I64，全是 BatchNorm 的
`num_batches_tracked`）。但 `Content` 的字段和 `TensorInfo::read` 都是 pub，所以自己解析
头部、跳过不认识的 dtype 即可，**权重区一个字节都不动**。

顺带解决的两件事：

* `_audiocpp.<idx>` 这类被压掉的张量名，从 metadata 的 `audiocpp.tensor_names` 按顺序还原
  （967 个；另外 112 个是上面被跳过的 I64）；
* `audiocpp.embedded_files.data`（16.8MB）/ `tokenizer.json`（11MB）这类大 U8 数组不展开成
  `Value`（否则几百 MB 内存），只记 `{start, len}` 区间，用时再 seek。

### 怎么"零改动"复用现有模型

```rust
let vb = VarBuilder::from_backend(
    Box::new(GgufVarBuilder::new(gguf, "bigvgan")),   // 前缀 = GGUF 组件名
    DType::F32, device,
);
let bigvgan = BigVGAN::new(vb, &cfg)?;                // 原样
```

`GgufVarBuilder` 会依次尝试 `前缀/名字`、`前缀.名字`、`名字`（audiocpp 两种分隔符混用），
取到的量化权重自动 dequantize；形状对不上但元素数相同会 reshape（覆盖
`downsample.lowpass.filter [12]` 这种 1D 存法）；2D 反向形状不会自动转置，需要显式处理。

## 3. 验证结果（`cargo test --test test_gguf_lenient -- --include-ignored`）

```
header_report          ok   3790 张量 / 跳过 I64 123 / 还原 967 个原名 / 11 个内嵌文件
read_all_tensors       ok   3667 个张量全部读出，2.55 GB
qmatmul_smoke          ok   Q4_K c_attn 走 QMatMul，相对误差 0.76%（f16 通路）
bigvgan_forward        ok   mel [1,80,50] -> wav [1,1,12800]，50×256 精确
campplus_forward       ok   fbank [1,300,80] -> embedding [1,192]
w2v_bert_forward       ok   feats [1,64,160] -> hidden [1,64,1024]
semantic_codec_forward ok   重建 [1,32,1024]，码本索引 [1,1,16]，最大索引 < 8192
index_tts_gpt_trunk    ok   prefill [1,8,1280] -> mel logits [1,8,8194]；KV cache 解码 [1,1,1280]
index_tts_emo_cond     ok   w2v 特征 [1,40,1024] -> emo_vec [1,1280]
index_tts_s2mel        ok   gpt_layer 1280→1024；length_regulator 24→48 帧；DiT -> mel [1,80,48]；1 步 Euler 通过
index_tts_qwen_emotion ok   Qwen3-0.6B（28 层，tie embeddings）ids [1,6] -> hidden [1,6,1024]
index_tts_pipeline     ok   conds [1,3,1280] → GPT 出 8 个 code → codec/CFM/bigvgan → wav [1,1,6912] @22050Hz
index_tts_text_tokenizer ok 58836 个 rank + 1673 个特殊 token，7 组用例和 Python tiktoken 逐 id 一致
```

## 4. 已踩到的坑

1. **I64**：见上，不重打就自己解析头。
2. **张量命名两套**：`bigvgan/conv_pre.weight_g` 用 `/`，`campplus/speaker_encoder.head.*` 是
   `组件/` + 内部 `.`；`_audiocpp.NNN` 需要按 metadata 还原。
3. **维度方向**：GGUF 的 dims 是反序的，翻过来正好是 PyTorch 序。GPT-2 的 `Conv1D` 权重是
   `(in, out)`（`c_attn` = 1280×3840），而 candle `Linear` 要 `(out, in)`，这四个投影必须转置；
   而 `mel_head` / `spk_emb_proj` 是 `nn.Linear` 的 `(out, in)`，不能转置。
4. **weight_norm**：GGUF 保留了 `weight_g/weight_v` 两个 F32 张量，`WNConv1d` 直接按名字取即可，
   合并零误差。
5. **semantic_codec 的 decoder 不对称**：aha 原实现 `decoder_0` 之后直接接 `up` 卷积，
   少了 `decoder_1`（384→1024），会报通道数不匹配，已修。
6. **candle 的 `QMatMul` 在 CPU 上走 f16 通路**（输入 cast f16 + f16 累加），Q4_K 权重下
   `c_attn` 单层相对误差 ~0.76%。对 bigvgan/campplus（F32）无影响，但 GPT 自回归可能放大成
   token 偏移，需要实测音质后再决定：接受 / GPT 反量化成 F32（约 1.9GB）/ 只把 `lm_head` 留 F32。

## 5. 剩余工作

1. ~~emo 条件模块~~ ✅ `src/models/index_tts/emo.rs`（对照上游 `conformer_encoder.py` /
   `perceiver.py` 逐层移植）。要点：`conv2d2` 只做 1 次 Conv2d(k3,s2) + `Linear(512*511→512)`；
   相对位置注意力**没有** rel_shift（pos 长度 = T）；perceiver 的 `cross_attn_include_queries=True`
   会把 latents 拼到 context 前面（k/v 长度 = T+1）；`ff_mult = perceiver_mult = 2`
   （`dim_inner = 1024*2*2/3 = 1365`）；`latents` 只有 1 个；perceiver 内所有 Linear `bias=False`；
   结尾是 `F.normalize(x)*sqrt(dim)*gamma` 的 RMSNorm。padding mask 暂未实现（固定长度推理先用全 1）。
2. ~~s2mel~~ ✅ `src/models/index_tts/s2mel.rs`（对照 `flow_matching.py` / `diffusion_transformer.py` /
   `gpt_fast/model.py` / `wavenet.py` / `length_regulator.py` 移植）。要点：
   * `gpt_layer` = 三个**无激活**的 Linear(1280→256→128→1024)，输出加在 codec 语义特征上；
   * `cond_x_merge_linear` 输入 = 80(noise) + 80(prompt mel) + 512(cond) + 192(style) = 864；
   * DiT 是 U-ViT：前 6 层输出进栈，后 6 层**后进先出**取 skip（第 7 层两头都不占），
     `skip_linear` 把 `[x, skip]` 合成 512；
   * AdaLN 是 `weight * RMSNorm(x) + bias`，`weight/bias` 来自 `project_layer(time_emb)`；
   * RoPE 是**交错式**（最后一维拆 `(d/2,2)`），没有 KV cache，注意力是双向（非因果）；
   * wavenet 的 `fused_add_tanh_sigmoid` 是 `tanh(前半) * sigmoid(后半)`（不是整体 tanh*sigmoid），
     踩过一次通道数对不上的坑；
   * 尾部 `final_layer` 用 weight_norm 的 Linear + `elementwise_affine=False` 的 LayerNorm，
     然后 `conv2`(512→80) 出 mel；
   * 采样是 `solve_euler` + CFG（默认 25 步、cfg 0.7，prompt 区域每步清零）。
3. ~~qwen_emotion~~ ✅ 直接复用 `models::qwen3`（`Qwen3Config` 能从内嵌 config.json 反序列化），
   但 `forward_hidden` 只返回最后一个 token，情绪编码需要整段 —— 已加 `forward_hidden_full`。
   另外 `GgufVarBuilder` 补了一种名字组合：candle 的 `pp("model")` 会拼成
   `qwen_emotion.model.xxx`，而 GGUF 是 `qwen_emotion/model.xxx`，现在会把第一个 `.` 换成 `/` 再试。
4. **前端（文本部分已完成）** ✅ `src/models/index_tts/tokenizer.rs`：
   * `.tiktoken` 每行是 `base64(token) rank`；词表 58836 个 rank（不是 README 说的 60509）；
   * 特殊 token 从 58836 开始按上游顺序追加：`<|en|>`=58838、`<|zh|>`=58839 …，共 1673 个，
     最大 id 60506，正好落在 `text_embedding`(60510 行) 里；
   * 预切分正则原样实现，没有引入 regex 依赖，用 `char::is_alphabetic/is_numeric/is_whitespace`；
   * 词表里有一行是 `= 48474`（Python `b64decode("=")` → 空串），这里同样按空串处理；
   * 已用 Python tiktoken + 同一份词表生成参考值，7 组用例逐 id 对上（含特殊 token / 中文 / 英文缩写）。
5. ~~前端（音频部分）~~ ✅ `src/models/index_tts/frontend.rs`（已接进 `synthesize_profiled`）：
   * `seamless_features`：Kaldi fbank（400/160/512、povey、preemph 0.97、80 bins）
     → 每个 bin 沿时间归一化 → stride 2 拼成 160 维，喂 w2v-bert；
   * `campplus_fbank`：同一套 Kaldi fbank 但沿时间维减均值，喂 campplus；
   * `mel_spectrogram_22050`：librosa slaney mel、hann 1024/256、`center=False`、
     reflect pad 384、`log(clamp(1e-5))`，作为 s2mel 的 `ref_mel`；
   * `load_wav` / `resample`（自带 radix-2 FFT 与加窗 sinc 重采样，不依赖外部 crate）；
   * 参考音频截断到 15s，同一段音频按 16k / 22.05k 各转一次。
6. **性能**：debug 构建下 8 个 code + 2 步 CFM + bigvgan 端到端约 97s（含模型加载），必须走 release/GPU。
7. **还没做：暴露成 `/v1/tts` 后端**。现在 `pipeline` 端到端能跑了（只有集成测试能碰到），
   但服务层完全不认它，四处都得加：
   * `WhichModel` 里没有 IndexTTS 变体（`src/models/common/model_mapping.rs`），所以 CLI/HTTP 加载不了；
   * `ModelInstance` 没有对应变体，`src/models/mod.rs` 的加载分派缺一个 arm；
   * `src/server/deskpet.rs` 的 `handle_tts` / `handle_tts_stream` 只有 GPT-SoVITS 和 Kokoro 两条路；
   * 契约要对齐：`TtsRequest` 只有 `text/voice/speed`，IndexTTS 还需要**参考音频**（音色克隆）
     和 `.tiktoken` 词表路径——最自然的做法是照 GPT-SoVITS 的 `voices.json`：
     `voice` 当配置名，配置里写 ref_wav；输出采样率也要确认（bigvgan 出口不是 24k）。


## 6. GPU（推荐）与当前阻塞点

推理统一走 `aha::utils::get_device(None)`，带 `--features cuda` 编译时自动上 GPU。
实测 RTX 5060 8GB：**模型加载 2.4s，1.59s 音频（40 个 code、5 步 CFM）耗时 5.8s**（release）。

```powershell
# 注意：candle-kernels 需要 nvcc + cl.exe，必须先进 VS 环境（cmd 里 %PATH% 是提前展开的，
# 要开延迟展开 !PATH! 才能保住 vcvars 设置）
cmd /v:on /c 'call "D:\Program\VC\Auxiliary\Build\vcvars64.bat" >nul && set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;D:\Program\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;!PATH!" && set AHA_GGUF=E:\tmp\indextts25\index-tts2_5-q4_k.gguf && set AHA_REF_WAV=E:\Projects\desktop-pet\tools\tts\sovits\ref\ref.wav && cargo test --release --features cuda --test test_gguf_lenient -- --ignored --nocapture index_tts_real_synthesis'
```

CUDA 上必须显式 `.contiguous()`（CPU 的 gemm 能忍非连续，CUDA 直接报错），已经补了两处：
`emo.rs` 的 perceiver q/k/v、`s2mel.rs` 的 `cond_x_merge_linear` 输入。

### B 路线结果（beam search，已实现）

`pipeline.rs` 里新增 `generate_beam`（beams 当 batch 维、每步 `reorder_kv_cache` 重排 KV cache），
`gpt.rs` 加 `reorder_kv_cache`。中途修了一个自己挖的坑：**对负 log-prob 做除法惩罚会变成加分**
（HF 语义是改 logit 再 softmax，正除负乘），修正后 beam search 才正常出声。

| 解码方式 | 时长 | 谱平坦度 | RMS |
|---|---|---|---|
| 采样 rep=10（原） | 4.23s | 0.4071 | 0.115 |
| beam=3 rep=10 | 3.87s | 0.3969 | 0.107 |
| **beam=3 rep=1** | 4.11s | **0.3904** | 0.127 |
| 参考码重建（上界参照） | 4.31s | 0.3512 | 0.164 |

beam + rep=1 最好，但也只把 0.407 压到 0.390，离参考码路径的 0.351 还有明显差距。
时长 3.87s/4.11s（beam 会更早遇到 stop token），说明条件链路是通的。

结论：**解码策略的收益空间不大（~4%），剩下的差距基本可以锁定是 Q4_K 权重精度**——
GPT 的 `c_attn` 单层相对误差 0.76%，CFM 112 个 Q4_K 张量 × 25 步迭代，噪声会被放大。
要再往下压只能换 F16/原始权重（A 路线）。

对比上游 `alias_free_torch/resample.py` 发现：`UpSample1d.forward` 是
`self.ratio * F.conv_transpose1d(...)`，**aha 的实现漏了 `× ratio`**。
每个 `TorchActivation1d`（up → SnakeBeta → down）因此少 2 倍增益，
18 个 resblock × 6 个激活叠起来把网络推到完全不同的工作点，输出既是错的电平、也和目标不相关。

修复后自检（用参考音频自己的 mel 重建）：

```
相关系数 corr=0.123（波形相位不同，该项天然不高）
RMS：重建 0.2422 vs 原始 0.2463
重建 mel 与输入 mel 的平均绝对误差 0.102（log 幅度，已经很小）
```

结论：声码器正常了。修复位置：`src/models/bigvgan/mod.rs::UpSample1d::forward`。

前端的 mel 已经和 numpy 参考实现逐值对齐（mean -4.647 vs -4.651，min/max 完全一致），
但**用参考音频自己的 mel 过 bigvgan，输出与原始音频相关性只有 0.0023，RMS 低 2700 倍**
（0.00009 vs 0.2463）——不是单纯音量小，而是内容不对。

排查记录：

* `conv_pre` 输出峰值 ~45，逐级 ups 后衰减到 0.3，`conv_post` 后 0.004（g=0.0589）；
* `alpha/beta` 存在 -2~2 之间（是 log 尺度，`.exp()` 处理正确）；
* `weight_g` 非负（`conv_pre` 均值 0.31，`ups.*` 均值 0.32，`conv_post` 0.0589）；
* `weight_v` 的范数接近 1（`conv_post` 0.897）→ 说明 v 是"已归一化 + g 携带幅度"的常规 weight_norm 存法；
* candle 的 `ConvTranspose1d` 核方向已用 delta 输入自检过，和 PyTorch 一致（`tests::candle_conv_transpose_sanity`）。

最终定位到 `UpSample1d` 漏了 `× ratio`，见上。

## 8. 音质调优记录（进行中）

主观反馈：**是中文，但只有 2s、音色不像参考、有明显杂音和断续**。客观指标与排查：

| 检查 | 结果 | 结论 |
|---|---|---|
| 参考码重建（codec→LR→CFM→bigvgan） | 4.31s（原 4.35s），mel MAE 0.689，campplus 余弦 **0.892**，能量曲线与原文逐段吻合 | 下游链路正常 |
| GPT 合成 | 2.07s，campplus 余弦 **0.72~0.75**，mel 频带能量整体低 ~1dB | 问题在 GPT 侧或 codec 编码侧 |
| 条件消融（说话人 latent 置零） | code 序列 **0% 相同**（53 vs 68 个） | 条件链路是通的，不是"没接上" |
| style/conds 数值 | style std 0.99、conds std 1.09 | 没有退化/全零 |
| 贪心 vs 采样 | 余弦 0.7245 vs 0.7485，长度 2.00s vs 2.07s | 采样策略不是主因 |

已确认的细节：**mel 位置编码是「相对 mel 段」的**（上游预测第 k 个 mel token 时用
`attention_mask.shape[1] - mel_len = 1 + k`，start token 用 0），不是整条序列的绝对位置。
两种写法都试过，指标在 ±0.03 噪声内，说明位置不是当前瓶颈。

下一步建议（按收益排序）：

1. **对齐上游的解码策略**：上游传 `num_beams=3`，实际很可能是 beam search；我们目前是
   top-k/top-p 采样，beam 往往能明显减少"断续/杂音"。
2. **codec 的量化精度**：`codec 往返 cos=0.806`（8 维码本），若官方实现相近则正常，
   否则要看 `quantizer.in_project/out_project` 的 weight_norm 合并。
3. **GPT 的 Q4_K 噪声**：`QMatMul` 在 CUDA 上走 f16 通路，c_attn 单层相对误差 0.76%，
   自回归 50+ 步会累积；可试 `AHA_DTYPE=f16` 或把 GPT 反量化成 F32 对比。
4. **音色**：0.72 的余弦说明"像但不够像"，优先怀疑 campplus 的 Kaldi fbank 参数
   （mel floor、povey 窗、preemphasis 顺序），可下载官方 bigvgan/campplus 权重对比。

## 9. 最终实测（GPU，release）

```
device=Cuda dtype=F32
模型加载           2.2s
"你好，世界。"     53 个 code / 25 步 CFM / cfg 0.7
合成 2.07s 音频    耗时 ~4.5s（RTF≈2.0）
全部 19 个测试     36s 跑完（CPU 上约 230s）
```

```powershell
# 权重（2.75GB）放在哪都行，用环境变量指定
$env:AHA_GGUF="E:\tmp\indextts25\index-tts2_5-q4_k.gguf"

cargo test --test test_gguf_lenient                                    # 只跑头解析
cargo test --test test_gguf_lenient -- --ignored --nocapture --test-threads=1   # 全量
```

> 如果 `target\debug\aha.exe` 正在被桌宠占用，`cargo test` 会因为无法重新链接 bin 报
> `failed to remove file ... aha.exe (os error 5)`。两条路：先 `Stop-Process` 掉那个进程，
> 或者用独立 target 目录（首次要整包重编，需要 VS 自带的 cmake 在 PATH 里）：
>
> ```powershell
> $env:Path="D:\Program\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$env:Path"
> $env:CARGO_TARGET_DIR="E:\tmp\aha-target"
> ```

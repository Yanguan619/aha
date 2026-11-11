# GPT-SoVITS（v2Pro 中文）接入方案

> 本文是**工作记录 + 计划**，不是 aha 上游文档。目标：让 aha 用原生 candle 跑
> GPT-SoVITS 的**零样本克隆 / 微调音色**，作为 `tts` 角色模型接进 `/v1/tts` 与
> `/v1/tts/stream`（桌宠嘴型驱动）。
>
> 范围：先 **v2Pro（中文）**；v4（CFM/DiT + 48k 声码器）留到二阶段。

---

## 0. 一句话结论

v2Pro 的推理链路里**没有 duration predictor、没有 DiT**，本质是
「HuBERT → 量化 → 文本+SSL 交叉注意力 → flow 逆变换 → HiFiGAN」，
再接一个 24 层 GPT 自回归出语义 token。全部算子 candle 都有，缺的只是**搬运**。

---

## 1. 版本与权重

| 版本 | 合成器 | 声码器 | 额外模型 | 备注 |
|---|---|---|---|---|
| v2 | `SynthesizerTrn`（VITS，32k） | 内置 `dec` | — | 基准复杂度和音质 |
| **v2Pro** | 同上 + `sv_emb` | 内置 `dec` | **eres2netv2** | 少漏音色、更稳，**本次目标** |
| v4 | `SynthesizerTrnV3` | 外置 HiFiGAN 48k | — | 多一个 CFM/DiT(dim1024×22) |

v2Pro 需要的一整套权重：

| 文件 | 角色 | 大小量级 |
|---|---|---|
| `s1v3.ckpt` | AR（Text2Semantic，24 层 GPT） | 148 MiB |
| `s2Gv2ProPlus.pth` | 合成器（v2 架构 + `sv_emb`） | ~100 MiB |
| `chinese-hubert-base/pytorch_model.bin` | 内容特征（`last_hidden_state`，768 维） | 361 MiB |
| `chinese-roberta-wwm-ext-large/pytorch_model.bin` | 文本特征（取 `hidden_states[-3]`，1024 维） | 1.2 GiB |
| `sv/pretrained_eres2netv2w24s4ep4.ckpt` | 说话人向量（20480 维） | ~150 MiB |

---

## 2. 端到端流水线

```
参考音频 ──> 16k 重采样 ──> HuBERT ──> ssl(768,T/2)
                              │
                              ├─> sovits.ssl_proj + quantizer.encode ──> prompt codes
                              └─> mel(704 band) ──> sovits.ref_enc ──┐
参考音频 ──> eres2net ──> sv_emb(20480) ──> sovits.sv_emb ──> + ──> prelu ──> ge
                                                                        │
目标文本 ──> 前端(phones,word2ph) ──> BERT ──> bert_feature(1024,T)     │
                     │                                                  │
                     └──> AR(s1) 自回归出 semantic codes(T/2) ───────────┤
                                                                        │
        codes ──> quantizer.decode ──> ×2 nearest 上采样 ──> ssl(768, T)│
                                    │                                   │
                     enc_p(ssl, text, ge_to512(ge), speed) ─────────────┤
                                    │                                   │
                     m_p, logs_p ──> z_p = m_p + N(0,1)*exp(logs_p)*noise_scale
                                    │                                   │
                     flow(z_p, ge, reverse) ──> z ──> dec(z, ge) ──> wav 32k
```

关键数值（**实测自本地 `s2Gv2ProPlus.pth` 的 hps**，注意与 `configs/s2.json` 的 v2 底模不同）：

| 项 | v2 底模 | **v2ProPlus（本次目标）** |
|---|---|---|
| 采样率 / hop | 32000 / 640 | 同左（20 ms 一帧） |
| 语义帧率 | 25 Hz，SSL 上 ×2 nearest → 50 Hz | 同左 |
| `gin_channels` | 512 | **1024** |
| `inter / hidden / filter` | 192 / 192 / 768 | 同左 |
| `n_heads / n_layers` | 2 / 6 | 2 / 6 |
| `upsample_rates` / kernels | 10,8,2,2,2 / 16,16,8,2,2 | 10,8,2,2,2 / **20,16,8,2,2** |
| `upsample_initial_channel` | 512 | **768** |
| `sv_emb` | — | `Linear(20480 → 1024)` |
| `ge_to512` | — | `Linear(1024 → 512)`（只给 `enc_p`/MRTE 用） |
| `y_lengths` | `codes.len * 2`，**长度完全由 AR 决定** | 同左 |
| speed | `F.interpolate(y, size=len/speed+1, mode="linear")` | 同左 |
| noise_scale | 0.5 | 0.5 |

AR（`s1v3.ckpt` 实测）：24 层 / hidden 512 / 16 头 / FFN 2048 / `vocab_size=1025`（EOS=1024）
/ `phoneme_vocab_size=732` / 该权重自带 `inference.top_k=5`。

> **注意力实现注意**：`enc_p` 里不是普通 MHA。权重里有
> `enc_p.encoder_ssl.attn_layers.*.emb_rel_k/emb_rel_v`（形状 `(1, 9, 96)`），
> 说明走的是 VITS 的 **相对位置注意力**（`use_rel_pos=True, window=4`）。
> candle 没有现成实现，必须照 `module/attentions.py` 搬。

> **重要**：推理不需要 `StochasticDurationPredictor` 和 `DurationPredictor`，
> 也不需要 `enc_q`（posterior encoder）。转换脚本默认把 `enc_q.*` 丢掉。

---

## 3. 模块映射（Python → Rust）

| Python | Rust 目标 | 复用 aha 现有 |
|---|---|---|
| `feature_extractor.cnhubert.CNHubert` | `models/gpt_sovits/hubert.rs` | `models/feature_extractor`（16k 归一化） |
| `module.models.TextEncoder` | `models/gpt_sovits/text_encoder.rs` | `attentions` 需新写（VITS 风格） |
| `module.mrte_model.MRTE` | `models/gpt_sovits/mrte.rs` | 同上 |
| `module.modules.MelStyleEncoder` | `models/gpt_sovits/ref_encoder.rs` | 键名是 `ref_enc.spectral.{0,3}.fc.*`（`LinearNorm` 里套了 `fc`）、`ref_enc.temporal.{0,1}.conv1.conv.*`、`ref_enc.slf_attn.*`、`ref_enc.fc.fc.*` |
| `module.modules.WN`（flow 内） | `models/gpt_sovits/wn.rs` | — |
| `module.models.ResidualCouplingBlock` | `models/gpt_sovits/flow.rs` | — |
| `module.models.Generator` | `models/gpt_sovits/hifigan.rs` | `models/bigvgan`（ResBlock1 需补） |
| `AR.models.t2s_model.Text2SemanticDecoder` | `models/gpt_sovits/t2s.rs` | `models/qwen3/generate.rs`（采样框架） |
| `sv.ERes2NetV2` | `models/gpt_sovits/eres2net.rs` | `models/campplus`（CAM 层模板） |
| `text/*`（cleaner/中文/英文） | `models/gpt_sovits/text_frontend/` | 无，全新 |
| `TTS_infer_pack.TTS` | `models/gpt_sovits/pipeline.rs` + `exec/gpt_sovits.rs` | `exec/voxcpm.rs` |

张量前缀（转换后保持不变，Rust 侧按前缀取 VarBuilder）：

```
gpt.safetensors        ar_text_embedding.* / ar_audio_embedding.* / bert_proj.* /
                       h.encoders.* | h.layers.* / ar_predict_layer.*
sovits.safetensors     enc_p.* / dec.* / flow.* / ref_enc.* / ssl_proj.* /
                       quantizer.* / sv_emb.* / ge_to512.* / prelu.*
hubert.safetensors     feature_extractor.conv_layers.* / encoder.layers.*
bert.safetensors       bert.embeddings.* / bert.encoder.layer.*
sv_eres2net.*          cam.* / layer.* / seg.*
```

---

## 4. P0：权重转换（**已完成并已验证**）

`scripts/gpt_sovits/convert_weights.py` + `scripts/gpt_sovits/pth_reader.py`：

- **不依赖 torch**：`pth_reader.py` 直接解析 PyTorch 的 zip+pickle 容器
  （这些权重只用到 `OrderedDict` / `_rebuild_tensor_v2` / `HalfStorage` 三类引用）；
  旧格式权重可加 `--backend torch`。
- 识别 `s2G*.pth` 的**非标准文件头**（`05`/`06` = v2Pro/v2ProPlus，读取时补回 `PK`）；
- 合并 `weight_norm`（`weight_g`/`weight_v` → `weight`），`dec.ups`/`resblocks`/`flow.WN` 全靠这一步；
- 丢弃 `enc_q.*`（70 个张量）、`scaler_state_dict`、`bert.cls.*`、`bert.pooler.*`、
  `bert.embeddings.position_ids`、`hubert.masked_spec_embed`；
- **统一升到 f32** 落盘（官方权重全是 fp16；Python CPU 推理也是 f32，等价）；
- 导出 `manifest.json`：hps、t2s 配置、采样率、各模块参数量、张量形状清单；
  `symbols.json` 导出 732 个 phone 符号。

实测（本地 v2ProPlus 一套）：

| 文件 | 张量数 | 参数 | f32 体积 |
|---|---|---|---|
| `gpt.safetensors` | 295 | 77.6 M | 310 MiB |
| `sovits.safetensors` | 547 | 86.0 M | 344 MiB |
| `hubert.safetensors` | 209 | 94.4 M | 376 MiB |
| `bert.safetensors` | 389 | 324.5 M | 1298 MiB |
| `sv_eres2net.safetensors` | 1001 | 53.6 M | 214 MiB |
| 合计 | 2441 | 636 M | **2.37 GiB** |

正确性校验：131 组合并后的 `weight_norm` 全部满足不变量 `‖w[o]‖₂ = |g[o]|`
（唯一偏差出现在 `g` 为 fp16 次正规数 ~1e-6 的行，属精度极限）；无 NaN/Inf。

```bash
uv venv .venv && uv pip install numpy safetensors
.venv/Scripts/python scripts/gpt_sovits/convert_weights.py \
    --gsv-root D:/GPT-SoVITS \
    --out      D:/model_download/gpt-sovits-v2pro-zh \
    --version  v2Pro
```

> 内存风险：f32 全套 2.4 GiB。桌宠常驻进程里建议
> ①`bert` 用 q8/q4 存放，或 ②只在需要时加载（文本特征算完即可释放），
> 或 ③ `bert`/`hubert` 保持 f16 存储、按层转 f32 计算。

---

## 5. 文本前端（工作量最大，且与 ML 无关）

目标输入 → `phones: Vec<i64>`（`symbols2` 索引）+ `word2ph: Vec<usize>`（每个字对应的 phone 数）。

### 5.1 数据包（已导出）

`scripts/gpt_sovits/export_frontend_data.py` 把纯数据部分固化成两个文件（放进模型目录）：

| 文件 | 内容 | 体积 |
|---|---|---|
| `frontend_zh.json` | `chars`(41923 字→拼音)、`phrases`(47111 词→拼音)、`opencpop`(429 拼音→声母/韵母)、`tone_sandhi`(420 必读轻声 / 40 不得轻声 / 标点)、`rep_map`、`must_erhua`(8) / `not_erhua`(44) | 2.3 MiB |
| `dict.txt` | jieba 主词典（词 词频 词性），**同时用于分词和词性查找** | 4.8 MiB |

### 5.2 对拍基准（已就绪）

`scripts/gpt_sovits/reference_frontend.py` 直接 import 仓库里的
`text/chinese2.py` + `text/tone_sandhi.py` + `zh_normalization`，只做两处替身：
`jieba_fast → jieba`、`text.g2pw → pypinyin 兜底`，因此**不需要 onnxruntime / transformers / torch**。

基准样例（`tests/gpt_sovits_data/reference_zh.jsonl`，15 句）：

```
「今天天气不错，我们出去走走吧。」
j in1 t ian1 t ian1 q i4 b u2 c uo4 , w o3 m en5 ch u1 q v5 z ou3 z ou5 b a5 .
「银行门口有一棵大树。」   → y in2 h ang2 ...     （词库优先，多音字正确）
「他行走的速度很快。」       → t a1 x ing2 ...     （同样正确）
「现在是2024年8月15日，气温25度。」
x ian4 z ai4 sh ir4 EE er4 l ing2 EE er4 s i04 n ian2 ...（数字→中文读法）
「这件衣服花了我三百块钱。」 → ... y i1 f u5 ...    （必读轻声 f u5）
```

### 5.3 剩余实现（Rust）

1. `models/gpt_sovits/text_frontend/`：加载数据包 → 归一化 → `jieba-rs` 分词 → 词库优先拼音
   → 变调（PaddleSpeech 那套：3-3 变调、一/不 变调、轻声、儿化）→ `opencpop` 映射 → `phones`/`word2ph`。
2. **词性**：`jieba-rs` 没有 POS，用 `dict.txt` 里每个词的第一个 tag 查表（`了/着/过`、`们/子`、
   `上/下/里`、儿化判定都依赖 POS）；OOV 词走启发式。
3. **归一化**：`zh_normalization`（数字/日期/百分比/单位）是 PaddleSpeech 的一大坨，
   先做「整数/小数/年月日/百分比」最小集，其余原样透传并告警。

### 5.4 状态：✅ 中文链路已落地并通过对拍

代码：`src/models/gpt_sovits/text_frontend/`

| 文件 | 内容 |
|---|---|
| `mod.rs` | 数据加载、`jieba` 分词+词性、词库优先取拼音、opencpop 映射、`G2pOutput` |
| `tone_sandhi.rs` | PaddleSpeech `ToneSandhi` 全量移植（预合并 + 三声/一/不/轻声变调） |
| `normalize.rs` | 数字/日期/百分比归一化 + 标点替换 + 非法字符清理 |

对拍结果（`tests/test_gpt_sovits_frontend.rs`，26 句 / 626 个 phone）：

```
✓ 26 句、626 个 phone 与 Python 参考逐元素一致
```

覆盖：多音字（银行/行走）、年份日期（2024年12月31日）、百分比、儿化（小院儿/胡同儿/哪儿）、
叠词轻声（慢慢/看看）、一/不变调（一天/不怕/第一名）、量词「个」、ASCII 字母剥离。

**已修掉的两个坑**（都容易踩）：

1. `str.isnumeric()`：Python 里 `'三'.isnumeric()` 是 `True`，Rust 的 `char::is_numeric()` 不认汉字数字
   → 量词「三个」的轻声判定会失败。
2. pypinyin 的零声母切分：`yin2 → ("", "in2")`、`wo3 → ("w", "o3")`，规则不成体系，
   因此不从音系逆推，而是把全部 1546 个「带调音节 → (声母, 韵母)」离线导出成表直接查。

**仍待补**：

- G2PW 多音字消歧（BERT onnx），现在靠 `phrases` 词库兜底；
- 英文（`g2p_en` + CMU 词典 → ARPA 音素）、日文/粤语/韩文；
- `zh_normalization` 的长尾（电话号码、单位、分数、朝代等）。

---

## 6. P2：HuBERT + RoBERTa（已实现，待数值对齐）

| 文件 | 内容 |
|---|---|
| `hubert.rs` | `chinese-hubert-base`（wav2vec2 结构）：7 层卷积 + 12 层 post-norm transformer |
| `roberta.rs` | `chinese-roberta-wwm-ext-large` 的 encoder（24 层 BERT），取第 21 层 |
| `wordpiece.rs` | 读 HF `tokenizer.json` 的 BertNormalizer + BertPreTokenizer + WordPiece |

**对照 transformers 4.51 源码确认的两个关键细节**（都容易写错）：

1. `feat_extract_norm == "group"` 时，**只有第 0 层**卷积是
   `GroupNorm(groups=512, channels=512)`，其余 6 层是「纯 conv + GELU」。
   这正是这份权重里只有 `conv_layers.0.layer_norm.*` 的原因，不是权重缺失。
2. `encoder.layer_norm` 在**加上位置卷积之后、进 transformer 之前**生效
   （`hidden = LN(hidden + pos_conv(hidden))`），不是在最后。

自检结果（`tests/test_gpt_sovits_features.rs`）：

```
hubert: dims=[49, 768]  mean=-0.00776 std=0.20606 max|x|=3.8043   （1 秒 16kHz → 49 帧，320 倍下采样）
bert  : dims=[1, 17, 1024] mean=+0.01071 std=0.92245
bert_phone_level: dims=[1024, 28] → 与 word2ph 求和一致
```

**待办**：数值对齐需要 torch 参考输出，脚本已备好
（`scripts/gpt_sovits/export_reference_features.py`，在能跑 GPT-SoVITS 的环境执行一次即可）：

```bash
python scripts/gpt_sovits/export_reference_features.py --gsv-root D:/GPT-SoVITS --out ref_feats.npz
```

拿到 `ref_feats.npz` 后跑
`AHA_GPT_SOVITS_DIR=... AHA_GSV_REF_FEATURES=ref_feats.npz cargo test --features gpt-sovits --test test_gpt_sovits_features`，
逐层比 cos 相似度（目标 > 0.999）。

### 6.x T2S（AR）：已实现，待对齐

`t2s.rs` 对齐 `AR/models/t2s_model.py::Text2SemanticDecoder`：

- 文本 `ar_text_embedding + bert_proj(bert^T) + sine_pe`，提示 `ar_audio_embedding + sine_pe`；
- 拼成 `[text; prompt]` 做一次全量 attention，然后逐步用 KV cache 解码，最多 1500 步；
- 每层 post-norm：`x = LN1(x + attn); x = LN2(x + mlp(x))`，MLP 用 **ReLU**；
- 前 11 步屏蔽 EOS（`logits[:, :-1]`），EOS=1024，vocab=1025；
- 采样：重复惩罚 → top_p → 温度 → top_k → softmax → 多项式采样（复用
  `models/common/sample.rs::simple_sample`，顺序与官方 `logits_to_probs` 一致）。

**必须照抄的反直觉 mask**：

|          | 文本列 | 音频列 |
|----------|--------|--------|
| 文本行   | 可见   | **屏蔽** |
| 音频行   | 可见   | 因果 |

自检（`tests/test_gpt_sovits_ar.rs`）：24 层加载正常，greedy 两次结果逐 token 一致，
输出 token 全部落在 `[0,1024)`；`ref_free`（无提示音频）也能跑。

**待办**：`greedy` 前缀对齐。`export_reference_features.py` 已扩展出
`ref_ar.npz`（固定文本/提示、第一步 logits、前 10 步 greedy token），拿到后在 Rust 侧比对即可。

### 6.y SoVITS 合成器（P4，主体完成）

| 文件 | 内容 |
|---|---|
| `quantizer.rs` | `ssl_proj` + RVQ（1024 码字）→ prompt codes / 反向 decode |
| `style.rs` | 线性幅度谱 → `MelStyleEncoder`（+`sv_emb`+PReLU）→ `ge` |
| `vits_attn.rs` | VITS 相对位置注意力 + Conv-FFN + `Encoder` + `MRTE` |
| `flow.rs` | `WN` + `ResidualCouplingLayer` + `Flip`（4 层，推理走 reverse） |
| `hifigan.rs` | `ResBlock1` + `Generator`（768→24→1，640 倍上采样） |
| `sovits.rs` | `enc_p` 组装 + 端到端 `synthesize()` |

端到端冒烟（`tests/test_gpt_sovits_synth.rs`）：

```
phones: 18 个，norm_text: 你好,今天天气不错.
prompt codes: 44 个
AR 生成语义 token: 31 个
波形: 39680 采样点 @ 32000 Hz（1.24 s），峰值 0.0883
已保存 target/gpt_sovits_smoke.wav
```

时长自洽：31 codes → ×2 → 62 帧 → 62×640 = 39680 采样点 ✓

**实现时踩到的 4 个「照抄才不错」的细节**：

1. `flow` 的 WN 里 **最后一层 `res_skip` 只输出 `hidden`**（不是 `2*hidden`），`cond_layer` 的**输入是 `gin_channels`（1024）不是 2×**；
2. `flow` 的 `kernel_size` 固定 **5**，和 `enc_p` 的 `kernel_size=3` 不是同一个；
3. `dec.conv_post` **没有 bias**（`is_bias=False`），最终 `leaky_relu` 用的是默认斜率 **0.01**（前面都是 0.1）；
4. `ge` 只有 1 帧，所有 `+ ge` / `+ cond(g)` 都要**广播**到时间维（candle 的 `+` 不广播，必须 `broadcast_add`）。

**还没做**：v2Pro 的 `eres2net` 说话人模型（现在 `sv` 用 20480 维零向量占位），
以及和 Python 的波形级对齐（SNR / cos）。

## 6.z P5：接入（已完成）

- `WhichModel::GptSovits`（`--features gpt-sovits`，模型 id `gpt-sovits-v2pro-zh`）+ `model_type() = "tts"`；
- `aha run -m gpt-sovits-v2pro-zh --weight-path <dir> -i "你好" -o out.wav`；
- `aha serv -m gpt-sovits-v2pro-zh --weight-path <dir> -p 10100` → `POST /v1/tts` 返回 32 kHz WAV
  （`/v1/tts/stream` 见下）。

音色定义在模型目录的 `voices.json`：

```json
{ "default": { "ref_audio": "ref.wav", "speed": 1.0 } }
```

没写时退化到目录下的 `ref.wav`；参考音频自动重采样到 16 kHz（HuBERT）和 32 kHz（谱），
长度超出 1~12 s 会打 warning。

实测（debug 构建）：

```
POST /v1/tts → 200，519,724 字节 = 259,840 采样点 @32 kHz（8.1 s）
```

**性能提醒**：debug 构建 RTF ≈ 42（8 s 音频要 339 s），必须用 release。
本机 `--release` 构建会因为 `sentencepiece-sys` 需要 cmake 而失败
（`uv pip install cmake` 后把它加进 PATH 即可）。

## 7. 服务接口设计

`/v1/tts` 已有 `{text, voice?, speed?}`，GPT-SoVITS 的 `voice` 解释为「音色配置名」，
配置在模型目录的 `voices.json`：

### 流式合成 `/v1/tts/stream`

按标点把文本切成片段（短于 8 个字的片段会并到下一段），逐段合成、逐段推
PCM16LE。响应头 `X-Audio-Rate: 32000`、`X-Audio-Channels: 1`，body 无 WAV 头。

首音频延迟从「整段合成完」降到「第一句合成完」。实测（CPU，4 句文本，
预热后中位）：

| 模式 | 首音频 | 总时长 |
|---|---|---|
| `/v1/tts`（整段） | 7875 ms | 7875 ms |
| `/v1/tts/stream`（按句） | **2734 ms** | 6705 ms |

短文本（单句）两者一致，没有额外开销。


```json
{
  "default": {
    "ref_audio": "assets/voices/ref.wav",
    "ref_text": "参考音频对应的文本",
    "ref_lang": "zh",
    "speed": 1.0
  }
}
```

- 参考音频的 HuBERT 特征、`ge`、prompt codes 是**常量**，首次加载时算一次并缓存
  （v2Pro 的 `ge` 还依赖 eres2net，同样缓存）。
- 流式：按句切分 → AR 逐句出 codes → `decode` 出 PCM 块 → `y_`/`y_mask_` 做重叠拼接
  （v2Pro 支持 `decode_streaming` 的窗口重叠，但先做「整句合成」版本更稳）。

---

## 8. 分期与验收

| 阶段 | 内容 | 验收 |
|---|---|---|
| P0 | 权重转换 | ✅ 已通过：`manifest.json` 完整；无 `weight_g/v` 残留；131 组 `weight_norm` 满足 `‖w‖=|g|` |
| P1 | 中文前端 | ✅ 26 句 / 626 phone 与 Python 参考逐元素一致（`jieba-rs` + 变调移植 + 数据包） |
| P2 | HuBERT + RoBERTa | 🟡 结构跑通（形状/数值自检通过）；数值对齐待 torch 参考（`export_reference_features.py`） |
| P3 | T2S AR | 🟡 增量解码跑通、greedy 可复现；token 序列对齐待 `ref_ar.npz` |
| P4 | SoVITS v2Pro | 🟡 端到端出音（1.24 s / 32 kHz）；eres2net 说话人模型与波形对齐待做 |
| P5 | 接入 | ✅ `aha run` / `aha serv` + `POST /v1/tts` 出 32 kHz WAV；流式仍为 sherpa 专用 |
| P6 | v4 | CFM/DiT + 48k 声码器 |

---

## 10. 数值对齐记录（2026-11-11，GPU 环境）

装好 torch（`uv pip install torch --link-mode=copy`，绕开解压卡死）+ transformers 后逐级比对，
**查出 1 个真 bug 和 1 个基准错误**。

### 10.1 真 bug：HuBERT 位置卷积的 `weight_norm(dim=2)`

`encoder.pos_conv_embed.conv` 用的是 `nn.utils.weight_norm(conv, name="weight", dim=2)`
（**按 kernel 位置**归一化），它的 `weight_g` 形状是 `(1, 1, 128)`，不是 `(768, 1, 1)`。
转换脚本原来一律按 `dim=0` 合并，导致这个卷积的权重被放大约 **4.03 倍**。

后果链：prompt 语义 codes 全错 → AR 失去参考锚点 → greedy 输出 240 个相同 token
→ 声码器忠实地渲染成持续嗡声（就是听到的「全是噪声」）。

修复：`merge_weight_norm` 从 `g` 的形状反推归一化维度（`g` 中为 1 而 `v` 中 >1 的维）。

### 10.2 基准错误：参考波形是否归一化

GPT-SoVITS 的 `_set_prompt_semantic` 调用 `cnhuhbert_model.model(wav16k)`，**绕过了
feature_extractor**（不做 mean/var 归一化）；导出脚本原来走了 `fe(...)`，多做了归一化。
两者输出 cos 恰好 0.96869，与最初的「错配」完全一致。

| 模块 | 修复前 | 修复后 |
|---|---|---|
| BERT layer21 | 1.0000000 | 1.0000000 |
| HuBERT | 0.4805（权重错）→ 0.9687（权重修好但基准错） | **1.0000000** |

### 10.3 GPU 才暴露的问题

- 注意力里 `reshape → transpose` 之后必须 `.contiguous()`，CUDA 的 matmul 直接拒绝非连续张量
  （CPU 能跑属侥幸）；
- 音频解码/重采样/STFT 留在 CPU，模型张量再搬到 GPU（`apply_stft` 是逐帧 host 计算）；
- 构建需要 `vcvars64.bat`（提供 cmake + cl.exe），`NVCC_CCBIN` 指向 cl.exe，
  `CUDA_COMPUTE_CAP=120`（RTX 5060 / Blackwell），运行时 PATH 要带 `CUDA\v12.8\bin`。

### 10.4 修复前后波形指标（同一句话）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| RMS | 0.0086 | 0.2477 |
| 静音占比 | 70.3% | 14.4% |
| 200 Hz 能量占比 | 1% | 14% |
| 过零率 | 0.313 | 0.236 |

### 10.5 性能（RTX 5060 / 8 GB，debug 构建）

```
加载模型 1.20 s
合成 2.54 s → 2.84 s 音频（RTF 0.44）
```

同一套代码在 CPU debug 上是 RTF ≈ 42。

### 10.6 第二轮对齐：SoVITS 的三个 bug（2026-11-11）

在 GPU 上装好 torch + transformers + librosa 后，用官方 `SynthesizerTrn` 逐级对拍，
又查出 3 个 bug：

| # | 位置 | 现象 | 原因 | 修复 |
|---|---|---|---|---|
| 1 | `vits_attn::merge_heads` | `enc_attn0` cos ≈ 0（与参考正交） | 写成 `transpose(2,3)`，把 `(B,H,T,Kc)` 变成 `(B,Kc,T)` 后直接 reshape 成 `(B,T,C)`，头/通道/时间全混 | 改成 `transpose(1,2)` |
| 2 | VITS `LayerNorm` | 静默使用未训练的仿射参数 | `module.modules.LayerNorm` 的参数名是 **`gamma`/`beta`**，不是 `weight`/`bias`；按 `weight/bias` 读时 candle 用默认值（1/0）补齐，不报错 | `vits_layer_norm()` 优先读 `gamma/beta` |
| 3 | HuBERT `pos_conv` 的 `weight_norm(dim=2)` | prompt codes 全错 → AR 退化 → 嗡声 | 见 10.1 | 按 `g` 形状推断归一化维 |

对齐结果（同一批 codes / text / ge）：

| 级 | 修复前 cos | 修复后 cos |
|---|---|---|
| `enc_ssl_proj` | 1.0000000 | 1.0000000 |
| `enc_ssl` | 0.5360 | **1.0000000** |
| `enc_text` | 0.7771 | **1.0000000** |
| `enc_mrte` | 0.9670 | **1.0000000** |
| `enc2` | 0.7655 | **1.0000000** |
| `m` / `logs` | 0.8396 / 0.9685 | **1.0000000** |
| `z`（flow 输出） | 0.8278 | **1.0000000** |
| 波形 | 0.2090 | **0.9999981** |

> 教训：**candle 的 `get_with_hints` 遇到缺失键会用默认值静默补齐**（`layer_norm`/`group_norm` 的
> weight/bias 都是这样）。命名对不上不会报错，只会悄悄用错参数——所以对齐必须做，且要按级做。

修复后 GPU 实测（debug 构建）：`加载 1.20 s / 合成 2.30 s → 5.64 s 音频（RTF 0.41）`。

### 10.7 第三轮对齐：AR 的 sine PE（2026-11-11）

`SinePositionalEmbedding` 的 `div_term = exp(arange(0, dim, 2) * -(ln(10000)/dim))`
—— 注意是 `arange(0, dim, 2)`（步长 2），我在 Rust 里写成了 `arange(0, dim/2)`（步长 1），
**频率差了一倍**。修复后：

| 级 | cos |
|---|---|
| `xy`（拼好进 transformer 的输入） | **1.0000000** |
| 第一步 logits | **1.0000000** |
| greedy codes（前 240 个） | **逐 token 一致** |

### 10.8 eres2net（v2Pro 说话人向量，已实现）

- `eres2net.rs`：`ERes2NetV2(baseWidth=24, scale=4, expansion=4)` 全量移植
  （BasicBlock + AFF 融合 + layer3_ds + fuse34 + forward3）；
- 自带 kaldi fbank（povey 窗、去直流、0.97 预加重、**不乘 32768**——和 GSV 的 kaldi.py 一致）；
- `Sovits::sv(&wav16k)`：16 kHz、**不加 0.3 s 静音尾**（官方就是这么用的）。

对齐：**sv cos = 0.9988**（残余差异来自两端重采样路径不同：Rust 用 sinc，Python 比较时用
scipy 的 polyphase；实际推理两端都用各自的 fbank，一致）。

**端到端**（同一 codes / text / ge(真实 sv)）：

| 级 | cos |
|---|---|
| `m` / `logs` | 1.0000000 |
| `z` | 1.0000000 |
| **波形** | **0.9999981** |

### 10.9 性能

- **参考特征缓存**：`GptSovitsModel` 现在缓存 `(prompt_codes, ge)`，同一音色第二次合成
  跳过 HuBERT/谱/ref_enc/eres2net（省几百毫秒）；
- GPU 实测（RTX 5060，**debug 构建**，含 AR 采样 + 全部合成）：

```
加载模型 1.31 s
合成 0.70-0.82 s → 2.16-2.24 s 音频（RTF ≈ 0.33-0.38）
```

release 构建预计再快 3-5 倍（RTF < 0.1）。

## 9. 风险

1. **文本前端是长尾**：`jieba` 的切词结果直接影响 G2PW 和变调，Rust 侧 `jieba-rs`
   与 Python `jieba` 的分词会有差异 → 可能要用同一份词典并固定 HMM 开关。
2. **采样不可复现**：AR 的 `top_k=15` + 随机源，数值对齐只能靠 greedy 模式；
   人耳等价要靠试听。
3. **CPU 性能**：AR 是逐 token 的，24 层 512 隐藏；目标是短句 RTF < 1。
4. **音色一致性**：`noise_scale=0.5` 与 `speed` 会明显改变音色，参数要按参考实现固定。

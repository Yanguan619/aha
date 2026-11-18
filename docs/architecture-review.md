# 架构审查：内聚与耦合（工作记录）

> 这份文件是**工作记录 + 重构路线**，不属于上游文档（`docs/*.md` 里那份是给使用者看的）。
> 度量基于 2026-11 的 `main`，重构成一项就在这里打勾，别让它变成一份过期清单。

## 摘要

主要结构债有三条，其余问题都是它们的症状：

1. **一个模型有多套注册表**（散弹式修改）：`WhichModel` / `ModelInstance` / `load_model` / `exec/` / `models/` / `mirrors` / 文档。
2. **下载职责反转**：HTTP + 重试 + 续传机制在 `utils`，`models/download.rs` 只是策略外壳，所以"下载统一走 download.rs"这条规则没有结构约束。
3. **`utils/mod.rs` 是杂物间**：设备、张量 IO、下载、文本预处理、格式化混在一个 1.1k 行的文件里。

## 度量（2026-11 重构前）

| 信号 | 数值 |
|---|---|
| `models/` | 178 文件 / 46,391 行 |
| `exec/` | 25 文件 / 1,591 行（平均 64 行，薄适配） |
| `WhichModel` 引用 | model_mapping 54、cli 48、models/mod 37、server/api 22 |
| `WhichModel` 变体 / `ModelInstance` 臂 | 49 / 36（87 处引用） |
| `cfg(feature)` 密度 | models/mod.rs 38 处 / 547 行；model_mapping.rs 21 处 / 284 行 |
| `#[allow(unused)]` / `#[allow(dead_code)]` | 34 / 11 |
| 单模型触达文件数 | Kokoro 15 个；Qwen3_5_0_8B 3 个 |
| 引用 `WhichModel` 的 trait | `InferenceModel`、`GenerateModel`、`ExecModel` 三个重叠 |

## 问题清单

### P0-1 多套注册表（散弹式修改）

同一模型要在下列位置各写一遍：`WhichModel` 变体、`ModelInstance` 臂、`load_model`
分支、`exec/<model>.rs` + `exec/mod.rs` 声明、`models/<model>/` + `models/mod.rs` 声明、
`mirrors::ALL`、两份文档。

AGENTS.md「新增模型必须同时改四处」这条规则本身就是缺陷的登记。

**目标**：一张表（宏或 `ModelSpec` 数组）展开所有注册点，加模型只改一处 + 模型目录 + 文档。

### P0-2 下载职责反转

| 位置 | 内容 |
|---|---|
| `utils/mod.rs`（原 535–928） | `download_files_to(_with_ctl)`、`download_repo`、`get_with_range`、`download_file`、`DownloadFilter/Progress/Ctl/Summary`、`MS_RESOLVE`/`MS_UA` |
| `models/download.rs` | `mirror_entry` / `ensure_*` / `download_into` 策略外壳，实际调 utils |

**目标**：机制搬进 `models/download/`，`utils` 不再导出下载原语；模型侧只允许调 `ensure_*`。

### P0-3 utils 杂物间

`utils/mod.rs` 1,139 行里混了：设备/dtype 解析、pth 张量解析、下载、TTS 文本预处理、
CJK 判断、语言码映射、ASR 响应清洗、目录大小、字节格式化、时间戳、分桶。

**目标**：拆成 `utils/{device,dtype,tensor_io,text,path,fmt}.rs`，`mod.rs` 只做门面 +
`pub use`（调用点无需改动）。`prepare_tts_text` / `clean_asr_response` 最终应回到
`gpt_sovits` / `qwen3_asr` 的文本前端——模型方言不该在通用 utils 里。

### P1-1 `exec/` 与 `models/` 并行 + 三个重叠 trait（已处理）

`exec/` 只被 `cli` 一个地方用（`use aha::exec::*` glob），却摆在 **lib** 层当第四套注册表。
而且那个 `ExecModel` trait 是**假抽象**：`run` 签名里没有 `self`（纯静态方法），25 个
`XxxExec` 结构体从来没被实例化过——trait 只起到"命名空间"的作用。

**已做**：
- 删掉 `ExecModel` trait 和 25 个 `XxxExec` 空结构体，`impl ExecModel for XxxExec { fn run(..) }`
  摊平成模块级 `pub fn run(..)`（`gpt_sovits` 的 `run_with_device`、`qwen3_5`/`llama_cpp` 的
  `run_gguf` 同样处理；`Self::` 引用跟着改成函数名）
- `src/exec/` → `src/cli/exec/`：`lib.rs` 不再导出它，`cli/mod.rs` 改 `mod exec;`，
  搬过去的文件 `crate::models|utils` → `aha::models|utils`（从 lib 模块变成 bin 模块）
- 干掉 glob 导入：调用点从 `kokoro::KokoroExec::run` 变成 `exec::kokoro::run`，依赖可见

**没做**：合并 `InferenceModel` / `GenerateModel` / `ExecModel` 三个 trait。前两个是真 trait
（有 `self`、被 `ModelInstance` 的每个变体实现），`ExecModel` 已删，所以"三个重叠"已经变成
"两个各司其职"，没必要再动。

### P1-2 `model_mapping.rs` 的谓词（已处理）

**更正**：早先写的"9 个谓词每个都是对 49 个变体的 match"是错的——那 54 处 `WhichModel::`
大部分来自枚举定义和 `model_type()`，谓词本身多是 `matches!`。真正的问题只有两个：

1. `is_gguf()` / `is_onnx()` 靠 **model id 字符串**判断（`id.contains("gguf")`），等于让展示名
   决定后端路由：改一下 id 就可能静悄悄换掉加载路径。`is_onnx()` 更糟——它只命中 SenseVoice
   （id 里有 `sherpa-onnx`），而 SenseVoice 恰好是"pet delegated"，所以唯一的三处调用
   `is_onnx() && !is_pet_delegated_model()` **永远为假**，是死分支。
2. `is_pet_delegated_model` 有两个 cfg 互斥的定义——它存在的唯一理由就是给上面那个死分支兜底。

**已做**：`is_gguf` 改成显式列变体；`is_onnx` / `is_pet_delegated_model` 连同三处死分支一起删掉；
加了一条回归测试（`gguf_is_decided_by_variant_not_by_id`）。`model_type()` 保留原样——它确实
是一张"模型 → 类型"的表，属于数据，不是坏味道；合并它的机会在 P0-1。

### P1-3 通用层里的 `if 模型 == X`（已处理，结论：不做能力位表）

`models/mod.rs` 里那两条告警（`warn_sherpa_ignores_device` / `warn_npu_provider_on_candle`）
看着像"通用加载器里判断具体模型"，但**它们本来就该在加载器里**——加载器是全流程里唯一同时
知道"选了哪个后端"和"用户传了什么 device/provider"的地方。真正的毛病是它还要绕 cfg 去问
"这个模型是不是 ORT 的"。

**已做**（约 −12 行）：`is_sherpa_model()` 去掉外层 `#[cfg(any(sherpa-asr, sherpa-tts))]`，
改成无条件存在、没编后端时恒 `false`；加载器里那 10 行 `let is_ort = { #[cfg] {..} #[cfg(not)] {..} }`
收成 `let is_ort = model.is_sherpa_model();`。

**决定不做**：原先设想的"ModelSpec 能力位"（`accepts_device: bool` / `provider_kind: Candle|Ort`）
属于过度抽象——为了消除两处 `if` 而引入一张能力和加载器都要读的表，收益是负的。判断标准：
**能从现有谓词直接答的问题，就别新建一层。**

### P2 其余（已做）

| # | 问题 | 状态 |
|---|---|---|
| 1 | `models/voxcpm_refact/` 半成品重构（`generate code refactoring progress 1/3`） | ✅ **已收尾**：把老实现里它复用的零件（`audio_vae` / `config` / `minicpm4` / `tokenizer` / 主干算子层）搬进 refact，删掉老 `generate.rs`，再把 `voxcpm_refact` 提正为 `models/voxcpm`（类型去掉 `Refact` 后缀）。两套 `GenerateModel` 分支逐行等价，所以是等价替换 |
| 2 | 巨文件 `server/realtime.rs` 1883 行（非空行） | ✅ 1985 → **1597**，拆出 `protocol.rs`(151) / `audio.rs`(206) / `sentences.rs`(74)；编排（`run` / `Conn` / `run_response` / `tts_worker`）留在 `mod.rs` |
| 3 | `cli` 7 处 + `server` 3 处直接 import `models::laya::*` | ✅ `laya::route::MIN_CONFIDENCE` 收成 `laya::MIN_CONFIDENCE`（其余 10 处本来就是走模块面，方向正确） |
| 4 | "全部模型"清单多处维护 | ✅ **单源**：`WhichModel` 枚举是唯一来源 → `aha list --json` 导出 → `scripts/gen_model_docs.py` 重写两份文档的清单块（表格带 `--check` 可当 CI 门禁）→ `docs_list_every_model` 单测确保枚举里每个模型都出现在文档里。顺带把 `model_owner()` 里那个 `matches!(HyMT2Gguf)` 特例改成从 mirrors 仓库 id 反推（短 id 模型通用）。首次跑出来文档缺 7 条（VoxCPM2 / SenseVoice 全名 / Kokoro / Laya / TurboVLA / gpt-sovits / LFM2.5-VL-450M），已补齐 |
| 5 | `allow(unused)` 34 处 | ✅ 27 处收窄为 `allow(dead_code)` / `cfg_attr(not(feature=...))`；顺带清掉被压住的真死代码（`exec::gpt_sovits::run`）。默认 / sherpa / gpt-sovits / qnn / cuda 五种配置 **warning = 0**（`--features llama` 的 592 条来自 `llama_bindings.rs` 的 FFI 声明） |

## 做得好的地方（重构时别改坏）

- `mirrors.rs` 单点管仓库 id / 文件清单；`download.rs` 是策略唯一出口。
- `runtime/{candle,llama_cpp,qnn}` 分层干净，外部 C 后端不再散落。
- feature 命名与 CI 组合（default / sherpa / gpt-sovits / laya / vla / all-no-cuda）到位。
- AGENTS.md 的规则具体、可检查——除了规则 3 本身在登记缺陷。

## 状态（2026-11 收尾）

| 项 | 结论 | 结果 |
|---|---|---|
| 审查文档 | ✅ | 本文件 |
| P0-2 下载职责反转 | ✅ | 机制搬进 `models/download/http.rs`（393 行），`utils` 不再导出下载原语 |
| P0-3 `utils` 杂物间 | ✅ | 835 → 333 行，拆出 `device` / `path` / `format` / `text`，门面重导出、调用点零改动 |
| P1-1 `exec` 平行注册表 | ✅ | 删假 trait（`run` 是静态方法、25 个 struct 从未实例化），`exec/` 降级为 `cli/exec/`，去 glob |
| P1-2 谓词 | ✅ | `is_gguf` 改显式列变体；删 `is_onnx` + 三处**永远为假**的死分支 + `is_pet_delegated_model` 的两份重复定义；加回归测试 |
| P1-3 `if 模型 == X` | ✅（缩小范围） | `is_sherpa_model` 改为无条件存在，加载器少 12 行 cfg 绕圈；**明确不做能力位表**（见该节） |
| P2-1 `voxcpm_refact` 半成品 | ✅ | 零件搬过去、删老 `generate.rs`、refact 提正为主实现（3,516 → 3,227 行，一套实现一条链） |
| P2-2 巨文件 | ✅ | `server/realtime.rs` 1985 → 1597，拆出 `protocol` / `audio` / `sentences` |
| P2-3 laya 泄漏 | ✅ | `laya::route::MIN_CONFIDENCE` 收成 `laya::MIN_CONFIDENCE` |
| P2-4 清单多处维护 | ✅ | 枚举 → `aha list --json` → `scripts/gen_model_docs.py` → 文档，外加单测 `docs_list_every_model` |
| P2-5 `allow(unused)` | ✅ | 27 处收窄为 `allow(dead_code)` / `cfg_attr(not(feature=...))`；默认 / sherpa / gpt-sovits / qnn / cuda / **all-no-cuda** 六种配置 warning = 0 |
| P0-1 单点注册表 | ❌ **不做** | 它的本质是引入 `ModelSpec` 表 + fn pointer，属于新抽象；按"不过度抽象"的决定缓办。真要拾，从 `models/mod.rs` 的 `load_model` 分支入手，把表建在 `models/` 内部即可 |

验收口径（每一步都跑过）：`cargo check` 覆盖 default / sherpa / gpt-sovits / qnn / cuda /
**sherpa,gpt-sovits,laya,vla**（CI 的 all-no-cuda 组合）/ llama；`cargo test --lib` 全过；
`rustfmt` 干净；`scripts/build_android.py` 能过。

> 教训一条：验证矩阵必须包含 **all-no-cuda 组合**。只测单个 feature 会漏掉
> 「A 特性下只赋值一次、B 特性下二次赋值」这类交叉问题——`let mut model_path` 就是这么翻车的。

## 后续建议（要做的话，按这个顺序）

1. **`prepare_tts_text` / `clean_asr_response` 从 `utils/text.rs` 回到各自模型的前端**
   （`gpt_sovits` / `qwen3_asr`）——现在它们还在通用层，模型方言不该在 utils 里。
2. **`llama_bindings.rs` 的 592 条 warning**：`extern "C"` 声明的 `snake_case` 风格告警，
   惯例做法是给那个文件加 `#![allow(non_camel_case_types, non_snake_case)]` 并说明它是绑定的。
3. **`ModelInstance` + `load_model` 仍是两套清单**（P0-1 的残余）。不想引入表的话，退而求其次：
   给 `load_model` 的兜底分支加一条明确的错误信息 + 一个 `every_variant_reaches_a_loader`
   的测试（用不存在的路径喂每个变体，断言报错不是"未知模型"）。
4. **`server/realtime/mod.rs` 还有 1597 行**：下一步能拆的是 `Conn` 的会话状态方法（约 380 行），
   但 `run_response` / `tts_worker` 是编排主体，拆开反而难读——拆到合适为止，别为了行数拆。

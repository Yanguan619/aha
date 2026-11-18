# 架构审查：内聚与耦合（工作记录）

> 这份文件是**工作记录 + 重构路线**，不属于上游文档（`docs/*.md` 里那份是给使用者看的）。
> 度量基于 2026-11 的 `main`，重构成一项就在这里打勾，别让它变成一份过期清单。

## 摘要

主要结构债有三条，其余问题都是它们的症状：

1. **一个模型有多套注册表**（散弹式修改）：`WhichModel` / `ModelInstance` / `load_model` / `exec/` / `models/` / `mirrors` / 文档。
2. **下载职责反转**：HTTP + 重试 + 续传机制在 `utils`，`models/download.rs` 只是策略外壳，所以"下载统一走 download.rs"这条规则没有结构约束。
3. **`utils/mod.rs` 是杂物间**：设备、张量 IO、下载、文本预处理、格式化混在一个 1.1k 行的文件里。

## 度量（重构前）

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

### P1-1 `exec/` 与 `models/` 并行 + 三个重叠 trait

`exec/` 平均 64 行，干的是"CLI 参数 → 模型调用"的翻译，本质是命令实现。
[cli/mod.rs] 还用 `use aha::exec::*;` 做 glob 导入，依赖不可见。

**目标**：三个 trait 合成一个；`exec/` 降级为命令层（`cli/commands/`）；禁 glob 导入。

### P1-2 `model_mapping.rs` 是 god enum

9 个 `is_*` 谓词 + `as_string` / `display_name` / `model_list` / `model_owner` /
`model_type`，每个谓词都是对 49 个变体的 match；`is_pet_delegated_model` 在
182/195 行各定义一份（cfg 互斥）。

**目标**：谓词变成表里的数据字段（`kind: ModelKind`、`pet_delegated: bool`）。

### P1-3 通用层里的 `if 模型 == X`

- `models/mod.rs`：`warn_sherpa_ignores_device`、`warn_npu_provider_on_candle`
- `server/api.rs`：`if matches!(model, WhichModel::SenseVoice / Kokoro)`

**目标**：变成 ModelSpec 的能力位（`accepts_device`、`provider_kind`），加载器统一读。

### P2 其余

| # | 问题 | 处理 |
|---|---|---|
| 1 | `models/voxcpm_refact/` 死代码，且反向依赖 `models::voxcpm` 内部件 | 删或 feature 门控 |
| 2 | 巨文件：`server/realtime.rs` 1883、`utils/audio_utils.rs` 1666、`qwen3_5/model.rs` 1451、`deepseek_ocr/model.rs` 1399、`common/modules.rs` 1331 | 优先拆 realtime（协议 / 会话 / TTS worker） |
| 3 | `cli` 7 处 + `server` 3 处直接 import `models::laya::*` | 收成 `LayaFacade` |
| 4 | "全部模型"清单多处维护（mirrors / model_list / 两份文档 / cli 分支） | 加 `aha models --json` + CI 校验文档 |
| 5 | 34 处 `allow(unused)`、11 处 `dead_code` | 收窄到具体 item 并清理 |

## 做得好的地方（重构时别改坏）

- `mirrors.rs` 单点管仓库 id / 文件清单；`download.rs` 是策略唯一出口。
- `runtime/{candle,llama_cpp,qnn}` 分层干净，外部 C 后端不再散落。
- feature 命名与 CI 组合（default / sherpa / gpt-sovits / laya / vla / all-no-cuda）到位。
- AGENTS.md 的规则具体、可检查——除了规则 3 本身在登记缺陷。

## 路线（按风险从低到高）

- [x] 文档：本文件
- [x] P0-2 下载机制搬进 `models/download/`（纯搬迁 + 重导出，调用点不变）
- [ ] P0-3 拆 `utils/mod.rs` 为子模块 + 门面重导出
      —— **试过一次，回退了**：按顶层条目切分本身没问题（`mod.rs` 835 → 343 行，
      `device/text/path/format` 四个文件各就位、能编译），但把原来的 13 条 `use`
      复制进每个子文件后要逐个剪枝，自动化剪枝会破坏 `use a::{b, c}` 的分组语法。
      结论：这一步得手工做，一次一个子模块，剪完立刻编译验证；别用脚本批量剪 import。
- [ ] P1-2 谓词数据化（为 P0-1 铺路）
- [ ] P1-3 `if 模型 == X` → 能力位
- [ ] P0-1 单点注册表 + P1-1 合并 trait / `exec` 降级
- [ ] P2 逐项清理

每步的验收口径一致：`cargo check` 覆盖 default / sherpa / gpt-sovits / qnn / llama /
cuda 组合，`cargo test --lib` 全过，Android 构建能过（`scripts/build_android.py`）。

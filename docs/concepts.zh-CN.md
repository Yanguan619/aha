# 架构与设计

本文档深入介绍 AHA 的架构和设计原则。

## 概述

AHA（高性能 AI 推理引擎）是一个基于 Rust 的库，构建在 [Candle](https://github.com/huggingface/candle) 框架之上。它提供了统一的接口，可以在本地运行多个最先进的 AI 模型，无需 API 密钥或云服务。

### 核心特性

- **本地优先**：所有推理都在您的机器上运行
- **多模态支持**：支持文本、视觉、音频、OCR 和 ASR 模型
- **跨平台**：支持 Linux、macOS 和 Windows
- **GPU 加速**：可选的 CUDA 和 Metal 支持
- **内存安全**：使用 Rust 构建，确保安全性和性能
- **OpenAI 兼容**：易于与现有工具集成

## 架构原则

### 1. 本地优先设计

AHA 设计为完全在本地运行：

- **无云依赖**：所有模型都在本地下载和运行
- **隐私保护**：您的数据永远不会离开您的机器
- **无需 API 密钥**：下载后模型可永久使用
- **离线可用**：下载后模型无需互联网连接

### 2. 统一模型接口

所有模型都实现通用的 `GenerateModel` trait，提供：

- 不同模型类型之间的一致 API
- 无需更改代码即可轻松切换模型
- 支持实时输出的流式响应
- 标准化的错误处理

### 3. 跨平台支持

AHA 抽象了平台差异：

- **设备抽象**：自动 CPU/GPU 检测和选择
- **精度处理**：基于硬件动态选择 F32/F16/BF16
- **路径管理**：跨平台一致的模型存储

## 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI 层                              │
│  (main.rs - 命令解析、模型下载、服务管理)                     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                        HTTP API 层                          │
│  (api.rs - OpenAI 兼容端点、流式传输、认证)                   │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     模型抽象层                              │
│           (GenerateModel trait - 统一接口)                   │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌────────▼─────────┐  ┌───────▼────────┐
│   文本模型     │  │   视觉模型       │  │   音频模型     │
│  - Qwen3       │  │  - Qwen2.5VL     │  │  - VoxCPM      │
│  - MiniCPM4    │  │  - Qwen3VL       │  │  - VoxCPM1.5   │
└────────────────┘  └──────────────────┘  └────────────────┘
        │                     │                     │
┌───────▼────────┐  ┌────────▼─────────┐  ┌───────▼────────┐
│   OCR 模型     │  │   ASR 模型       │  │   图像模型     │
│  - DeepSeek    │  │  - GLM-ASR       │  │  - RMBG2.0     │
│  - Hunyuan     │  │  - Fun-ASR       │  │                │
│  - PaddleOCR   │  │  - Qwen3-ASR     │  │                │
└────────────────┘  └──────────────────┘  └────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                       工具模块                              │
│  - tokenizer: 分词工具                                       │
│  - chat_template: 聊天格式处理                               │
│  - position_embed: 位置编码                                  │
│  - utils: 通用工具（音频、图像、下载）                        │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Candle ML 框架                           │
│  (张量操作、模型加载、设备管理)                               │
└─────────────────────────────────────────────────────────────┘
```

### CLI 层 (`src/main.rs`)

CLI 层提供命令行界面功能：

- **命令解析**：使用 `clap` 进行参数解析
- **模型管理**：自动下载和缓存
- **服务控制**：启动/停止 HTTP 服务器
- **直接推理**：无需服务器即可运行模型

**可用命令**：
- `cli` - 下载模型并启动服务（默认）
- `serv` - 使用现有模型启动服务
- `download` - 仅下载模型
- `run` - 直接模型推理
- `list` - 列出支持的模型

### HTTP API 层 (`src/api.rs`)

HTTP API 层提供 REST 端点：

- **OpenAI 兼容**：匹配 OpenAI API 格式
- **流式支持**：实时响应生成
- **多模态**：处理文本、图像和音频
- **线程安全**：使用 RwLock 处理并发请求

**端点**：
- `POST /chat/completions` - 聊天和文本生成
- `POST /images/remove_background` - 图像背景移除
- `POST /audio/speech` - 文本转语音合成

### 模型抽象层

所有模型都实现 `GenerateModel` trait：

```rust
pub trait GenerateModel {

    // 生成响应
    fn generate(&mut self, prompt: &str, params: GenerationParams) -> Result<String>;

    // 流式生成
    fn generate_stream(&mut self, prompt: &str, params: GenerationParams)
        -> Result<Box<dyn Iterator<Item = Result<String>>>>;
}
```

这提供了：
- **多态性**：统一处理不同模型
- **可扩展性**：易于添加新模型
- **类型安全**：编译时保证

### 工具模块

#### 分词器 (`src/tokenizer/`)

- 从模型配置加载分词器
- 处理特殊标记
- 管理词汇表

#### 聊天模板 (`src/chat_template/`)

- 将聊天消息格式化为模型提示
- 支持多种聊天格式（ChatML 等）
- 处理系统消息和角色标签

#### 位置编码 (`src/position_embed/`)

- 为 transformer 实现位置编码
- 支持 RoPE（旋转位置编码）
- 处理多模态模型的 M-RoPE

#### 工具 (`src/utils/`)

- `audio_utils.rs` - 音频处理（WAV、MP3）
- `image_utils.rs` - 图像处理（调整大小、编码/解码）
- `tensor_utils.rs` - Tensor常用方法
- `mod.rs` - 通用工具和常量

## 设计模式

### 1. 基于 Trait 的抽象

`GenerateModel` trait 提供统一接口：

```rust
// 所有模型都实现此 trait
impl GenerateModel for Qwen3VL { /* ... */ }
impl GenerateModel for VoxCPM { /* ... */ }
impl GenerateModel for DeepSeekOCR { /* ... */ }

// 使用方式与模型无关
let mut model: Box<dyn GenerateModel> = load_model(model_type)?;
let result = model.generate(prompt, params)?;
```

### 2. 工厂模式

模型加载使用工厂函数：

```rust
pub fn load_model(
    model_type: WhichModel,
    model_path: &str,
    device: &Device,
) -> Result<Box<dyn GenerateModel>> {
    match model_type {
        WhichModel::Qwen3VL2B => Ok(Box::new(qwen3vl::generate::Qwen3VLGenerate::init(...)?)),
        WhichModel::VoxCPM1_5 => Ok(Box::new(voxcpm::generate::VoxCPMGenerate::init(...)?)),
        // ... other models
        _ => Err(anyhow!("Unsupported model: {}", model_type)),
    }
}
```

### 3. 命令模式

CLI 子命令封装不同的操作：

```rust
match command {
    Commands::Cli { model, port, address } => { /* 下载并启动服务 */ }
    Commands::Serv { model, weight_path, port } => { /* 仅启动服务 */ }
    Commands::Download { model, save_dir } => { /* 仅下载 */ }
    Commands::Run { model, input, weight_path } => { /* 直接推理 */ }
    Commands::List => { /* 列出模型 */ }
}
```

## 模型组织结构

每个模型都遵循一致的结构：

```
src/models/{model_name}/
├── config.rs       # 模型配置和生成参数
├── model.rs        # 核心模型架构（层、注意力）
├── generate.rs     # 推理逻辑（实现 GenerateModel trait）
├── processor.rs    # 模型特定处理（用于复杂模型）
└── mod.rs          # 模块声明和导出
```

### 示例：Qwen3VL

```
src/models/qwen3vl/
├── config.rs       # Qwen3VLConfig、GenerationConfig
├── model.rs        # Qwen3VL transformer 层、注意力机制
├── generate.rs     # Qwen3VLGenerate 实现
├── processor.rs    # 多模态输入的图像和文本处理
└── mod.rs          # 导出公共 API
```

## 性能优化

### GPU 加速

AHA 通过以下方式支持 GPU 加速：

- **CUDA**：用于 NVIDIA GPU（Linux、Windows）
- **Metal**：用于 Apple Silicon（macOS）

启用方式：
```bash
cargo build --features cuda    # NVIDIA GPU
cargo build --features metal   # Apple Silicon
```

### Flash Attention

Flash Attention 优化长序列处理：

- 减少内存使用
- 提高推理速度
- 对视觉模型特别有益

启用方式：
```bash
cargo build --features cuda,flash-attn
```

### 内存映射张量

模型使用内存映射文件：

- 更快的加载时间
- 减少内存占用
- 支持并发模型加载

### 精度优化

基于硬件的动态精度选择：

- **F32**：最高精度（仅 CPU）
- **F16**：平衡性能（GPU）
- **BF16**：最适合现代 GPU

## 安全考虑

### 本地处理

- 模型下载后无外部 API 调用
- 无遥测或数据收集
- 数据完全保留在本地

### 内存安全

- Rust 所有权系统防止内存泄漏
- 无缓冲区溢出或使用后释放错误
- 线程安全的并发操作

### 输入验证

- 文件大小限制（字符串 5MB，文件 100MB）
- 路径验证防止目录遍历
- 类型安全的请求处理

## 数据流

### 请求流程

```
┌─────────┐
│ 客户端  │
└────┬────┘
     │ HTTP 请求
     ▼
┌──────────────────────────────────────────────────────────┐
│  Axum   HTTP 服务器                                      │
│  - 将请求路由到端点                                      │
│  - 解析请求体                                            │
│  - 提取参数                                              │
└────────────┬─────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│  API 处理器 (api.rs)                                    │
│  - 获取模型锁                                            │
│  - 准备输入（分词、处理图像/音频）                        │
│  - 调用 model.generate() 或 generate_stream()           │
└────────────┬─────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│  模型实现 (models/{model}/generate.rs)                   │
│  - 从内存映射文件加载权重                                │
│  - 通过 Candle 张量运行前向传播                          │
│  - 解码输出标记                                          │
└────────────┬─────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│  Candle 框架                                            │
│  - 在 CPU 或 GPU 设备上执行                              │
│  - 管理张量操作                                          │
└────────────┬─────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│  响应生成                                               │
│  - 格式化响应（JSON / 流式）                             │
│  - 返回给客户端                                          │
└──────────────────────────────────────────────────────────┘
```

### 模型加载流程

```
用户指定模型
        │
        ▼
检查是否提供了 --weight-path
        │
    ┌───┴───┐
    │       │
   是       否
    │       │
    ▼       ▼
使用本地    从 ModelScope 下载
路径         │
    │        ▼
    │    保存到 ~/.aha/{model}/
    │        │
    └───┬────┘
        ▼
将模型权重加载到内存
        │
        ▼
初始化模型 (init())
        │
        ▼
准备就绪，可以进行推理
```

## 扩展点

### 添加新模型

1. 在 `src/models/` 下创建模型目录
2. 实现 `GenerateModel` trait
3. 在 `mod.rs` 的工厂函数中添加模型
4. 在 `main.rs` 中添加 CLI 映射
5. 在 `tests/` 中添加测试用例

### 自定义处理

模型可以覆盖默认处理：

- 自定义分词
- 特殊的输入/输出格式
- 模型特定的优化

## 另见

- [安装指南](./installation.zh-CN.md) - 设置和安装
- [快速入门](./getting-started.zh-CN.md) - 快速入门指南
- [API 参考](./api.zh-CN.md) - REST API 文档
- [开发指南](./development.zh-CN.md) - 贡献指南

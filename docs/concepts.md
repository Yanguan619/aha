# Architecture & Design

This document provides an in-depth look at the architecture and design principles behind AHA.

## Overview

AHA (High-performance AI inference engine) is a Rust-based library built on the [Candle](https://github.com/huggingface/candle) framework. It provides a unified interface for running multiple state-of-the-art AI models locally, without requiring API keys or cloud services.

### Key Characteristics

- **Local-First**: All inference runs on your machine
- **Multi-Modal**: Support for text, vision, audio, OCR, and ASR models
- **Cross-Platform**: Linux, macOS, and Windows support
- **GPU-Accelerated**: Optional CUDA and Metal support
- **Memory-Safe**: Built with Rust for safety and performance
- **OpenAI-Compatible**: Easy integration with existing tools

## Architecture Principles

### 1. Local-First Design

AHA is designed to run entirely on your local machine:

- **No cloud dependencies**: All models are downloaded and run locally
- **Privacy-preserving**: Your data never leaves your machine
- **No API keys required**: Once downloaded, models work indefinitely
- **Offline capable**: Models work without internet connection after download

### 2. Unified Model Interface

All models implement a common `GenerateModel` trait, providing:

- Consistent API across different model types
- Easy model switching without code changes
- Streaming response support for real-time outputs
- Standardized error handling

### 3. Cross-Platform Support

AHA abstracts platform differences:

- **Device abstraction**: Automatic CPU/GPU detection and selection
- **Precision handling**: Dynamic F32/F16/BF16 selection based on hardware
- **Path management**: Consistent model storage across platforms

## Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI Layer                           │
│  (main.rs - Command parsing, model download, service mgmt)  │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                        HTTP API Layer                        │
│  (api.rs - OpenAI-compatible endpoints, streaming, auth)     │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                     Model Abstraction Layer                  │
│           (GenerateModel trait - unified interface)          │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌────────▼─────────┐  ┌───────▼────────┐
│  Text Models   │  │  Vision Models   │  │  Audio Models  │
│  - Qwen3       │  │  - Qwen2.5VL     │  │  - VoxCPM      │
│  - MiniCPM4    │  │  - Qwen3VL       │  │  - VoxCPM1.5   │
└────────────────┘  └──────────────────┘  └────────────────┘
        │                     │                     │
┌───────▼────────┐  ┌────────▼─────────┐  ┌───────▼────────┐
│   OCR Models   │  │   ASR Models     │  │  Image Models  │
│  - DeepSeek    │  │  - GLM-ASR       │  │  - RMBG2.0     │
│  - Hunyuan     │  │  - Fun-ASR       │  │                │
│  - PaddleOCR   │  │  - Qwen3-ASR     │  │                │
└────────────────┘  └──────────────────┘  └────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Utility Modules                         │
│  - tokenizer: Tokenization utilities                         │
│  - chat_template: Chat format handling                       │
│  - position_embed: Positional embeddings                     │
│  - utils: Common utilities (audio, image, download)          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Candle ML Framework                        │
│  (Tensor operations, model loading, device management)       │
└─────────────────────────────────────────────────────────────┘
```

### CLI Layer (`src/main.rs`)

The CLI layer provides command-line interface functionality:

- **Command parsing**: Uses `clap` for argument parsing
- **Model management**: Automatic download and caching
- **Service control**: Start/stop HTTP server
- **Direct inference**: Run models without server

**Available Commands**:
- `cli` - Download model and start service (default)
- `serv` - Start service with existing model
- `download` - Download model only
- `run` - Direct model inference
- `list` - List supported models

### HTTP API Layer (`src/api.rs`)

The HTTP API layer provides REST endpoints:

- **OpenAI-compatible**: Matches OpenAI API format
- **Streaming support**: Real-time response generation
- **Multi-modal**: Handles text, images, and audio
- **Thread-safe**: Uses RwLock for concurrent requests

**Endpoints**:
- `POST /chat/completions` - Chat and text generation
- `POST /images/remove_background` - Image background removal
- `POST /audio/speech` - Text-to-speech synthesis

### Model Abstraction Layer

All models implement the `GenerateModel` trait:

```rust
pub trait GenerateModel {
    
    // Generate response
    fn generate(&mut self, prompt: &str, params: GenerationParams) -> Result<String>;

    // Generate with streaming
    fn generate_stream(&mut self, prompt: &str, params: GenerationParams)
        -> Result<Box<dyn Iterator<Item = Result<String>>>>;
}
```

This provides:
- **Polymorphism**: Treat different models uniformly
- **Extensibility**: Easy to add new models
- **Type safety**: Compile-time guarantees

### Utility Modules

#### Tokenizer (`src/tokenizer/`)

- Loads tokenizers from model configurations
- Handles special tokens
- Manages vocabulary

#### Chat Template (`src/chat_template/`)

- Formats chat messages into model prompts
- Supports multiple chat formats (ChatML, etc.)
- Handles system messages and role tags

#### Position Embeddings (`src/position_embed/`)

- Implements positional encoding for transformers
- Supports RoPE (Rotary Position Embedding)
- Handles M-RoPE for multimodal models

#### Utils (`src/utils/`)

- `audio_utils.rs` - Audio processing (WAV, MP3)
- `image_utils.rs` - Image processing (resize, encode/decode)
- `tensor_utils.rs` - Tensor utility methods
- `mod.rs` - Common utilities and constants

## Design Patterns

### 1. Trait-Based Abstraction

The `GenerateModel` trait provides a unified interface:

```rust
// All models implement this trait
impl GenerateModel for Qwen3VL { /* ... */ }
impl GenerateModel for VoxCPM { /* ... */ }
impl GenerateModel for DeepSeekOCR { /* ... */ }

// Usage is model-agnostic
let mut model: Box<dyn GenerateModel> = load_model(model_type)?;
let result = model.generate(prompt, params)?;
```

### 2. Factory Pattern

Model loading uses a factory function:

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

### 3. Command Pattern

CLI subcommands encapsulate different operations:

```rust
match command {
    Commands::Cli { model, port, address } => { /* download and serve */ }
    Commands::Serv { model, weight_path, port } => { /* serve only */ }
    Commands::Download { model, save_dir } => { /* download only */ }
    Commands::Run { model, input, weight_path } => { /* direct inference */ }
    Commands::List => { /* list models */ }
}
```

## Model Organization

Each model follows a consistent structure:

```
src/models/{model_name}/
├── config.rs       # Model configuration and generation parameters
├── model.rs        # Core model architecture (layers, attention)
├── generate.rs     # Inference logic (implements GenerateModel trait)
├── processor.rs    # Model-specific processing (for complex models)
└── mod.rs          # Module declaration and exports
```

### Example: Qwen3VL

```
src/models/qwen3vl/
├── config.rs       # Qwen3VLConfig, GenerationConfig
├── model.rs        # Qwen3VL transformer layers, attention mechanisms
├── generate.rs     # Qwen3VLGenerate implementation
├── processor.rs    # Image and text processing for multimodal input
└── mod.rs          # Exports public API
```

## Performance Optimizations

### GPU Acceleration

AHA supports GPU acceleration through:

- **CUDA**: For NVIDIA GPUs (Linux, Windows)
- **Metal**: For Apple Silicon (macOS)

Enable with:
```bash
cargo build --features cuda    # NVIDIA GPUs
cargo build --features metal   # Apple Silicon
```

### Flash Attention

Flash Attention optimizes long-sequence processing:

- Reduces memory usage
- Improves inference speed
- Especially beneficial for vision models

Enable with:
```bash
cargo build --features cuda,flash-attn
```

### Memory-Mapped Tensors

Models use memory-mapped files for:

- Faster loading times
- Reduced memory footprint
- Concurrent model loading

### Precision Optimization

Dynamic precision selection based on hardware:

- **F32**: Maximum accuracy (CPU-only)
- **F16**: Balanced performance (GPU)
- **BF16**: Best for modern GPUs

## Security Considerations

### Local-Only Processing

- No external API calls after model download
- No telemetry or data collection
- Data remains entirely on the local system

### Memory Safety

- Rust's ownership system prevents memory leaks
- No buffer overflows or use-after-free bugs
- Thread-safe concurrent operations

### Input Validation

- File size limits (5MB strings, 100MB files)
- Path validation to prevent directory traversal
- Type-safe request handling

## Data Flow

### Request Flow

```
┌─────────┐
│ Client  │
└────┬────┘
     │ HTTP Request
     ▼
┌──────────────────────────────────────────────────────────┐
│  Axum HTTP Server                                        │
│  - Route request to endpoint                             │
│  - Parse request body                                    │
│  - Extract parameters                                    │
└────────────┬─────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│  API Handler (api.rs)                                    │
│  - Acquire model lock                                    │
│  - Prepare input (tokenize, process images/audio)        │
│  - Call model.generate() or generate_stream()           │
└────────────┬─────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│  Model Implementation (models/{model}/generate.rs)       │
│  - Load weights from memory-mapped files                 │
│  - Run forward pass through Candle tensors              │
│  - Decode output tokens                                  │
└────────────┬─────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│  Candle Framework                                        │
│  - Execute on CPU or GPU device                          │
│  - Manage tensor operations                              │
└────────────┬─────────────────────────────────────────────┘
             │
             ▼
┌──────────────────────────────────────────────────────────┐
│  Response Generation                                     │
│  - Format response (JSON / streaming)                    │
│  - Return to client                                      │
└──────────────────────────────────────────────────────────┘
```

### Model Loading Flow

```
User specifies model
        │
        ▼
Check if --weight-path provided
        │
    ┌───┴───┐
    │       │
   Yes      No
    │       │
    ▼       ▼
Use local    Download from ModelScope
path         │
    │        ▼
    │    Save to ~/.aha/{model}/
    │        │
    └───┬────┘
        ▼
Load model weights into memory
        │
        ▼
Initialize model (init())
        │
        ▼
Ready for inference
```

## Extension Points

### Adding a New Model

1. Create model directory under `src/models/`
2. Implement `GenerateModel` trait
3. Add model to factory function in `mod.rs`
4. Add CLI mapping in `main.rs`
5. Add test case in `tests/`

### Custom Processing

Models can override default processing:

- Custom tokenization
- Special input/output formats
- Model-specific optimizations

## See Also

- [Installation Guide](./installation.md) - Setup and installation
- [Getting Started](./getting-started.md) - Quick start guide
- [API Reference](./api.md) - REST API documentation
- [Development](./development.md) - Contributing guide

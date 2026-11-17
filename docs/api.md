# API Reference

Complete reference for the AHA REST API.

## Overview

AHA provides an OpenAI-compatible REST API for running AI model inference. All endpoints follow standard HTTP conventions and return JSON responses.

### Base URL

By default, the API server runs on:
```
http://127.0.0.1:10100
```

You can customize this when starting the service:
```bash
aha cli -m Qwen/Qwen3-0.6B -a 0.0.0.0 -p 8080
```

### Startup warmup

Before the port starts listening, the server replays each loaded role once to move the
one-time costs of the *first* inference ahead of time:

- **ASR**: feeds 0.5s of silence through the recognizer;
- **TTS**: does a full synthesis of the shortest possible utterance with the first voice;
- **Decisions (Laya)**: runs one routing pass;
- **VLA**: runs one forward pass with a placeholder image and instruction (CUDA graph
  capture depends on the instruction so it can't be pre-captured, but the text/vision
  encoder kernels can be).

The first request otherwise pays for ONNX Runtime provider selection, weight staging and
CUDA kernel loading — exactly when the user starts speaking.

Measured (GPT-SoVITS v2Pro, CPU, median of 6 restarts):

| | first TTS request | steady state |
|---|---|---|
| warmup off | 1404 ms | ~850 ms |
| **warmup on** | **870 ms** | ~850 ms |

Warmup itself takes about 2s (one-off, before the port opens).

Use `--no-warmup` to skip it (useful when debugging or when the model lives elsewhere).

### Authentication

Currently, AHA does not require authentication. All endpoints are publicly accessible on the configured address/port.

**Security Note**: If you expose the API to external networks, consider implementing authentication through a reverse proxy (e.g., nginx, traefik).

### Content Types

All requests should use:
```
Content-Type: application/json
```

### Response Format

Success responses follow this structure:
```json
{
  "data": { ... },
  "model": "model-name",
  "usage": {
    "total_tokens": 30
  }
}
```

Error responses:
```json
{
  "error": {
    "message": "Error description",
    "type": "error_type",
    "code": "error_code"
  }
}
```

## Endpoints

### Health Check

Check the service health status. This endpoint is useful for container orchestration (Kubernetes), load balancers, and monitoring systems.

#### Endpoint
```
GET /health
```

#### Response

**Healthy (HTTP 200):**

```json
{
  "status": "ok"
}
```

**Unhealthy (HTTP 503):**

```json
{
  "status": "unhealthy",
  "error": "model not initialized"
}
```

#### Example

```bash
curl http://127.0.0.1:10100/health
```

### Runtime metrics and panel

`/metrics` returns in-process counters (nothing is persisted, no external service needed);
`GET /` serves a dependency-free HTML panel reading the same data (1 s polling):
per-route QPS / error rate / p50·p95·p99 / in-flight, the models loaded for the four roles
(asr / tts / system1 / vla), the process commit size and total-normalized CPU.

```
GET /metrics
GET /
```

Notes:

- Counters are keyed by **route** (`/audio/speech`, `/v1/audio/transcriptions`, `/v1/decisions`, …);
  requests that match no route are counted as `(unmatched)`, and `/metrics` / `/` themselves are not counted.
- Quantiles are the **upper bound** of a fixed millisecond bucket
  (5/10/25/50/100/250/400/600/800/1000/1500/2000/3000/…/30000) with no in-bucket interpolation,
  so with very few samples you will see values snapped to bucket edges.
- Streaming endpoints (`pcm` on `/audio/speech`, SSE on `/v1/chat/completions`) record the time from
  receiving the request to handing back the response / first chunk — not the client's full playback.
- `process.mem_mb` is the Windows commit size (`PagefileUsage`); aha's weights mostly live in commit
  rather than resident pages, so the working set is not a usable proxy for memory usage.

#### Example

```bash
curl http://127.0.0.1:10100/metrics
# open the panel in a browser
start http://127.0.0.1:10100/
```

### Models

Get information about the currently loaded model (OpenAI API compatible format).

#### Endpoint
```
GET /models
```

#### Response

**Success (HTTP 200):**

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen3-0.6B",
      "object": "model",
      "created": null,
      "owned_by": "Qwen"
    }
  ]
}
```

**Not Initialized (HTTP 503):**

```json
{
  "error": {
    "message": "Model not initialized",
    "type": "service_unavailable",
    "code": "model_not_loaded"
  }
}
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `object` | string | Fixed value: "list" |
| `data` | array | Array of model objects (currently contains one loaded model) |
| `id` | string | Model identifier in kebab-case (e.g., "Qwen/Qwen3-0.6B") |
| `object` | string | Fixed value: "model" |
| `created` | integer\|null | Unix timestamp (currently null) |
| `owned_by` | string | Model owner/organization name |

#### Example

```bash
curl http://127.0.0.1:10100/models
```

### Chat Completions

Generate chat completions or text responses.

#### Endpoint
```
POST /chat/completions
```

#### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier (e.g., "Qwen/Qwen3-0.6B") |
| `messages` | array | Yes | Array of message objects |
| `temperature` | number | No | Sampling temperature (0-2, default: 1) |
| `top_p` | number | No | Nucleus sampling (0-1, default: 1) |
| `max_tokens` | integer | No | Maximum tokens to generate |
| `stream` | boolean | No | Enable streaming (default: false) |

#### Message Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | Yes | "system", "user", or "assistant" |
| `content` | string/array | Yes | Message content (string or multimodal array) |

#### Multimodal Content

For vision/audio models, content can be an array:

```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "Describe this image"},
    {"type": "image", "image_url": {"url": "file:///path/to/image.jpg"}}
  ]
}
```

Supported content types:
- `text` - Text content
- `image_url` - Image file (file://, base64://, https:// or http://)
- `audio_url` - Audio file (file://, base64://, https:// or http://)

#### Examples

**Simple Chat:**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ]
  }'
```

**With System Message:**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain Rust in one sentence."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Vision Understanding:**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "What is in this image?"},
          {"type": "image", "image_url": {"url": "file:///path/to/image.jpg"}}
        ]
      }
    ]
  }'
```

**OCR (Text Extraction):**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-OCR",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Extract all text"},
          {"type": "image", "image_url": {"url": "file:///path/to/document.png"}}
        ]
      }
    ]
  }'
```

**ASR (Speech Recognition):**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ZhipuAI/GLM-ASR-Nano-2512",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Transcribe this audio"},
          {"type": "audio", "audio_url": {"url": "file:///path/to/audio.wav"}}
        ]
      }
    ]
  }'
```

> **Note:** For OpenAI-standard audio transcription with `multipart/form-data` file upload,
> see the [Audio Transcriptions](#audio-transcriptions) endpoint.

**Streaming Response:**

```bash
curl http://127.0.0.1:10100/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [
      {"role": "user", "content": "Tell me a story"}
    ],
    "stream": true
  }'
```

Streaming responses are sent as Server-Sent Events (SSE):
```
data: {"id": "1", "choices": [{"delta": {"content": "Once"}}]}

data: {"id": "1", "choices": [{"delta": {"content": " upon"}}]}

data: [DONE]
```

#### Response

**Non-streaming:**

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "Qwen/Qwen3-0.6B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I help you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "total_tokens": 19
  }
}
```

### Audio Speech

Generate speech from text (Text-to-Speech). One endpoint, two request shapes — picked by whether
the body carries a non-empty `input`.

#### Endpoint
```
POST /audio/speech
```

#### Shape 1: `{input}` — the TTS role (Kokoro / GPT-SoVITS)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input` | string | Yes | Text to synthesize |
| `voice` | string | No | Voice name (Kokoro: `zf_001`; GPT-SoVITS: a name from `voices.json`). Omit for the model default |
| `response_format` | string | No | `wav` (default, whole file) or `pcm` (chunked raw PCM16LE). Anything else is a 400, never a silent fallback |
| `speed` | number | No | 0.25–4.0, default 1.0 |
| `model` | string | No | Ignored (the loaded TTS is always used) |

```bash
curl -X POST http://127.0.0.1:10100/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello, this is AHA speaking.","voice":"zf_001"}' --output out.wav
```

With `response_format: "pcm"` the response carries `X-Audio-Rate` (24000 for Kokoro, 32000 for
GPT-SoVITS) and `X-Audio-Channels: 1`, and the body is chunked raw PCM pushed while it is
being synthesized.

#### Shape 2: `{messages}` — backends that treat TTS as a generative model (e.g. VoxCPM)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier (e.g. "OpenBMB/VoxCPM1.5") |
| `messages` | array | Yes | Array of message objects |

```bash
curl http://127.0.0.1:10100/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "OpenBMB/VoxCPM1.5",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Hello, this is AHA speaking."}
        ]
      }
    ]
  }'
```

The response is a chat-shaped JSON body (the audio sits in `content` as base64 WAV).

### Audio Transcriptions

Transcribe audio files to text (Automatic Speech Recognition).

This endpoint provides OpenAI-compatible audio transcription using `multipart/form-data` format.

#### Endpoints

```
POST /audio/transcriptions
POST /v1/audio/transcriptions
```

Both endpoints use the same handler and return identical responses. The `/v1/audio/transcriptions` path follows OpenAI's standard API convention.

When the `Content-Type` is **not** `multipart/form-data`, the body is treated as **raw PCM16LE
mono 16 kHz**: it goes straight into the model, no temp file on disk (the desktop-pet `/v1/asr`
alias is this same branch).

```bash
curl -X POST http://127.0.0.1:10100/v1/audio/transcriptions \
  -H "Content-Type: audio/pcm" \
  --data-binary @pcm16k.bin
# {"text":"...","dur_ms":1000,"took_ms":26}
```

Only 16000 Hz is accepted, declared with `X-Sample-Rate` (16000 by default). `dur_ms` / `took_ms`
are diagnostics; OpenAI clients ignore them.


#### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | Yes | The audio file to transcribe (wav, mp3, m4a, etc.) |
| `model` | string | No | Model identifier (optional, ignored - uses loaded model) |
| `language` | string | No | Language code (e.g., "zh", "en", "yue") |
| `prompt` | string | No | Optional text to guide transcription (not implemented, ignored) |
| `response_format` | string | No | Response format, only "json" or "text" supported (default: "json") |
| `temperature` | number | No | Sampling temperature (0.0 to 1.0, default: 0.0) |

#### Supported Languages

| Code | Language | Code | Language |
|------|----------|------|----------|
| `zh` | Chinese | `en` | English |
| `yue` | Cantonese | `ar` | Arabic |
| `de` | German | `fr` | French |
| `es` | Spanish | `pt` | Portuguese |
| `id` | Indonesian | `it` | Italian |
| `ko` | Korean | `ru` | Russian |
| `th` | Thai | `vi` | Vietnamese |
| `ja` | Japanese | `tr` | Turkish |
| `hi` | Hindi | `ms` | Malay |
| `nl` | Dutch | `sv` | Swedish |
| `da` | Danish | `fi` | Finnish |
| `pl` | Polish | `cs` | Czech |
| `fil` | Filipino | `fa` | Persian |
| `el` | Greek | `ro` | Romanian |
| `hu` | Hungarian | `mk` | Macedonian |

#### Examples

**Basic transcription:**

```bash
curl -X POST http://127.0.0.1:10100/audio/transcriptions \
  -H "Authorization: Bearer NO_NEED" \
  -F file="@./audio.wav" \
  -F model="Qwen/Qwen3-ASR-0.6B"
```

**With language specification:**

```bash
curl -X POST http://127.0.0.1:10100/v1/audio/transcriptions \
  -H "Authorization: Bearer NO_NEED" \
  -F file="@./chinese_audio.wav" \
  -F model="Qwen/Qwen3-ASR-0.6B" \
  -F language="zh"
```

**With temperature:**

```bash
curl -X POST http://127.0.0.1:10100/v1/audio/transcriptions \
  -H "Authorization: Bearer NO_NEED" \
  -F file="@./audio.wav" \
  -F model="Qwen/Qwen3-ASR-0.6B" \
  -F temperature="0.0"
```

#### Response

**Success (HTTP 200):**

```json
{
  "text": "Transcribed text from the audio file"
}
```

**Error (HTTP 400):**

```json
{
  "error": {
    "message": "Audio file is required",
    "type": "invalid_request_error",
    "code": "missing_file"
  }
}
```

**Error (HTTP 503):**

```json
{
  "error": {
    "message": "Model not initialized",
    "type": "service_unavailable",
    "code": "model_not_loaded"
  }
}
```

#### File Upload Limit

Maximum audio file size: 100 MB

### Images Remove Background

Remove background from images.

#### Endpoint
```
POST /images/remove_background
```

#### Request Body

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model identifier (e.g., "AI-ModelScope/RMBG-2.0") |
| `messages` | array | Yes | Array of message objects |

#### Example

**From File:**

```bash
curl http://127.0.0.1:10100/images/remove_background \
  -H "Content-Type: application/json" \
  -d '{
    "model": "AI-ModelScope/RMBG-2.0",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image_url": {"url": "file:///path/to/document.jpg"}}
        ]
      }
    ]
  }'
```

**From Base64:**

```bash
curl http://127.0.0.1:10100/images/remove_background \
  -H "Content-Type: application/json" \
  -d '{
    "model": "AI-ModelScope/RMBG-2.0",
    "messages": [
      {
        "role": "user",
        "content": [
          {"type": "image", "image_url": {"url": "base64://$(base64 -w 0 photo.png)"}}
        ]
      }
    ]
  }'
```

#### Response

Returns the processed image in base64 PNG format.

### Embeddings
Generate text embeddings.

#### Endpoints
```
POST /embeddings
POST /v1/embeddings
```

#### Request Body
| Parameter | Type | Required | Description |
|------|------|------|------|
| `model` | string | No | Model identifier (optional, ignored - uses loaded model)  |
| `input` | string or array | Yes | Text or array of texts to embed |

#### Examples
Single text：
```bash
curl http://127.0.0.1:10100/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Hello world"
  }'
```

Multiple texts：
```bash
curl http://127.0.0.1:10100/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "How are you?", "Goodbye"]
  }'
```

#### Response
**Success (HTTP 200):**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.1, 0.2, 0.3, ...]
    }
  ],
  "model": "model-name"
}
```

**Error (HTTP 400):**
```json
{
  "error": {
    "message": "embedding input must be a string or an array of strings",
    "type": "invalid_request_error",
    "param": "input",
    "code": "invalid_input"
  }
}
```

### Rerank
Re-rank a list of documents according to a query.

#### Endpoint
```
POST /rerank
POST /v1/rerank
```

#### Request Body
| Parameter | Type | Required | Description |
|------|------|------|------|
| `model` | string | No | 模型标识符 |
| `query` | string | Yes | Query text |
| `documents` | array | Yes | Array of document texts to re-rank |
| `top_n` | int | No | Return top N results (optional) |

#### Example
Basic re-ranking：
```bash
curl http://127.0.0.1:10100/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "documents": [
      "Machine learning is a form of artificial intelligence",
      "Apple is a fruit",
      "Deep learning belongs to the field of artificial intelligence"
    ]
  }'
```

Limit return count：
```bash
curl http://127.0.0.1:10100/rerank \
  -H "Content-Type: application/json" \
  -d '{
    "query": "artificial intelligence",
    "documents": [
      "Machine learning is a form of artificial intelligence",
      "Apple is a fruit", 
      "Deep learning belongs to the field of artificial intelligence"
    ],
    "top_n": 2
  }'
```

#### Response
**Success (HTTP 200):**
```json
{
  "object": "list",
  "model": "model-name",
  "results": [
    {
      "index": 0,
      "relevance_score": 0.95,
      "document": "Machine learning is a form of artificial intelligence"
    },
    {
      "index": 2,
      "relevance_score": 0.87,
      "document": "Deep learning belongs to the field of artificial intelligence"
    }
  ]
}
```

**Error (HTTP 400):**
```json
{
  "error": {
    "message": "rerank query cannot be empty",
    "type": "invalid_request_error",
    "code": "invalid_input"
  }
}
```

#### Parameter Description
| Parameter | Type | Description |
|------|------|-----|
| `model` | string | Model identifier |
| `object` | string | Fixed value: "list" |
| `results` | array | Re-ranked results array |
| `index` | int | Original document index |
| `relevance_score` | f32 | Relevance score (higher is more relevant) |
| `document` | string | Original document text |

### Realtime (WebSocket)

Speech-to-speech over a single WebSocket connection, shaped like OpenAI's Realtime API.
The client streams microphone audio in; the server runs **ASR → LLM → TTS** and streams the
synthesized audio (and/or text) back on the same socket. Everything happens inside this
process: no separate ASR/TTS/LLM services are involved, so the models must already be loaded
(see [`serv`](cli.md) and `GET /v1/models`).

#### Endpoints

```
GET /realtime        (with Upgrade: websocket)
GET /v1/realtime     (same handler)
```

Query parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Which loaded LLM to talk to (defaults to the server's default model) |
| `voice` | string | TTS voice passed to the speech model (default: `zf_001`) |

`/realtime` needs the chat model plus the `asr` and `tts` roles in the *same* process, so start it
with `aha multi --llm <dir> --asr <dir> --tts <dir>` and connect with `?model=llm` (the key `multi`
registers the chat model under). With a single-model `serv` you still get the session protocol, but
turns fail with `transcription_failed` / `tts_failed`.

If the client sends `Sec-WebSocket-Protocol: realtime` (the OpenAI SDKs do), the server echoes
it back. Audio is always **PCM16LE**; JSON events are text frames, audio may also be sent as
binary frames.

#### Talking to it

1. The server opens with `session.created`.
2. The client may adjust the session with `session.update` (voice, instructions, modalities,
   sample rates, turn detection).
3. The client streams audio: `input_audio_buffer.append` with base64 PCM16, or raw PCM16 in
   binary frames. Input defaults to 24 kHz (`input_audio_sample_rate`); the server resamples to
   the 16 kHz the ASR model wants.
4. With `turn_detection` set to `server_vad` the server decides when the user stopped talking
   and answers by itself. Otherwise the client sends `input_audio_buffer.commit` and then
   `response.create`.
5. The reply streams back as `response.audio.delta` (base64 PCM16, 200 ms chunks) plus
   `response.audio_transcript.delta` / `response.text.delta`; it ends with `response.done`.

#### Client events

| Event | Description |
|-------|-------------|
| `session.update` | Merge a `session` object (fields below). Answers `session.updated` |
| `input_audio_buffer.append` | `{"audio": "<base64 PCM16>"}` — append microphone audio |
| `input_audio_buffer.commit` | Run ASR on the buffered audio and add it as a user item |
| `input_audio_buffer.clear` | Drop the buffer. Answers `input_audio_buffer.cleared` |
| `conversation.item.create` | Add a message by hand (`input_text` / `text` / `input_image` / `image_url` parts; images work with multimodal LLMs) |
| `conversation.item.delete` | `{"item_id": "..."}` — drop one item from the conversation |
| `response.create` | Generate a reply from the conversation so far |
| `response.cancel` | Interrupt the response that is currently streaming |

`conversation.item.truncate` / `input_audio_buffer.truncate` are accepted and ignored (the audio
has already been sent to the client).

#### `session` fields

| Field | Default | Description |
|-------|---------|-------------|
| `model` | server default | LLM to use |
| `voice` | `zf_001` | TTS voice |
| `speed` | `1.0` | TTS speed (clamped to 0.25–4.0) |
| `instructions` | `""` | System prompt prepended to the conversation |
| `modalities` | `["audio"]` | Any of `audio`, `text` |
| `temperature` | `0.8` | Sampling temperature (0.0–2.0) |
| `max_response_output_tokens` | `"inf"` | Number or `"inf"` |
| `input_audio_sample_rate` | `24000` | 8000–96000, multiple of 100 |
| `output_audio_sample_rate` | `24000` | Same range; audio deltas use this rate |
| `input_audio_format` / `output_audio_format` | `pcm16` | Only `pcm16` is supported |
| `turn_detection` | `null` | `null` (manual) or `{"type":"server_vad", "threshold":0.02, "prefix_padding_ms":300, "silence_duration_ms":300, "create_response":true, "interrupt_response":true}` |
| `input_audio_transcription` | `{"model":"aha-asr"}` | Non-null enables ASR on commit; `null` disables |
| `transcription_enabled` | `true` | Convenience switch for the same thing |

#### Server events

| Event | Description |
|-------|-------------|
| `session.created` / `session.updated` | Full session object, `id` starts with `sess_` |
| `input_audio_buffer.speech_started` / `speech_stopped` | Emitted by `server_vad` |
| `input_audio_buffer.committed` | Buffered audio became a conversation item |
| `input_audio_buffer.cleared` | Buffer dropped |
| `conversation.item.created` / `conversation.item.deleted` | Item bookkeeping |
| `conversation.item.input_audio_transcription.completed` | ASR result for the user's utterance |
| `response.created` | A reply started |
| `response.output_item.added` / `response.content_part.added` | Reply scaffolding |
| `response.audio.delta` | Base64 PCM16 chunk (`200 ms`) at `output_audio_sample_rate` |
| `response.audio_transcript.delta` / `response.text.delta` | Text deltas (transcript of the audio, or plain text) |
| `response.audio_transcript.done` / `response.text.done` | Part finished |
| `response.output_item.done` / `response.content_part.done` | Item finished |
| `response.done` | Reply finished (`status`: `completed` / `cancelled` / `failed`), includes the final text |
| `error` | `{"error": {"type", "code", "message"}}` — the connection stays open |

Every event carries `type` and a unique `event_id`.

#### Example

```javascript
const ws = new WebSocket("ws://127.0.0.1:10100/realtime?voice=zf_001");
ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "response.audio.delta") playPcm16(msg.delta, msg.sample_rate);
};
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: "session.update",
    session: {
      modalities: ["audio", "text"],
      instructions: "你是一个桌面宠物，回答要短。",
      turn_detection: { type: "server_vad", silence_duration_ms: 600 },
    },
  }));
  // 每 20 ms 的麦克风 PCM16（24 kHz）直接当二进制帧发
  mic.on("data", (pcm) => ws.send(pcm));
};
```

#### Notes and limitations

- Only `pcm16` is supported (no G.711 / `g711_ulaw`) — the desktop pet needs raw samples anyway.
- Tool calls are not implemented; `tools` in `session.update` is accepted but ignored.
- `turn_detection` uses a lightweight energy VAD (RMS over 20 ms frames), not a neural one:
  no GPU cost, and it reacts within one frame, at the price of being loudness-based.
- Text generation holds the LLM's write lock, so one reply at a time per model; audio synthesis
  runs concurrently with generation, which is what keeps the time-to-first-audio short.
- Audio deltas are cut every 200 ms; the TTS model itself produces a sentence at a time.
- `usage` is not reported (`null`).

#### Latency tuning

Time to first audio (commit the buffer → first `response.audio.delta`) measured at
**0.55–0.87 s** on an RTX 4060 Laptop / 32 cores with Qwen3-0.6B + SenseVoice + Kokoro —
the same on a cold server, because the chat model is warmed before the port opens. Broken down:
ASR 79 ms → first LLM token (38–324 ms once warm) → first sentence (5 chars) 117–150 ms →
first TTS chunk 308–440 ms.

Highest-leverage knobs, roughly in order:

1. **Use a `--release` build.** In a debug build candle is unoptimized and the same turn took
   160 seconds (measured, not a typo). `Cargo.toml` now sets
   `[profile.dev.package."*"] opt-level = 3`, but release is still what you want.
2. **`--threads 16`.** sherpa-onnx defaults to 4 threads for ASR/TTS and `--threads 0`
   only auto-picks 4–8. A 5-char utterance dropped from 527 ms to 312 ms.
3. **`--device cuda`.** Affects the candle LLM only; sherpa ASR/TTS stay on CPU (a WARN line
   in the startup log says so).
4. **`enable_thinking: false`** (the default). With the thinking chain on, the model emits
   thousands of reasoning tokens first — in a voice loop that means no audio at all.
5. **4-character first sentence.** The first sentence is flushed at a comma once it has 4
   chars (「小柴迎客，」), later sentences wait for 24; see `FIRST_SENTENCE_MIN_CHARS`.
6. `turn_detection.silence_duration_ms` defaults to 300 — the wait between you stopping and
   recognition starting. Measured turn latency (last speech frame → first audio) at 16 threads:
   **0.97–1.2 s** at 300 ms. Going to 250 cuts 「你好，我是小桌宠，」 at its comma pause and
   splits one sentence into two turns; 500 feels sluggish by comparison.

### Graceful Shutdown

Gracefully shut down the AHA server. This endpoint initiates a graceful shutdown process that:
1. Stops accepting new connections
2. Waits for existing requests to complete (up to 1 second)
3. Cleans up PID files
4. Exits the process

#### Endpoint
```
POST /shutdown
```

#### Request Body

None (empty request)

#### Response

**Success (HTTP 200):**

```json
{
  "message": "Shutting down..."
}
```

**Forbidden (HTTP 403):**

When remote shutdown is not allowed:

```json
{
  "error": {
    "message": "Remote shutdown not allowed. Use --allow-remote-shutdown flag to enable (not recommended).",
    "type": "invalid_request_error",
    "code": "remote_shutdown_forbidden"
  }
}
```

#### Security

By default, the shutdown endpoint only allows requests from localhost (127.0.0.1). To enable remote shutdown, start the server with the `--allow-remote-shutdown` flag:

```bash
aha serv -m Qwen/Qwen3-0.6B --allow-remote-shutdown
```

**Warning:** Enabling remote shutdown is not recommended for production use unless properly secured.

#### Example

```bash
curl -X POST http://127.0.0.1:10100/shutdown
```

#### Logging

All shutdown requests are logged to stderr with the format:

```
[SHUTDOWN] Shutdown requested (remote_allowed: false)
```

## Error Handling

### Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters or unparsable request body |
| 404 | Not Found - Unknown endpoint |
| 405 | Method Not Allowed - Endpoint exists but does not accept this HTTP method |
| 413 | Payload Too Large - Request body exceeds the size limit (5 MiB JSON / 100 MiB multipart) |
| 415 | Unsupported Media Type - Wrong `Content-Type` for the endpoint |
| 500 | Internal Server Error - Model inference error |
| 503 | Service Unavailable - Model not loaded |

### Error Response Format

Every error response — including 404/405/413/415 generated by the framework — uses the
same OpenAI-compatible envelope. `param` and `code` are omitted when not applicable.

```json
{
  "error": {
    "message": "Model 'unknown-model' not found",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

## Rate Limiting

Currently, AHA does not implement rate limiting. The server can handle concurrent requests limited only by system resources.

## File Upload Limits

- String data: 5 MB
- File uploads: 100 MB

## OpenAI Compatibility

AHA's text generation API is designed to be compatible with OpenAI's API format. Multimodal APIs are derived from the text generation API with minimal changes:

### Python Example

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:10100",
    api_key="dummy"  # Not used but required by library
)

response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript Example

```javascript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://127.0.0.1:10100',
  apiKey: 'dummy'  // Not used but required
});

const response = await client.chat.completions.create({
  model: 'Qwen/Qwen3-0.6B',
  messages: [{ role: 'user', content: 'Hello!' }]
});

console.log(response.choices[0].message.content);
```

## Best Practices

### 1. Use Streaming for Long Responses

For long text generation, use streaming to get responses incrementally:

```bash
curl ... -d '{"stream": true, ...}'
```

### 2. Set Appropriate Token Limits

Prevent excessively long responses:

```json
{
  "max_tokens": 500
}
```

### 3. Adjust Temperature

Control response creativity:
- `0.0-0.3`: Deterministic, focused
- `0.4-0.7`: Balanced (default: 1.0)
- `0.8-2.0`: Creative, varied

### 4. Use System Messages

Set behavior with system messages:

```json
{
  "messages": [
    {"role": "system", "content": "You are a technical writer."},
    {"role": "user", "content": "..."}
  ]
}
```

## See Also

- [Getting Started](./getting-started.md) - Quick start guide
- [CLI Reference](./cli.md) - Command-line usage
- [Installation](./installation.md) - Installation guide
- [Development](./development.md) - Contributing guide

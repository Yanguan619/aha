//! Optional sherpa-onnx backends (SenseVoice ASR / Kokoro TTS).
//!
//! Enabled with `--features sherpa`.

pub mod kokoro;
pub mod sense_voice;

pub use kokoro::KokoroTtsModel;
pub use sense_voice::SenseVoiceModel;

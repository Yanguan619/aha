/* aha C ABI — embed the aha inference engine in any runtime.
 *
 * Build:  cargo build -p aha-ffi --release [--features llama]
 * Output: target/<triple>/release/libaha_ffi.so   (or .dylib / .dll / .a)
 *
 * Conventions
 * -----------
 *  - Handles are opaque `void*`; free them with the matching `*_free`.
 *  - Strings returned by this library are owned by the caller: free with `aha_string_free`.
 *    Strings passed *in* stay the caller's property.
 *  - Errors: functions take `char** err`. On failure they return NULL / -1, write an owned
 *    message into `*err` (free it with `aha_string_free`), and leave `*err` NULL on success.
 *  - Nothing unwinds across this boundary: a Rust panic becomes an error string.
 */
#ifndef AHA_H
#define AHA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---------------------------------------------------------------- library ---- */

/* "aha-ffi <version>" (+ " + llama.cpp" when that backend is compiled in). */
char *aha_version(void);

void aha_string_free(char *value);

/* ------------------------------------------------------------- chat models ---- */

/* model_type: a WhichModel name, e.g. "glm-asr-nano-2512", "qwen3-0.6b". */
void *aha_model_load(const char *model_type, const char *path, int threads, char **err);
void  aha_model_free(void *handle);

/* One utterance of mono PCM16. sample_rate is normally 16000. */
char *aha_transcribe(void *handle, const uint8_t *pcm, size_t len, uint32_t sample_rate,
                     const char *prompt, int max_tokens, char **err);

/* messages_json: a bare message array or a full request object. */
char *aha_generate(void *handle, const char *messages_json, int max_tokens,
                   int temperature_percent, char **err);

/* Streaming callbacks. `user` is passed back untouched. `text` is only valid during the call. */
typedef struct {
    void (*on_chunk)(const char *text, void *user);
    void (*on_done)(const char *full_text, void *user);
    void (*on_error)(const char *message, void *user);
    void *user;
} AhaCallbacks;

/* Returns 0 on success, -1 on failure. */
int aha_generate_stream(void *handle, const char *messages_json, int max_tokens,
                        int temperature_percent, const AhaCallbacks *callbacks, char **err);

/* ------------------------------------------------- streaming ASR session ---- */

/* vad_dir: FireRedVAD files. asr_dir: ASR model dir. partial_ms: 0 => 700. */
void *aha_stream_create(const char *vad_dir, const char *asr_type, const char *asr_dir,
                        int partial_ms, char **err);
void  aha_stream_free(void *handle);

typedef struct {
    void (*on_speech_start)(void *user);
    void (*on_partial)(const char *text, void *user);
    void (*on_final)(const char *text, void *user);
    void (*on_error)(const char *message, void *user);
    void *user;
} AhaAsrCallbacks;

/* Feed PCM16 frames (any size; 16 kHz mono expected, other rates are resampled).
 * Returns 0 on success, -1 on failure. */
int aha_stream_push(void *handle, const uint8_t *pcm, size_t len, uint32_t sample_rate,
                    const AhaAsrCallbacks *callbacks, char **err);

/* Flush the tail when the speaker stops. Returns 0 on success, -1 on failure. */
int aha_stream_flush(void *handle, const AhaAsrCallbacks *callbacks, char **err);

/* --------------------------------------------------------- model download ---- */

/* Where the weights of `model_id` live (or will live): `<save_dir>/<model_id>`.
 * `save_dir` may be NULL => ~/.aha, or $AHA_HOME when set (Android: set it, there is no home).
 * Returns an owned string (free with aha_string_free), or NULL with *err set.
 *
 * After a successful download, pass this path to aha_model_load(). */
char *aha_model_dir(const char *model_id, const char *save_dir, char **err);

/* Progress callback: `path` is the file being written (valid only during the call),
 * `done` / `total` are bytes for the whole job. `user` is passed back untouched. */
typedef void (*AhaDownloadProgress)(const char *path, uint64_t done, uint64_t total, void *user);

/* Downloads the weights of `model_id` from its ModelScope mirror. Blocking: call it from a
 * worker thread, never from the UI thread.
 *
 *  - model_id: the `-m` id, e.g. "qwen3.5-gguf", "gpt-sovits-v2pro-zh", "OpenBMB/TurboVLA",
 *              "csukuangfj/kokoro-multi-lang-v1_1". An id with no built-in mirror is treated
 *              as a ModelScope repo id and downloaded whole.
 *  - max_retries: per-file attempts, <= 0 means 1.
 *  - on_progress: may be NULL (poll aha_model_download_done/total instead).
 *
 * Returns 0 on success, -1 on failure (*err set), -2 when cancelled.
 * Safe to call again after being killed: finished files are skipped by size. */
int aha_model_download(const char *model_id, const char *save_dir, int max_retries,
                       AhaDownloadProgress on_progress, void *user, char **err);

/* Requests cancellation of the running aha_model_download (takes effect at the next chunk).
 * Returns immediately; the download call then returns -2. Safe from another thread. */
void aha_model_download_cancel(void);

/* Progress counters of the last aha_model_download, for polling hosts. */
uint64_t aha_model_download_done(void);
uint64_t aha_model_download_total(void);
uint64_t aha_model_download_files_done(void);
uint64_t aha_model_download_files_total(void);

/* ------------------------------------------------------- llama.cpp backend ---- */

/* Present only when built with `--features llama`. Runs any GGUF llama.cpp understands,
 * including architectures candle has no implementation for (e.g. hunyuan-dense / STQ). */
void *aha_llama_load(const char *path, int n_ctx, int threads, char **err);
void  aha_llama_free(void *handle);
char *aha_llama_chat(void *handle, const char *messages_json, int max_tokens,
                     int temperature_percent, char **err);
int   aha_llama_chat_stream(void *handle, const char *messages_json, int max_tokens,
                            int temperature_percent, const AhaCallbacks *callbacks, char **err);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* AHA_H */

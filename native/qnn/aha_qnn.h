// Minimal C ABI over Qualcomm QNN (QAIRT), for loading a *context binary* and
// executing one of its graphs.
//
// The point of this shim is to keep QNN's C++-flavoured headers out of the Rust
// tree: QnnInterface_t / QnnSystemContext_GraphInfo_t are big structs whose
// layout must match the SDK exactly, and the compiler does that for us here.
// aha loads the shim at runtime (dlopen), so a build without the shim still
// compiles and runs.
//
// Every entry point returns 0 on success and -1 on failure; the `err` buffer,
// when given, receives a human readable reason (never longer than err_len).

#ifndef AHA_QNN_H
#define AHA_QNN_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AHA_QNN_MAX_DIMS 8
#define AHA_QNN_MAX_NAME 128

typedef struct AhaQnnTensor {
  char name[AHA_QNN_MAX_NAME];
  // Tensor id inside the graph. QNN matches tensors by id when executing, so it
  // has to travel from the context binary metadata to the caller and back.
  int32_t id;
  // QNN data type enum value (Qnn_DataType_t), e.g. 0x0350 for FLOAT_32.
  int32_t data_type;
  int32_t rank;
  int32_t dims[AHA_QNN_MAX_DIMS];
  // Element count implied by dims, filled in by the descriptor call.
  int64_t elements;
  // Client buffer for graph execution; ignored by the descriptor call.
  void *data;
} AhaQnnTensor;

typedef struct AhaQnnSession AhaQnnSession;

// Opens a context binary. `backend_lib` is usually "libQnnHtp.so" (an absolute
// path also works, and is what Android needs), `system_lib` "libQnnSystem.so".
// The device side QNN runtimes must be reachable through the loader path.
AhaQnnSession* aha_qnn_open(const char *backend_lib, const char *system_lib,
                            const char *context_bin, char *err, int err_len);

// Opens a compiled model library (the lib<name>.so that
// qnn-model-lib-generator emits next to <name>.bin) instead of a context
// binary. The library composes its own graphs, which is the only way to run
// graphs whose IO is float32 on HTP v81 (graphExecute on a context binary of
// that shape fails with "Dma execution failed on the skel side").
AhaQnnSession* aha_qnn_open_model_lib(const char *backend_lib,
                                      const char *model_lib, char *err,
                                      int err_len);

void aha_qnn_close(AhaQnnSession *session);
int aha_qnn_graph_count(AhaQnnSession *session);
const char *aha_qnn_graph_name(AhaQnnSession *session, int graph_index);

// `is_input` != 0 lists graph inputs, 0 lists outputs.
int aha_qnn_tensor_count(AhaQnnSession *session, int graph_index, int is_input);
int aha_qnn_tensor_desc(AhaQnnSession *session, int graph_index, int is_input,
                        int tensor_index, AhaQnnTensor *out);

// Bytes per element for a QNN data type value, so callers never have to mirror
// Qnn_DataType_t. Returns 0 for types the shim does not know.
size_t aha_qnn_bytes_per_element(int32_t data_type);

// Runs one graph. `inputs[i].data` must be a client buffer of exactly
// `elements * sizeof(dtype)` bytes, and so must each `outputs[i].data`.
// QNN matches tensors by `id`, so descriptors must be the ones returned above
// (they carry name, id, dtype and dims); callers only fill in `data`.
int aha_qnn_execute(AhaQnnSession *session, int graph_index,
                    const AhaQnnTensor *inputs, int input_count,
                    const AhaQnnTensor *outputs, int output_count, char *err,
                    int err_len);

#ifdef __cplusplus
}
#endif

#endif  // AHA_QNN_H

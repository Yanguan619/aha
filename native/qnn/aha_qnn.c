#include "aha_qnn.h"

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "QnnBackend.h"
#include "QnnContext.h"
#include "QnnDevice.h"
#include "QnnGraph.h"
#include "QnnLog.h"
#include "QnnTensor.h"
#include "QnnTypes.h"
#include "System/QnnSystemContext.h"
#include "System/QnnSystemInterface.h"

#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

// One tensor as described by the context binary.
//
// `desc` is the descriptor verbatim as QNN reported it. Reconstructing one by
// hand loses the quantization encoding and the exact `clientBuf.dataSize` the
// graph expects, and HTP then rejects the call with a generic error, so the
// descriptor is copied around and only `type` / `clientBuf.data` are overridden
// at execute time.
typedef struct {
  const char *name;
  uint32_t id;
  int32_t data_type;
  uint32_t rank;
  const uint32_t *dims;
  Qnn_Tensor_t desc;
} Meta;

// One graph of the context binary. QNN keeps the names and shapes; the runtime
// only has to point client buffers at them.
typedef struct {
  const char *name;
  Meta *inputs;
  int num_inputs;
  Meta *outputs;
  int num_outputs;
} GraphMeta;

// One graph as reported by a model library's QnnModel_composeGraphs. This
// mirrors GraphInfo_t from QnnWrapperUtils.hpp, which is C++ only, so the C
// shim carries its own copy of the layout.
typedef struct {
  Qnn_GraphHandle_t graph;
  char *graphName;
  Qnn_Tensor_t *inputTensors;
  uint32_t numInputTensors;
  Qnn_Tensor_t *outputTensors;
  uint32_t numOutputTensors;
} ModelGraphInfo;

struct AhaQnnSession {
  void *backend_lib;
  void *system_lib;
  // Model-library path (see aha_qnn_open_model_lib). When `model_graphs` is
  // set the graph handles come from the library and graphRetrieve is skipped.
  void *model_lib;
  ModelGraphInfo **model_graphs_info;
  Qnn_GraphHandle_t *model_graphs;
  // A composed graph still has to be finalized before it can run (that is the
  // step qnn-net-run calls "Finalizing Graphs"). It is deferred to the first
  // execute because finalizing is what prepares the HTP graph, and `--info`
  // should not pay for it.
  int *model_finalized;
  uint32_t model_num_graphs;
  // QnnModel_freeGraphsInfo takes GraphInfoPtr_t** (a pointer to the array
  // pointer), not the array itself: it walks (*graphsInfo)[i] and then frees
  // the array through the same indirection.
  int32_t (*model_free_graphs)(ModelGraphInfo ***, uint32_t);
  QNN_INTERFACE_VER_TYPE qnn;
  QNN_SYSTEM_INTERFACE_VER_TYPE sys;
  Qnn_LogHandle_t log;
  Qnn_BackendHandle_t backend;
  Qnn_DeviceHandle_t device;
  Qnn_ContextHandle_t context;
  QnnSystemContext_Handle_t system_context;
  const QnnSystemContext_BinaryInfo_t *binary_info;
  GraphMeta *graphs;
  uint32_t num_graphs;
  void *binary;
  size_t binary_size;
};

static void set_err(char *err, int err_len, const char *fmt, const char *arg) {
  if (err == NULL || err_len <= 0) return;
  snprintf(err, (size_t)err_len, fmt, arg != NULL ? arg : "");
}

static void set_err_code(char *err, int err_len, const char *what,
                         Qnn_ErrorHandle_t code) {
  if (err == NULL || err_len <= 0) return;
  snprintf(err, (size_t)err_len, "%s failed (Qnn_ErrorHandle_t %lld)", what,
           (long long)code);
}

// Set AHA_QNN_TRACE=1 to see how far init got; the shim has no logging sink of
// its own and stderr is the only thing a shell user can see.
#define TRACE(...)                             \
  do {                                         \
    if (getenv("AHA_QNN_TRACE") != NULL) {     \
      fprintf(stderr, "[qnn] " __VA_ARGS__);   \
      fprintf(stderr, "\n");                   \
    }                                          \
  } while (0)

// QNN only tells you "error 1100" for most HTP failures; the backend's own log
// callback carries the sentence that actually explains it, so the shim keeps a
// handle and prints everything at AHA_QNN_LOG_LEVEL (default: warn).
static void qnn_log_callback(const char *fmt, QnnLog_Level_t level,
                             uint64_t timestamp, va_list args) {
  (void)timestamp;
  const char *tag = "?";
  switch (level) {
    case QNN_LOG_LEVEL_ERROR:
      tag = "error";
      break;
    case QNN_LOG_LEVEL_WARN:
      tag = "warn";
      break;
    case QNN_LOG_LEVEL_INFO:
      tag = "info";
      break;
    case QNN_LOG_LEVEL_VERBOSE:
      tag = "verbose";
      break;
    case QNN_LOG_LEVEL_DEBUG:
      tag = "debug";
      break;
    default:
      break;
  }
  fprintf(stderr, "[qnn %s] ", tag);
  vfprintf(stderr, fmt, args);
  fprintf(stderr, "\n");
}

// AHA_QNN_LOG=error|warn|info|verbose picks how much the backend itself says.
static QnnLog_Level_t log_level(void) {
  const char *v = getenv("AHA_QNN_LOG");
  if (v == NULL) return QNN_LOG_LEVEL_WARN;
  if (strcmp(v, "error") == 0) return QNN_LOG_LEVEL_ERROR;
  if (strcmp(v, "info") == 0) return QNN_LOG_LEVEL_INFO;
  if (strcmp(v, "verbose") == 0) return QNN_LOG_LEVEL_VERBOSE;
  if (strcmp(v, "debug") == 0) return QNN_LOG_LEVEL_DEBUG;
  return QNN_LOG_LEVEL_WARN;
}

// QNN wants an aligned binary blob; Android's malloc only guarantees 16 bytes.
static void *read_file(const char *path, size_t *size_out) {
  FILE *f = fopen(path, "rb");
  if (f == NULL) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long len = ftell(f);
  if (len <= 0) {
    fclose(f);
    return NULL;
  }
  rewind(f);
  size_t padded = ((size_t)len + 4095u) & ~(size_t)4095u;
  // Align the buffer itself: storing a back-pointer in front of the aligned
  // address writes outside the allocation whenever malloc already returned an
  // aligned block, which corrupts the heap.
  void *buf = NULL;
  if (posix_memalign(&buf, 4096u, padded) != 0) {
    fclose(f);
    return NULL;
  }
  memset(buf, 0, padded);
  size_t got = fread(buf, 1, (size_t)len, f);
  fclose(f);
  if (got != (size_t)len) {
    free(buf);
    return NULL;
  }
  *size_out = (size_t)len;
  return buf;
}

static void free_file(void *aligned) { free(aligned); }

// GraphInfo versions 1..3 share their leading members (name, input list, output
// list), and each tensor's own version says which union arm is live.
static const QnnSystemContext_GraphInfoV1_t *graph_v1(
    const QnnSystemContext_GraphInfo_t *info) {
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2:
      return (const QnnSystemContext_GraphInfoV1_t *)&info->graphInfoV2;
    case QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3:
      return (const QnnSystemContext_GraphInfoV1_t *)&info->graphInfoV3;
    default:
      return &info->graphInfoV1;
  }
}

static void read_tensor(const Qnn_Tensor_t *tensor, Meta *out) {
  memset(out, 0, sizeof(*out));
  out->desc = *tensor;
  if (tensor->version == QNN_TENSOR_VERSION_2) {
    out->name = tensor->v2.name;
    out->id = tensor->v2.id;
    out->data_type = (int32_t)tensor->v2.dataType;
    out->rank = tensor->v2.rank;
    out->dims = tensor->v2.dimensions;
  } else {
    out->name = tensor->v1.name;
    out->id = tensor->v1.id;
    out->data_type = (int32_t)tensor->v1.dataType;
    out->rank = tensor->v1.rank;
    out->dims = tensor->v1.dimensions;
  }
}

static int collect_graphs(AhaQnnSession *s, char *err, int err_len) {
  const QnnSystemContext_BinaryInfo_t *info = s->binary_info;
  // The union arms are *not* layout-compatible: V2/V3 put buildId, version
  // structs and the context tensor list before numGraphs, so the arm has to
  // match the version exactly. Every arm shares GraphInfo_t* though.
  uint32_t num_graphs = 0;
  const QnnSystemContext_GraphInfo_t *graphs = NULL;
  switch (info->version) {
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2:
      num_graphs = info->contextBinaryInfoV2.numGraphs;
      graphs = info->contextBinaryInfoV2.graphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3:
      num_graphs = info->contextBinaryInfoV3.numGraphs;
      graphs = info->contextBinaryInfoV3.graphs;
      break;
    case QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1:
      num_graphs = info->contextBinaryInfoV1.numGraphs;
      graphs = info->contextBinaryInfoV1.graphs;
      break;
    default:
      set_err(err, err_len, "unsupported context binary info version", NULL);
      return -1;
  }
  TRACE("collect: version %u, %u graph(s) at %p", (unsigned)info->version,
        num_graphs, (const void *)graphs);
  s->num_graphs = num_graphs;
  if (s->num_graphs == 0 || graphs == NULL) {
    set_err(err, err_len, "%s", "context binary reports no graphs");
    return -1;
  }
  s->graphs = (GraphMeta *)calloc(s->num_graphs, sizeof(GraphMeta));
  if (s->graphs == NULL) {
    set_err(err, err_len, "%s", "out of memory");
    return -1;
  }
  for (uint32_t g = 0; g < s->num_graphs; ++g) {
    const QnnSystemContext_GraphInfoV1_t *gi = graph_v1(&graphs[g]);
    GraphMeta *meta = &s->graphs[g];
    meta->name = gi->graphName;
    meta->num_inputs = (int)gi->numGraphInputs;
    meta->num_outputs = (int)gi->numGraphOutputs;
    meta->inputs = (Meta *)calloc((size_t)meta->num_inputs + 1, sizeof(Meta));
    meta->outputs = (Meta *)calloc((size_t)meta->num_outputs + 1, sizeof(Meta));
    if (meta->inputs == NULL || meta->outputs == NULL) {
      set_err(err, err_len, "%s", "out of memory");
      return -1;
    }
    for (int i = 0; i < meta->num_inputs; ++i) {
      read_tensor(&gi->graphInputs[i], &meta->inputs[i]);
    }
    for (int i = 0; i < meta->num_outputs; ++i) {
      read_tensor(&gi->graphOutputs[i], &meta->outputs[i]);
    }
  }
  return 0;
}

static int load_interfaces(AhaQnnSession *s, char *err, int err_len) {
  Qnn_ErrorHandle_t (*get_providers)(const QnnInterface_t ***, uint32_t *) =
      NULL;
  *(void **)(&get_providers) =
      dlsym(s->backend_lib, "QnnInterface_getProviders");
  if (get_providers == NULL) {
    set_err(err, err_len, "backend library does not export %s",
            "QnnInterface_getProviders");
    return -1;
  }
  const QnnInterface_t **providers = NULL;
  uint32_t count = 0;
  Qnn_ErrorHandle_t code = get_providers(&providers, &count);
  if (code != QNN_SUCCESS || count == 0 || providers == NULL) {
    set_err_code(err, err_len, "QnnInterface_getProviders", code);
    return -1;
  }
  s->qnn = providers[0]->QNN_INTERFACE_VER_NAME;

  Qnn_ErrorHandle_t (*get_sys_providers)(const QnnSystemInterface_t ***,
                                         uint32_t *) = NULL;
  *(void **)(&get_sys_providers) =
      dlsym(s->system_lib, "QnnSystemInterface_getProviders");
  if (get_sys_providers == NULL) {
    set_err(err, err_len, "system library does not export %s",
            "QnnSystemInterface_getProviders");
    return -1;
  }
  const QnnSystemInterface_t **sys_providers = NULL;
  uint32_t sys_count = 0;
  code = get_sys_providers(&sys_providers, &sys_count);
  if (code != QNN_SUCCESS || sys_count == 0 || sys_providers == NULL) {
    set_err_code(err, err_len, "QnnSystemInterface_getProviders", code);
    return -1;
  }
  s->sys = sys_providers[0]->QNN_SYSTEM_INTERFACE_VER_NAME;
  return 0;
}

static void trace_context_probe(AhaQnnSession *s, const char *where_);

// Compose only builds a graph; it still has to be finalized before it can run
// (that is qnn-net-run's "Finalizing Graphs" step, where the HTP prepares the
// graph). Executing an unfinalized graph crashes.
static int finalize_model_graph(AhaQnnSession *s, int graph_index,
                                const char *graph_name,
                                Qnn_GraphHandle_t graph, char *err,
                                int err_len) {
  if (s->model_finalized[graph_index]) {
    return 0;
  }
  TRACE("execute: finalizing graph[%d] %s", graph_index, graph_name);
  Qnn_ErrorHandle_t code = s->qnn.graphFinalize(graph, NULL, NULL);
  if (code != QNN_SUCCESS) {
    set_err_code(err, err_len, "graphFinalize", code);
    return -1;
  }
  s->model_finalized[graph_index] = 1;
  TRACE("execute: graph[%d] finalized", graph_index);
  return 0;
}

AhaQnnSession *aha_qnn_open(const char *backend_lib, const char *system_lib,
                            const char *context_bin, char *err, int err_len) {
  if (backend_lib == NULL || system_lib == NULL || context_bin == NULL) {
    set_err(err, err_len, "%s",
            "backend, system and context paths are required");
    return NULL;
  }
  AhaQnnSession *s = (AhaQnnSession *)calloc(1, sizeof(AhaQnnSession));
  if (s == NULL) {
    set_err(err, err_len, "%s", "out of memory");
    return NULL;
  }
  s->backend_lib = dlopen(backend_lib, RTLD_NOW | RTLD_LOCAL);
  if (s->backend_lib == NULL) {
    set_err(err, err_len, "cannot load backend library: %s", dlerror());
    free(s);
    return NULL;
  }
  s->system_lib = dlopen(system_lib, RTLD_NOW | RTLD_LOCAL);
  if (s->system_lib == NULL) {
    set_err(err, err_len, "cannot load system library: %s", dlerror());
    dlclose(s->backend_lib);
    free(s);
    return NULL;
  }
  TRACE("open: libs loaded");
  if (load_interfaces(s, err, err_len) != 0) {
    aha_qnn_close(s);
    return NULL;
  }
  TRACE("open: interfaces resolved (backend %p, system %p)",
        (void *)s->qnn.backendCreate, (void *)s->sys.systemContextCreate);
  s->binary = read_file(context_bin, &s->binary_size);
  if (s->binary == NULL) {
    set_err(err, err_len, "cannot read context binary: %s", context_bin);
    aha_qnn_close(s);
    return NULL;
  }
  TRACE("open: binary %p (%zu bytes)", s->binary, s->binary_size);

  // Names and shapes come first: they live in the binary itself, so nothing here
  // knows which model it is looking at.
  Qnn_ErrorHandle_t code = s->sys.systemContextCreate(&s->system_context);
  if (code != QNN_SUCCESS) {
    set_err_code(err, err_len, "systemContextCreate", code);
    aha_qnn_close(s);
    return NULL;
  }
  TRACE("open: systemContextCreate ok");
  Qnn_ContextBinarySize_t version = 0;
  code = s->sys.systemContextGetBinaryInfo(s->system_context, s->binary,
                                           s->binary_size, &s->binary_info,
                                           &version);
  if (code != QNN_SUCCESS) {
    set_err_code(err, err_len, "systemContextGetBinaryInfo", code);
    aha_qnn_close(s);
    return NULL;
  }
  TRACE("open: binary info version %lu", (unsigned long)version);
  if (collect_graphs(s, err, err_len) != 0) {
    aha_qnn_close(s);
    return NULL;
  }
  TRACE("open: %u graph(s)", s->num_graphs);

  code = s->qnn.logCreate(qnn_log_callback, log_level(), &s->log);
  if (code != QNN_SUCCESS) {
    TRACE("open: logCreate failed (%lu), continuing without logs",
          (unsigned long)code);
    s->log = NULL;
  }
  code = s->qnn.backendCreate(s->log, NULL, &s->backend);
  if (code != QNN_SUCCESS) {
    set_err_code(err, err_len, "backendCreate", code);
    aha_qnn_close(s);
    return NULL;
  }
  TRACE("open: backendCreate ok");
  // A device handle is optional in QNN; some HTP versions reject deviceCreate
  // (we see QNN_DEVICE_ERROR_ASSOCIATED_TO_CONTEXT) and are happy with a NULL
  // device in contextCreateFromBinary, which is what sherpa-onnx does too.
  code = s->qnn.deviceCreate(s->log, NULL, &s->device);
  if (code != QNN_SUCCESS) {
    TRACE("open: deviceCreate failed (%lu), continuing without a device",
          (unsigned long)code);
    s->device = NULL;
  } else {
    TRACE("open: deviceCreate ok");
  }
  code = s->qnn.contextCreateFromBinary(s->backend, s->device, NULL, s->binary,
                                        s->binary_size, &s->context, NULL);
  if (code != QNN_SUCCESS) {
    set_err_code(err, err_len, "contextCreateFromBinary", code);
    aha_qnn_close(s);
    return NULL;
  }
  TRACE("open: contextCreateFromBinary ok (%p)", (void *)s->context);
  trace_context_probe(s, "open");
  if (err != NULL && err_len > 0) err[0] = '\0';
  return s;
}

// Diagnostic: only a real context can report its binary size. If this answers
// with an invalid-handle error the handle we hold is a placeholder, and the
// HTP then refuses to DMA client buffers ("Dma execution failed on the skel
// side", 1100), which is what the SoVITS graphs hit.
static void trace_context_probe(AhaQnnSession *s, const char *where_) {
  if (s == NULL || s->context == NULL ||
      s->qnn.contextGetBinarySize == NULL) {
    return;
  }
  uint64_t size = 0;
  Qnn_ErrorHandle_t code = s->qnn.contextGetBinarySize(s->context, &size);
  TRACE("%s: contextGetBinarySize -> %lu (%llu bytes)", where_,
        (unsigned long)code, (unsigned long long)size);
}

// Signature of the two C entry points a generated model library exports.
typedef int32_t (*ComposeGraphsFn)(Qnn_BackendHandle_t,
                                   QNN_INTERFACE_VER_TYPE,
                                   Qnn_ContextHandle_t,
                                   const void *,
                                   uint32_t,
                                   ModelGraphInfo ***,
                                   uint32_t *,
                                   int,
                                   QnnLog_Callback_t,
                                   QnnLog_Level_t);
typedef int32_t (*FreeGraphsFn)(ModelGraphInfo ***, uint32_t);

// Opens a compiled model library (lib<name>.so produced by
// qnn-model-lib-generator) instead of a context binary. The library composes
// the graphs itself and hands back ready-made tensor descriptors, which is the
// path qnn-net-run uses when it is given `--model`.
//
// Why the shim needs both: HTP v81 rejects graphExecute on a context binary
// whose IO is float32 and whose activations are fixed point ("Dma execution
// failed on the skel side"), while the same graph composed from the model
// library executes fine. For such models the library is the working path.
AhaQnnSession *aha_qnn_open_model_lib(const char *backend_lib,
                                     const char *model_lib, char *err,
                                     int err_len) {
  if (backend_lib == NULL || model_lib == NULL) {
    set_err(err, err_len, "%s", "backend library and model library are required");
    return NULL;
  }
  AhaQnnSession *s = (AhaQnnSession *)calloc(1, sizeof(AhaQnnSession));
  if (s == NULL) {
    set_err(err, err_len, "%s", "out of memory");
    return NULL;
  }
  s->backend_lib = dlopen(backend_lib, RTLD_NOW | RTLD_LOCAL);
  if (s->backend_lib == NULL) {
    set_err(err, err_len, "cannot load backend library: %s", dlerror());
    free(s);
    return NULL;
  }
  Qnn_ErrorHandle_t (*get_providers)(const QnnInterface_t ***, uint32_t *) =
      NULL;
  *(void **)(&get_providers) =
      dlsym(s->backend_lib, "QnnInterface_getProviders");
  if (get_providers == NULL) {
    set_err(err, err_len, "backend library does not export %s",
            "QnnInterface_getProviders");
    aha_qnn_close(s);
    return NULL;
  }
  const QnnInterface_t **providers = NULL;
  uint32_t count = 0;
  Qnn_ErrorHandle_t code = get_providers(&providers, &count);
  if (code != QNN_SUCCESS || count == 0 || providers == NULL) {
    set_err_code(err, err_len, "QnnInterface_getProviders", code);
    aha_qnn_close(s);
    return NULL;
  }
  s->qnn = providers[0]->QNN_INTERFACE_VER_NAME;
  TRACE("model: backend interface resolved");

  code = s->qnn.logCreate(qnn_log_callback, log_level(), &s->log);
  if (code != QNN_SUCCESS) {
    TRACE("model: logCreate failed (%lu), continuing without logs",
          (unsigned long)code);
    s->log = NULL;
  }
  code = s->qnn.backendCreate(s->log, NULL, &s->backend);
  if (code != QNN_SUCCESS) {
    set_err_code(err, err_len, "backendCreate", code);
    aha_qnn_close(s);
    return NULL;
  }
  // HTP does not need a device handle: qnn-net-run creates the context with
  // `device = nullptr`. Handing it the sentinel handle deviceCreate returns
  // here (0x1) makes the DSP's skel side reject client buffers with "Dma
  // execution failed on the skel side" (1100). AHA_QNN_DEVICE=1 restores the
  // old behaviour for comparison.
  s->device = NULL;
  const char *dev_env = getenv("AHA_QNN_DEVICE");
  if (dev_env != NULL && strcmp(dev_env, "1") == 0) {
    code = s->qnn.deviceCreate(s->log, NULL, &s->device);
    if (code != QNN_SUCCESS) {
      TRACE("model: deviceCreate failed (%lu), continuing without a device",
            (unsigned long)code);
      s->device = NULL;
    }
    TRACE("model: device %p (AHA_QNN_DEVICE=1)", (void *)s->device);
  }
  code = s->qnn.contextCreate(s->backend, s->device, NULL, &s->context);
  if (code != QNN_SUCCESS) {
    set_err_code(err, err_len, "contextCreate", code);
    aha_qnn_close(s);
    return NULL;
  }
  TRACE("model: contextCreate ok (%p)", (void *)s->context);
  trace_context_probe(s, "model");

  s->model_lib = dlopen(model_lib, RTLD_NOW | RTLD_LOCAL);
  if (s->model_lib == NULL) {
    set_err(err, err_len, "cannot load model library: %s", dlerror());
    aha_qnn_close(s);
    return NULL;
  }
  // The generated library exports these two C entry points; the compose call
  // takes the interface struct by value, so the signature has to spell it out.
  ComposeGraphsFn compose = NULL;
  *(void **)(&compose) = dlsym(s->model_lib, "QnnModel_composeGraphs");
  *(void **)(&s->model_free_graphs) =
      dlsym(s->model_lib, "QnnModel_freeGraphsInfo");
  if (compose == NULL) {
    set_err(err, err_len, "model library does not export %s",
            "QnnModel_composeGraphs");
    aha_qnn_close(s);
    return NULL;
  }
  code = compose(s->backend, s->qnn, s->context, NULL, 0,
                 &s->model_graphs_info, &s->model_num_graphs, 0,
                 qnn_log_callback, log_level());
  if (code != 0 || s->model_graphs_info == NULL || s->model_num_graphs == 0) {
    set_err(err, err_len, "QnnModel_composeGraphs failed", NULL);
    aha_qnn_close(s);
    return NULL;
  }
  TRACE("model: composed %u graph(s)", s->model_num_graphs);

  s->num_graphs = s->model_num_graphs;
  s->graphs = (GraphMeta *)calloc(s->num_graphs, sizeof(GraphMeta));
  s->model_graphs =
      (Qnn_GraphHandle_t *)calloc(s->num_graphs, sizeof(Qnn_GraphHandle_t));
  s->model_finalized = (int *)calloc(s->num_graphs, sizeof(int));
  if (s->graphs == NULL || s->model_graphs == NULL ||
      s->model_finalized == NULL) {
    set_err(err, err_len, "%s", "out of memory");
    aha_qnn_close(s);
    return NULL;
  }
  for (uint32_t g = 0; g < s->num_graphs; ++g) {
    ModelGraphInfo *gi = s->model_graphs_info[g];
    GraphMeta *meta = &s->graphs[g];
    s->model_graphs[g] = gi->graph;
    meta->name = gi->graphName;
    meta->num_inputs = (int)gi->numInputTensors;
    meta->num_outputs = (int)gi->numOutputTensors;
    meta->inputs = (Meta *)calloc((size_t)meta->num_inputs + 1, sizeof(Meta));
    meta->outputs = (Meta *)calloc((size_t)meta->num_outputs + 1, sizeof(Meta));
    if (meta->inputs == NULL || meta->outputs == NULL) {
      set_err(err, err_len, "%s", "out of memory");
      aha_qnn_close(s);
      return NULL;
    }
    for (int i = 0; i < meta->num_inputs; ++i) {
      read_tensor(&gi->inputTensors[i], &meta->inputs[i]);
    }
    for (int i = 0; i < meta->num_outputs; ++i) {
      read_tensor(&gi->outputTensors[i], &meta->outputs[i]);
    }
  }
  if (err != NULL && err_len > 0) err[0] = '\0';
  return s;
}

void aha_qnn_close(AhaQnnSession *s) {
  if (s == NULL) return;
  // Model-library graphs belong to the library and have to be handed back
  // before the context goes away. The entry point takes the address of the
  // array pointer (it walks (*graphsInfo)[i] and nulls it out afterwards).
  if (s->model_graphs_info != NULL && s->model_free_graphs != NULL) {
    TRACE("close: freeing %u model graph(s)", s->model_num_graphs);
    s->model_free_graphs(&s->model_graphs_info, s->model_num_graphs);
    s->model_graphs_info = NULL;
    TRACE("close: model graphs freed");
  }
  free(s->model_graphs);
  s->model_graphs = NULL;
  free(s->model_finalized);
  s->model_finalized = NULL;
  if (s->context != NULL && s->qnn.contextFree != NULL) {
    TRACE("close: contextFree %p", (void *)s->context);
    s->qnn.contextFree(s->context, NULL);
    s->context = NULL;
    TRACE("close: context freed");
  }
  if (s->device != NULL && s->qnn.deviceFree != NULL) {
    TRACE("close: deviceFree %p", (void *)s->device);
    s->qnn.deviceFree(s->device);
    s->device = NULL;
    TRACE("close: device freed");
  }
  if (s->backend != NULL && s->qnn.backendFree != NULL) {
    TRACE("close: backendFree %p", (void *)s->backend);
    s->qnn.backendFree(s->backend);
    s->backend = NULL;
    TRACE("close: backend freed");
  }
  if (s->system_context != NULL && s->sys.systemContextFree != NULL) {
    TRACE("close: systemContextFree %p", (void *)s->system_context);
    s->sys.systemContextFree(s->system_context);
    s->system_context = NULL;
    TRACE("close: system context freed");
  }
  if (s->graphs != NULL) {
    for (uint32_t g = 0; g < s->num_graphs; ++g) {
      free(s->graphs[g].inputs);
      free(s->graphs[g].outputs);
    }
    free(s->graphs);
    TRACE("close: graph metadata freed");
  }
  free_file(s->binary);
  // The QNN runtime registers static destructors / atexit handlers inside its
  // own libraries; unloading them from under the loader leaves those callbacks
  // pointing into unmapped memory, and the process dies at exit with a jump to
  // a data address (single "unknown" frame). qnn-net-run never unloads them
  // either. Leaking three library handles is the cheap, correct trade.
  if (getenv("AHA_QNN_DLCLOSE") != NULL) {
    if (s->backend_lib != NULL) dlclose(s->backend_lib);
    if (s->system_lib != NULL) dlclose(s->system_lib);
    if (s->model_lib != NULL) dlclose(s->model_lib);
  }
  free(s);
  TRACE("close: done");
}

int aha_qnn_graph_count(AhaQnnSession *s) {
  return (s == NULL) ? -1 : (int)s->num_graphs;
}

const char *aha_qnn_graph_name(AhaQnnSession *s, int graph_index) {
  if (s == NULL || graph_index < 0 || (uint32_t)graph_index >= s->num_graphs) {
    return NULL;
  }
  return s->graphs[graph_index].name;
}

static const Meta *meta_at(AhaQnnSession *s, int graph_index, int is_input,
                           int tensor_index) {
  if (s == NULL || graph_index < 0 || (uint32_t)graph_index >= s->num_graphs) {
    return NULL;
  }
  const GraphMeta *g = &s->graphs[graph_index];
  const Meta *list = is_input ? g->inputs : g->outputs;
  int count = is_input ? g->num_inputs : g->num_outputs;
  if (tensor_index < 0 || tensor_index >= count) return NULL;
  return &list[tensor_index];
}

int aha_qnn_tensor_count(AhaQnnSession *s, int graph_index, int is_input) {
  if (s == NULL || graph_index < 0 || (uint32_t)graph_index >= s->num_graphs) {
    return -1;
  }
  const GraphMeta *g = &s->graphs[graph_index];
  return is_input ? g->num_inputs : g->num_outputs;
}

int aha_qnn_tensor_desc(AhaQnnSession *s, int graph_index, int is_input,
                        int tensor_index, AhaQnnTensor *out) {
  if (out == NULL) return -1;
  const Meta *meta = meta_at(s, graph_index, is_input, tensor_index);
  if (meta == NULL) return -1;
  memset(out, 0, sizeof(*out));
  if (meta->name != NULL) {
    snprintf(out->name, sizeof(out->name), "%s", meta->name);
  }
  out->id = (int32_t)meta->id;
  out->data_type = meta->data_type;
  out->rank = (int32_t)meta->rank;
  int64_t elements = 1;
  for (uint32_t i = 0; i < meta->rank && i < AHA_QNN_MAX_DIMS; ++i) {
    out->dims[i] = (int32_t)meta->dims[i];
    elements *= (int64_t)meta->dims[i];
  }
  out->elements = elements;
  return 0;
}

// Bytes per element for the QNN types a converted graph can carry. The raw
// Qnn_DataType_t value travels in the descriptor so the Rust side stays free of
// QNN's enums.
static size_t storage_bytes(int32_t data_type) {
  switch ((Qnn_DataType_t)data_type) {
    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_INT_8:
    case QNN_DATATYPE_SFIXED_POINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
    case QNN_DATATYPE_BOOL_8:
      return 1;
    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_INT_16:
    case QNN_DATATYPE_FLOAT_16:
    case QNN_DATATYPE_SFIXED_POINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      return 2;
    case QNN_DATATYPE_FLOAT_64:
    case QNN_DATATYPE_UINT_64:
    case QNN_DATATYPE_INT_64:
      return 8;
    default:
      return 4;
  }
}

size_t aha_qnn_bytes_per_element(int32_t data_type) {
  return storage_bytes(data_type);
}

// The HTP ships tensors over DMA, and its skel side refuses host buffers that
// are not page aligned ("Dma execution failed on the skel side"). Callers hand
// us ordinary heap pointers, so every tensor goes through a page-aligned
// staging buffer that is copied in before the call and copied back after it.
static void *page_buffer(size_t bytes) {
  void *buf = NULL;
  if (bytes == 0) {
    bytes = 1;
  }
  if (posix_memalign(&buf, 4096u, bytes) != 0) {
    return NULL;
  }
  memset(buf, 0, bytes);
  return buf;
}

// DMA-BUF heaps are how Android hands out memory the DSP can read without a
// bounce buffer. HTP only needs this for graphs whose IO is float: the skel
// side maps plain heap buffers for quantized tensors but fails float ones with
// "Dma execution failed on the skel side" (error 1100).
#ifndef DMA_HEAP_IOC_MAGIC
#define DMA_HEAP_IOC_MAGIC 'H'
struct aha_dma_heap_allocation_data {
  uint64_t len;
  uint32_t fd;
  uint32_t fd_flags;
  uint64_t heap_flags;
};
#define AHA_DMA_HEAP_IOCTL_ALLOC \
  _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct aha_dma_heap_allocation_data)
#endif

static int dma_alloc(size_t bytes, void **data_out) {
  if (bytes == 0) {
    bytes = 4096;
  }
  size_t len = (bytes + 4095u) & ~(size_t)4095u;
  static const char *heaps[] = {"/dev/dma_heap/system",
                                "/dev/dma_heap/system-uncached",
                                "/dev/ion"};
  for (size_t i = 0; i < sizeof(heaps) / sizeof(heaps[0]); ++i) {
    int heap = open(heaps[i], O_RDWR | O_CLOEXEC);
    if (heap < 0) {
      // Shell users can only read the heap nodes; the ioctl still works.
      heap = open(heaps[i], O_RDONLY | O_CLOEXEC);
    }
    if (heap < 0) {
      TRACE("dma_alloc: open %s failed (errno %d)", heaps[i], errno);
      continue;
    }
    struct aha_dma_heap_allocation_data req;
    memset(&req, 0, sizeof(req));
    req.len = len;
    req.fd_flags = O_RDWR | O_CLOEXEC;
    if (ioctl(heap, AHA_DMA_HEAP_IOCTL_ALLOC, &req) != 0 || req.fd == 0) {
      req.fd_flags = O_RDONLY | O_CLOEXEC;
      if (ioctl(heap, AHA_DMA_HEAP_IOCTL_ALLOC, &req) != 0 || req.fd == 0) {
        TRACE("dma_alloc: ioctl %s failed (errno %d)", heaps[i], errno);
        close(heap);
        continue;
      }
    }
    close(heap);
    void *map = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, (int)req.fd, 0);
    if (map == MAP_FAILED) {
      map = mmap(NULL, len, PROT_READ, MAP_SHARED, (int)req.fd, 0);
    }
    if (map == MAP_FAILED) {
      TRACE("dma_alloc: mmap %s failed (errno %d)", heaps[i], errno);
      close((int)req.fd);
      continue;
    }
    memset(map, 0, len);
    *data_out = map;
    TRACE("dma_alloc: %zu bytes from %s fd=%u", len, heaps[i],
          (unsigned)req.fd);
    return (int)req.fd;
  }
  return -1;
}

// One staging buffer per tensor. DMA-BUF wins when the heap is available
// (float IO needs it); otherwise the page-aligned heap buffer keeps the
// quantized path working, which is what most converted models use.
typedef struct {
  void *data;
  size_t bytes;
  size_t mapped;
  int fd;
  Qnn_MemHandle_t handle;
  int registered;
  // 0 for a buffer that belongs to the model library: we copy into it but must
  // not free it.
  int owned;
} Staging;

static int staging_open(Staging *st, size_t bytes, int want_dma) {
  memset(st, 0, sizeof(*st));
  st->bytes = bytes;
  st->fd = -1;
  st->owned = 1;
  if (want_dma) {
    st->fd = dma_alloc(bytes, &st->data);
    if (st->fd >= 0) {
      st->mapped = (bytes + 4095u) & ~(size_t)4095u;
      return 0;
    }
  }
  st->data = page_buffer(bytes);
  return st->data == NULL ? -1 : 0;
}

static void staging_close(Staging *st, QNN_INTERFACE_VER_TYPE *qnn) {
  if (st->registered && qnn != NULL && qnn->memDeRegister != NULL) {
    qnn->memDeRegister(&st->handle, 1);
  }
  if (!st->owned) {
    memset(st, 0, sizeof(*st));
    st->fd = -1;
    return;
  }
  if (st->fd >= 0) {
    if (st->data != NULL) {
      munmap(st->data, st->mapped);
    }
    close(st->fd);
  } else {
    free(st->data);
  }
  memset(st, 0, sizeof(*st));
  st->fd = -1;
}

// `clientBuf` of a descriptor, whichever struct version it uses.
static void tensor_client_buf(const Qnn_Tensor_t *t, void **data,
                              uint32_t *size) {
  *data = NULL;
  *size = 0;
  if (t->version == QNN_TENSOR_VERSION_2) {
    *data = t->v2.clientBuf.data;
    *size = t->v2.clientBuf.dataSize;
  } else {
    *data = t->v1.clientBuf.data;
    *size = t->v1.clientBuf.dataSize;
  }
}

int aha_qnn_execute(AhaQnnSession *s, int graph_index,
                    const AhaQnnTensor *inputs, int input_count,
                    const AhaQnnTensor *outputs, int output_count, char *err,
                    int err_len) {
  if (s == NULL || inputs == NULL || outputs == NULL) {
    set_err(err, err_len, "%s", "session and tensors are required");
    return -1;
  }
  const char *graph_name = aha_qnn_graph_name(s, graph_index);
  if (graph_name == NULL) {
    set_err(err, err_len, "%s", "unknown graph index");
    return -1;
  }
  Qnn_GraphHandle_t graph = NULL;
  Qnn_ErrorHandle_t code = QNN_SUCCESS;
  // Finalize before or after the IO tensors are set up. qnn-net-run finalizes
  // first ("Finalizing Graphs", then the pointers), and both orders behave the
  // same for us, so the default follows net-run; AHA_QNN_FINALIZE=late flips it
  // for experiments.
  const char *fin_env = getenv("AHA_QNN_FINALIZE");
  const int finalize_early =
      !(fin_env != NULL && strcmp(fin_env, "late") == 0);
  if (s->model_graphs != NULL) {
    // Composed from a model library: the handle is already usable.
    graph = s->model_graphs[graph_index];
    if (finalize_early &&
        finalize_model_graph(s, graph_index, graph_name, graph, err,
                             err_len) != 0) {
      return -1;
    }
  } else {
    code = s->qnn.graphRetrieve(s->context, graph_name, &graph);
    if (code != QNN_SUCCESS) {
      set_err_code(err, err_len, "graphRetrieve", code);
      return -1;
    }
  }

  int rc = -1;
  const GraphMeta *gm = &s->graphs[graph_index];
  if (input_count != gm->num_inputs || output_count != gm->num_outputs) {
    set_err(err, err_len, "%s", "tensor count does not match the graph");
    return -1;
  }
  // DMA-BUF staging is opt-in: HTP v81 rejects QNN_MEM_TYPE_DMA_BUF
  // ("Unsupported Platform Mem Type 3 on arch 81"), so turning it on by
  // default would only add one failed registration per tensor on every run.
  const char *dmabuf = getenv("AHA_QNN_DMABUF");
  const int want_dma = dmabuf != NULL && strcmp(dmabuf, "0") != 0;
  // Registering the IO tensors through the context (instead of handing the
  // binary's own descriptor straight to graphExecute) is what QNN documents
  // for graphs that came out of a context binary; AHA_QNN_CTX_TENSOR=0 turns
  // it off for comparison.
  const char *ct_env = getenv("AHA_QNN_CTX_TENSOR");
  const int want_ctx_tensor = ct_env == NULL || strcmp(ct_env, "0") != 0;
  int total = input_count + output_count;
  Qnn_Tensor_t *tensors =
      (Qnn_Tensor_t *)calloc((size_t)total, sizeof(Qnn_Tensor_t));
  Staging *stags = (Staging *)calloc((size_t)total, sizeof(Staging));
  if (tensors == NULL || stags == NULL) {
    set_err(err, err_len, "%s", "out of memory");
    goto done;
  }
  for (int i = 0; i < total; ++i) {
    int is_input = i < input_count;
    const Meta *meta = is_input ? &gm->inputs[i] : &gm->outputs[i - input_count];
    const AhaQnnTensor *desc = is_input ? &inputs[i] : &outputs[i - input_count];
    Qnn_Tensor_t *t = &tensors[i];
    *t = meta->desc;
    // The system context binary only describes tensors well enough to name
    // them: `clientBuf` carries no size, so the buffer size has to come from
    // dims × element width, exactly as the graph expects it.
    uint32_t bytes =
        (uint32_t)((size_t)desc->elements * storage_bytes(meta->data_type));
    if (desc->data == NULL) {
      set_err(err, err_len, "%s", "tensor data pointer is NULL");
      goto done;
    }
    Staging *st = &stags[i];
    // A model library owns its IO buffers: `clientBuf` comes from the library
    // and those addresses are what the DSP already has mapped. Handing
    // graphExecute a buffer of our own instead fails with "Dma execution failed
    // on the skel side" (1100) although the same graph runs through
    // qnn-net-run, whose own model path (sherpa's SetupPointers) also keeps the
    // library's pointers and only copies the payload in. So borrow it.
    const int from_model = s->model_graphs != NULL;
    int borrowed = 0;
    if (from_model) {
      void *lib_data = NULL;
      uint32_t lib_size = 0;
      tensor_client_buf(t, &lib_data, &lib_size);
      if (lib_data != NULL && lib_size >= bytes) {
        memset(st, 0, sizeof(*st));
        st->fd = -1;
        st->data = lib_data;
        st->bytes = lib_size;
        borrowed = 1;
      } else {
        TRACE("execute: %s clientBuf is %p / %u bytes, staging our own",
              meta->name, lib_data, (unsigned)lib_size);
      }
    }
    if (!borrowed && staging_open(st, (size_t)bytes, want_dma) != 0) {
      set_err(err, err_len, "%s", "out of memory");
      goto done;
    }
    if (is_input) {
      memcpy(st->data, desc->data, (size_t)bytes);
    }
    // A graph that came out of a context binary is already finalized, so its
    // tensors cannot be registered with tensorCreateGraphTensor. The context
    // level call is the one that accepts them; QNN only honours APP_* tensors
    // there if the buffer pointers are NULL at creation time, so the data
    // pointer is attached afterwards.
    if (!from_model && want_ctx_tensor && s->qnn.tensorCreateContextTensor != NULL) {
      Qnn_Tensor_t ct;
      memset(&ct, 0, sizeof(ct));
      ct.version = QNN_TENSOR_VERSION_2;
      ct.v2.id = meta->id;
      ct.v2.name = (char *)meta->name;
      ct.v2.type = is_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_APP_READ;
      ct.v2.dataFormat = QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER;
      ct.v2.dataType = (Qnn_DataType_t)meta->data_type;
      ct.v2.quantizeParams = meta->desc.v1.quantizeParams;
      ct.v2.rank = meta->rank;
      ct.v2.dimensions = (uint32_t *)meta->dims;
      ct.v2.memType = QNN_TENSORMEMTYPE_RAW;
      ct.v2.clientBuf.data = NULL;
      ct.v2.clientBuf.dataSize = 0;
      code = s->qnn.tensorCreateContextTensor(s->context, &ct);
      if (code == QNN_SUCCESS) {
        *t = ct;
      } else {
        TRACE("execute: tensorCreateContextTensor(%s) failed (%lu), using the "
              "binary descriptor",
              meta->name, (unsigned long)code);
      }
    }
    // A model library hands out complete, finalised descriptors: type, id,
    // name, dims and quantize params are all set by the converter, and
    // qnn-net-run / sherpa pass them through untouched, only pointing
    // `clientBuf` at memory. Re-deriving the fields from our own metadata is
    // what the DSP rejected ("Dma execution failed on the skel side"), so for
    // the model path nothing but the buffer is written.
    if (!from_model) {
      if (t->version == QNN_TENSOR_VERSION_2) {
        t->v2.type = is_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_APP_READ;
        t->v2.id = meta->id;
        t->v2.name = (char *)meta->name;
        t->v2.dataType = (Qnn_DataType_t)meta->data_type;
        t->v2.rank = meta->rank;
        t->v2.dimensions = (uint32_t *)meta->dims;
      } else {
        t->v1.type = is_input ? QNN_TENSOR_TYPE_APP_WRITE : QNN_TENSOR_TYPE_APP_READ;
        t->v1.id = meta->id;
        t->v1.name = (char *)meta->name;
        t->v1.dataType = (Qnn_DataType_t)meta->data_type;
        t->v1.rank = meta->rank;
        t->v1.dimensions = (uint32_t *)meta->dims;
      }
    }

    int via_handle = 0;
    if (st->fd >= 0 && s->qnn.memRegister != NULL) {
      // qnn-net-run hands Android memory to the DSP as ION buffers (its binary
      // carries ion_alloc / dma_buf strings). ION is the one HTP here accepts:
      // QNN_MEM_TYPE_DMA_BUF comes back as QNN_MEM_ERROR_INVALID_HANDLE (8001),
      // so try ION first and only fall back to DMA-BUF when asked to.
      // AHA_QNN_MEMTYPE=ion|dma-buf pins one of them for experiments.
      const char *memtype = getenv("AHA_QNN_MEMTYPE");
      int types[2];
      int ntypes = 0;
      if (memtype != NULL && strcmp(memtype, "dma-buf") == 0) {
        types[ntypes++] = QNN_MEM_TYPE_DMA_BUF;
      } else if (memtype != NULL && strcmp(memtype, "ion") == 0) {
        types[ntypes++] = QNN_MEM_TYPE_ION;
      } else {
        types[ntypes++] = QNN_MEM_TYPE_ION;
        types[ntypes++] = QNN_MEM_TYPE_DMA_BUF;
      }
      for (int k = 0; k < ntypes && !via_handle; ++k) {
        Qnn_MemDescriptor_t md;
        memset(&md, 0, sizeof(md));
        md.memShape.numDim = meta->rank;
        md.memShape.dimSize = (uint32_t *)meta->dims;
        md.memShape.shapeConfig = NULL;
        md.dataType = (Qnn_DataType_t)meta->data_type;
        md.memType = (Qnn_MemType_t)types[k];
        if (types[k] == QNN_MEM_TYPE_ION) {
          md.ionInfo.fd = st->fd;
        } else {
          md.dmaBufInfo.fd = st->fd;
          md.dmaBufInfo.data = st->data;
        }
        code = s->qnn.memRegister(s->context, &md, 1, &st->handle);
        if (code == QNN_SUCCESS) {
          st->registered = 1;
          via_handle = 1;
          if (t->version == QNN_TENSOR_VERSION_2) {
            t->v2.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            t->v2.memHandle = st->handle;
          } else {
            t->v1.memType = QNN_TENSORMEMTYPE_MEMHANDLE;
            t->v1.memHandle = st->handle;
          }
          TRACE("execute: %s registered as a memhandle (%s)", meta->name,
                types[k] == QNN_MEM_TYPE_ION ? "ion" : "dma-buf");
        } else {
          TRACE("execute: memRegister(%s, %s) failed (%lu)", meta->name,
                types[k] == QNN_MEM_TYPE_ION ? "ion" : "dma-buf",
                (unsigned long)code);
        }
      }
      if (!via_handle) {
        TRACE("execute: %s falls back to a plain buffer", meta->name);
      }
    }
    if (!via_handle && !borrowed) {
      if (t->version == QNN_TENSOR_VERSION_2) {
        t->v2.memType = QNN_TENSORMEMTYPE_RAW;
        t->v2.clientBuf.data = st->data;
        t->v2.clientBuf.dataSize = bytes;
      } else {
        t->v1.memType = QNN_TENSORMEMTYPE_RAW;
        t->v1.clientBuf.data = st->data;
        t->v1.clientBuf.dataSize = bytes;
      }
    }
    TRACE("execute: %s %s id=%u dtype=%d rank=%u v=%u ttype=%d buf=%p "
          "size=%u dma=%d lib=%d",
          is_input ? "in" : "out", meta->name, (unsigned)meta->id,
          meta->data_type, meta->rank, (unsigned)t->version, (int)t->v1.type,
          st->data, (unsigned)bytes, st->fd >= 0, borrowed);
  }

  if (!finalize_early && s->model_graphs != NULL &&
      finalize_model_graph(s, graph_index, graph_name, graph, err, err_len) !=
          0) {
    goto done;
  }

  code = s->qnn.graphExecute(graph, tensors, (uint32_t)input_count,
                             tensors + input_count, (uint32_t)output_count, NULL,
                             NULL);
  if (code != QNN_SUCCESS) {
    set_err_code(err, err_len, "graphExecute", code);
    goto done;
  }
  for (int i = input_count; i < total; ++i) {
    const AhaQnnTensor *desc = &outputs[i - input_count];
    uint32_t bytes = (uint32_t)((size_t)desc->elements *
                                storage_bytes(gm->outputs[i - input_count].data_type));
    memcpy(desc->data, stags[i].data, (size_t)bytes);
  }
  rc = 0;

done:
  if (stags != NULL) {
    for (int i = 0; i < total; ++i) {
      staging_close(&stags[i], &s->qnn);
    }
    free(stags);
  }
  free(tensors);
  return rc;
}

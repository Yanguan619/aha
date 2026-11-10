# TurboVLA（aha / candle）性能优化计划

> 这份文件是**工作记录 + 计划**，不属于 aha 上游文档（`docs/*.md`）。
> 相关背景与结论同步写在仓库外的 `doc/VLA-AHA.md`（desktop-pet 主仓）§5。

---

## 0. 当前状态（2026-11-10）

| 阶段 | GPU bf16 中位 | 相对起点 |
|---|---|---|
| 原始（f32、CUDA 直接崩） | 48.6 ms | — |
| 修 bug + 打通 bf16 | 29.9 ms | −38% |
| 两视角合批 + 文本缓存 + RoPE 融合 | 21.7 ms | −55% |
| CUDA graph | 14.7 ms | −70% |
| bias epilogue（cublasLt） | 12.1 ms | −75% |
| 注意力去掉 f32 往返 | 11.1 ms | −77% |
| 拆头/合头专用 kernel | **10.75 ms** | **−78%** |

对照：同机 Python（torch 2.9.1+cu128，CUDA graph）**8.2 ms**。

## 1. 当前 kernel 级画像（nsys，10.75ms/帧）

| kernel | 每帧 | 备注 |
|---|---|---|
| cutlass GEMM（5 种 tile） | **4.76 ms** | 与 torch 同形状持平（`examples/gemm_probe`），无优化空间 |
| `ucopy_bf16` | 1.18 ms | candle 通用 copy，**≈47 GB/s** |
| `bmul_bf16` | 0.63 ms | |
| `ugelu_erf_bf16` | 0.39 ms | |
| `layernorm_bf16` | 0.38 ms | 已是融合 kernel |
| `badd_bf16` | 0.29 ms | 融合 bias 后剩下的残差加法 |
| `softmax_bf16` | 0.28 ms | |
| `affine_bf16` | 0.26 ms | |
| `cast_f32_bf16` | 0.19 ms | |
| `vla_split_heads`（我们的） | **0.13 ms** | 同样搬 bf16，**≈400 GB/s** |

**结论**：GEMM 已经追平 torch，剩下的差距全在 **elementwise kernel 的访存效率**
（candle 50–100 GB/s，本机可跑 300+ GB/s）。

---

## 2. 已完成

### 2.1 A：bias epilogue 融合（`src/models/turbovla/linear.rs`）

`candle_nn::Linear` = `matmul` + `broadcast_add(bias)` 两个 kernel（每帧 270 次 `badd`，2.98ms）。
改成自己实现 `Linear`（同签名、同 `Module` 实现），bf16 + CUDA 走 cuBLASLt 的
`CUBLASLT_EPILOGUE_BIAS`，其余情况原样回退。

- 关闭开关：`TURBOVLA_NO_FUSED_LINEAR=1`
- 数值：每个 GEMM 差 1 个 bf16 ulp（`examples/weight_probe`，用真实权重）
- **f32 一律走回退路径**：cuBLASLt 对 fp32 默认开 TF32（1e-3），会污染验证

### 2.2 B-1：注意力不做 f32 往返（`src/models/turbovla/vision.rs`）

candle 的 bf16 softmax kernel 内部本来就是 f32 累加；原来的
`to_dtype(F32) → softmax → to_dtype(BF16)` 纯搬运（1.17ms）。
关闭开关：`TURBOVLA_SOFTMAX_F32=1`。

### 2.3 B-2：拆头/合头专用 kernel（`src/models/turbovla/heads_cuda.rs`）

`reshape+transpose+contiguous` 走 candle 通用 `ucopy`（47GB/s）。实际上这两步是
「按行块搬连续内存」（`out[b][h][s][:]] = in[b][s][h][:]`，每次 `D` 个连续元素），
一个纯拷贝 kernel 即可，实测 400GB/s，0.8ms → 0.13ms。

- 运行时用 NVRTC 编译（不引入 build.rs / cudaforge，加 kernel 不用重编 crate）
- 逐位一致：`examples/heads_probe` diff `0.000e0`
- 踩过的坑：`(B,S,H,D)` 线性下标里 `d` 之后是 `h` 不是 `s`；`is_contiguous()`
  为真也可能带 `start_offset`（narrow 视图），要 `slice(layout.start_offset()..)`

---

## 3. P0：candle elementwise kernel 向量化（→ 上游 PR）

### 3.1 证据

`candle-kernels/src/unary.cu` 与 `binary_op_macros.cuh` 的连续分支都是：

```cuda
for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
    TYPENAME x = inp ? inp[i] : out[i];
    out[i] = FUNC;
}
```

**一个元素一个线程、16 位类型只有 2 字节访存、没有向量化**。对照组：我们手写的
coalesced kernel 在同一台机器上跑 400 GB/s（0.13ms vs 0.8ms，同一批数据）。

torch 的实现是 `vectorized_elementwise_kernel<4, ...>`，用 16 字节向量读写。

### 3.2 改法（逐位等价，只改访存宽度）

`unary.cu` 的连续分支加一条快路径（其余分支不动，降低风险）：

```cuda
// 16 字节一组：f32 4 个，f16/bf16 8 个
template <typename T>
__device__ __forceinline__ void unary_vec_loop(size_t numel, const T* in, T* out) {
    using Vec = typename Vec16<T>::type;          // float4 / uint4
    constexpr int N = 16 / sizeof(T);
    const size_t nvec = numel / N;
    const Vec* vin = reinterpret_cast<const Vec*>(in);
    Vec* vout = reinterpret_cast<Vec*>(out);
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < nvec;
         i += blockDim.x * gridDim.x) {
        Vec v = vin ? vin[i] : vout[i];
        // 逐 lane 应用 FUNC
        vout[i] = apply_lanes<T, N>(v, FUNC);
    }
    // 尾巴（numel % N）用标量循环，只有最后一个 block 会进
    for (size_t i = nvec * N + blockIdx.x * blockDim.x + threadIdx.x; i < numel;
         i += blockDim.x * gridDim.x) {
        T x = in ? in[i] : out[i];
        out[i] = FUNC;
    }
}
```

`binary_op_macros.cuh` 只改 `lhs_cont && rhs_cont` 那个分支（广播/非连续分支保持原样）。
`ucopy`（`copy.cu`）如果也是标量实现，一并处理——它是当前最大单项（1.18ms）。

### 3.3 验收标准

1. **逐位不变**：向量化只改访存宽度，数学不变；用现有单测 + 一个 bf16/f32 的
   `unary/binary/copy` 对比脚本（同一输入，改进前后输出完全相等）。
2. 带宽：`bmul/ugelu/cast/ucopy` 在 (522,768) bf16 上从 ~50–100 提升到 ≥ 250 GB/s。
3. 整模型：本仓 10.75ms → 目标 **9.0–9.3ms**（−1.5~1.8ms）。

### 3.4 落地步骤

1. `cargo vendor` 式地拷入 `third-party/candle-kernels/`（只需该 crate：`Cargo.toml` +
   `build.rs` + `src/*.cu|.cuh` + `src/lib.rs`），根 `Cargo.toml` 加
   `[patch.crates-io] candle-kernels = { path = "..." }`。
2. 改 kernel → `CUDA_COMPUTE_CAP=120 cargo build --release --features vla,cuda`
   （首次全量 PTX 约 5–10 分钟）。
3. 用 `examples/turbovla_bench --no-stages` + nsys 复测。
4. 数字成立后整理成最小 diff，向上游 `huggingface/candle` 提 PR：
   - 分支：`perf/vectorize-elementwise`
   - 内容：`candle-kernels/src/{unary.cu,binary_op_macros.cuh,copy.cu}` + 说明带宽对比
   - PR 描述里带：改前/改后 GB/s、整模型耗时、以及"逐位等价"的验证方法

### 3.5 风险

- `apply_lanes` 需要在 device 端展开成 per-lane 标量运算（宏/模板），注意不要让
  编译器生成 `ld.global.v4` 之外的额外指令。
- 非连续/广播分支**必须**保持原实现（风险最低），只优化 `contiguous` 快路径。
- `numel` 用 `size_t`，但 grid 用 `unsigned`，大张量下要防止回绕（保持现有约定即可）。

---

## 4. P1：tiled flash-attention（收益 ~1.5ms，风险高）

现状：`qkᵀ`（cuBLAS）+ softmax（candle kernel）+ `pv`（cuBLAS）三段，
中间要落盘 `(B,H,261,261)`。一次 naive 融合尝试**失败**（vision 11.5 → 25.7ms，
数值差 5.9e-2），原因是每行一个 block、串行扫 261 次 v、无分块/向量化。

要打过 torch 的 SDPA 必须是标准 FA2 结构：

- grid `(ceil(S/Br), B*H)`，`Br=64`，block 128 线程
- K/V 按 `Bc=64` 分块进 shared（bf16，各 8KB）
- 在线 softmax（running max / running sum），O 累加在寄存器
- `half2`/`uint4` 向量化 load，warp 级 reduce

验收：`examples/heads_probe` 式的**逐段对比**（不是 action 对齐，见 §6）。

---

## 5. P2：其他（不改模型语义的前提下）

| 项 | 预估 | 备注 |
|---|---|---|
| 端到端：抓屏 + HTTP + base64/PNG | −20ms 级 | 服务端 45–62ms 里模型只占 11ms，**这是最大的一块** |
| 视觉降频（动作头每帧、ViT 每 2 帧） | −4ms | 语义变了，需要产品确认 |
| 剩余 `ucopy` 1.18ms 里的非拆头部分（`cat`/`flatten`/`reshape`） | −0.5ms | 需要逐个定位调用点 |
| `bmul`（LayerScale）+ 残差融合 | −0.3ms | 收益小 |

---

## 6. 验证方法论（重要）

**`action` 逐值对齐在 bf16 下不是有效标准。** `examples/chaos_probe`：输入图改 1 个 LSB，
动作最大差 **0.115**（f32 下同样是 0.115）。也就是说任何改变舍入顺序的优化
（换算法、换 kernel、融合）都会让动作移动 ~0.1，落在模型自身噪声内。

正确的验收方式：

1. **`--f32` 对 CPU 参考**：当前 1e-7（f32 路径永远不融合，保证精确）；
2. **中间阶段对拍**：`stages()` 返回 `text/visual/fused_v/fused_l/actions`，
   优化前后比 `visual`/`fused_v`（相对误差应 ≤ 1%）；
3. **逐算子单测**：新 kernel 必须有 `probe` 和 `candle` 路径的 max diff（应为 0 或 1 ulp）；
4. **整机基准**：`turbovla_bench`，**每轮后同步**（图模式下 launch 异步，不同步会量到入队时间）。

---

## 7. 复现环境

```powershell
# 编译（nvcc 不在 PATH 时要显式给；本机 CUDA_PATH 未设，脚本会静默退回无 cuda）
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:PATH = "$env:CUDA_PATH\bin;$env:PATH"
$env:LIB  = "$env:CUDA_PATH\lib\x64;$env:LIB"
$env:CUDA_COMPUTE_CAP = "120"      # 5060 = sm_120；脚本默认 89 是 4060
# CUDA graph 默认就开（TURBOVLA_GRAPH=0 才是逐算子），不用再设

cargo build --release --features vla,cuda --example turbovla_bench
.\target\release\examples\turbovla_bench.exe "E:\models\turbovla\turbovla-libero" 50 --verify

# kernel 级画像
$nsys = "D:\Program Files\NVIDIA Corporation\Nsight Systems 2026.6.1\target-windows-x64\nsys.exe"
& $nsys profile --cuda-graph-trace=node --trace=cuda --sample=none --stats=true `
    -o "$env:TEMP\vla_prof" -- .\target\release\examples\turbovla_bench.exe `
    "E:\models\turbovla\turbovla-libero" 20 --no-stages
```

注意：
- `sentencepiece-sys` 需要 **CMake**（这台机器没装，临时用 `%TEMP%\cmake-4.4.5` 顶）；
- `cargo check` 通过 ≠ 能链接（cuda 的 `.lib` 在 `CUDA_PATH\lib\x64`，缺了报 `LNK1181`）；
- 5 个探针 example 需要 `--features vla`，其中 cuda 相关的要 `vla,cuda`。

## 8. 文件清单

新增：

```
src/models/turbovla/ops.rs          双向跨模态注意力 / 加性 mask（f32 构造）
src/models/turbovla/graph.rs        CUDA graph（静态输入缓冲、预热、捕获、重放）
src/models/turbovla/linear.rs       带 bias epilogue 的 Linear
src/models/turbovla/heads_cuda.rs   拆头/合头专用拷贝 kernel
examples/turbovla_bench.rs          基准（--cpu/--f32/--verify/--no-stages）
examples/{gemm,ops,linear,weight,chaos,stage,heads}_probe.rs   诊断探针
```

改动：`fusion.rs`、`action.rs`、`vision.rs`、`text.rs`、`mod.rs`（注意力合并到 `ops.rs`、
Linear 换实现、softmax 去 f32 往返、拆合头走专用 kernel）。

## 9. 回滚

| 开关 | 作用 |
|---|---|
| `TURBOVLA_GRAPH=0`（默认开） | CUDA graph |
| `TURBOVLA_NO_FUSED_LINEAR=1` | 关 bias epilogue |
| `TURBOVLA_SOFTMAX_F32=1` | 注意力 softmax 回 f32 往返 |

拆头/合头 kernel 没有开关（已验证逐位一致），要回退就删 `heads_cuda` 分支。

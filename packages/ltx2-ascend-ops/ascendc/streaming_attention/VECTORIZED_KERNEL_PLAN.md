# Vectorized streaming attention follow-up

## 2026-06 status

The generated kernel is no longer the original scalar/two-pass bring-up path.
The active default native path now keeps the same quality contract but uses:

- CANN Matmul for QK/PV, still single-core by default for stability;
- row/tile vectorized softmax work in UB instead of per-score scalar `ScalarExp()`;
- one-pass online softmax accumulation with fp32 row max/sum/output accumulator;
- **full-sequence Matmul mode is now the native default** (`mode=1`): one QK and
  one PV CANN Matmul per 16-row query tile (`blockN=seqLen`) instead of a
  per-64-key-block QK/PV launch loop. Fallback to the blocked streaming kernel is
  `LTX2_ASCEND_STREAMING_ATTN_BLOCKED=1` (or
  `LTX2_ASCEND_STREAMING_ATTN_FULL_MATMUL=0`).

### Full-sequence (single-Matmul-per-tile) results

Collapsing the per-key-block Matmul loop into one QK + one PV Matmul per query
tile is a validated, quality-preserving native speedup (fp32 softmax contract
unchanged, `max_abs<=1e-3`, bit-identical across repeats on the probed shapes):

- direct BNSD T16 D64 `~0.18ms -> ~0.15ms` (now faster than chunked);
- direct BNSD T32 D64/D128 now ~16% faster than chunked;
- T512 D128 native `~220ms -> ~106ms`;
- T2048 D64 native `~2650ms -> ~1613ms`;
- T8192 D128 native `~55s -> ~26s`.

It is still **much slower than `AscendChunkedAttention`** on representative
long shapes (T512 D128 native ~106ms vs chunked ~0.5ms) because the generated op
remains single-core with scalar-reduced UB softmax. Native streaming attention
therefore stays opt-in and is **not** a default performance path; full-sequence
mode is only the default *within* the already opt-in native backend.

### Required UB->GM barriers for the per-row UB transfer path

The PV partial and final output rows are staged through single-row UB buffers
(`DataCopy` pvGm half -> `Cast` fp32; output `Muls`/`Cast`/`DataCopy`). Each
UB->GM probability/output `DataCopy` MUST be followed by `PipeBarrier<PIPE_ALL>`
before the next row reuses the single-row UB staging buffer. Without those
barriers, small shapes pass but larger repeated/representative shapes show sparse
non-deterministic outliers (`max_abs` ~0.8-4.8 on random rows, run-to-run
different). With the barriers, repeated calls are bit-identical.

### Rejected: CANN fused big op as a default

`torch_npu.npu_fusion_attention` (AscendFusedAttention) is numerically exact
(`max_abs=0`) but slower than chunked on this 910A box across all probed shapes
(tiny `-90%`, audio `-48%`, T8192 `-16%`). It is not a default candidate; keep it
as the `LTX2_ASCEND_ATTENTION=fused` opt-in.

### Rejected: fp32 score-row vector reductions

CANN `ReduceMax`/`ReduceSum` cannot be used for fp32 score reductions in the
generated `ascend910`/dav-c100 path: simply calling the reduce API instantiates
`vcmax`/`vcmin` with fp32 UBUF pointers and fails compilation. A hand-written
64-lane vector tree (`Max`/`Add` on halves of the row buffer) compiled, but larger
validation hit an AICore illegal-instruction / unaligned-UUB exception at T128
D128. Keep the scalar fp32 max/sum loops unless a new aligned-buffer reduction
scheme is validated.

Rejected candidates that should not be retried without redesign:

- D64 `blockN=128`: sparse representative audio T2048 D64 outlier (`max_abs`
  around 1.5).
- D128 `blockN=128`: sparse T128 D128 wrapper outlier (`max_abs` around 38).
- CANN Matmul `blockDim>1` multicore path: now behind
  `LTX2_ASCEND_STREAMING_ATTN_MULTICORE=1` and optional
  `LTX2_ASCEND_STREAMING_ATTN_MULTICORE_CORES=N`, but 8-core and 2-core both
  failed representative audio T2048 D64 despite passing small/T256 validation.
  Keep it off. Production multicore needs a different design (e.g. custom cube
  pipeline/per-core independent decomposition), not the shared CANN Matmul object
  path.

The numerical contract stays unchanged:

```python
scores = (q @ k.transpose(-2, -1)) * (head_dim ** -0.5)
probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
out = probs @ v
```

All inference tensors stay on NPU. HCCL remains outside the attention kernel:
q/k RMSNorm reductions happen before attention and output-projection all-reduce
happens after attention exactly as today.

## V2 target: tiled online-softmax kernel

Use a FlashAttention-like loop over query/key tiles instead of one scalar row at
a time.

### CANN primitive probe result: Matmul maybe; SoftmaxFlashV2 no for dav-m200

CANN 9.0.0 ships useful AscendC libraries, but a reversible generated-op probe
showed an important 910A constraint:

* `AscendC::Matmul` is usable as a building block for the two GEMMs
  (`Q @ K^T` and `P @ V`). A reversible generated-op probe on
  `ASCEND_COMPUTE_UNIT=ascend910` compiled/linked a minimal `lib/matmul_intf.h`
  kernel using `REGIST_MATMUL_OBJ`, `matmul::MatmulType`, `SetTensorA/B`,
  `Iterate`, and `GetTensorC`. The probe emitted a non-fatal "do not registe
  tiling struct" diagnostic because it used a minimal local tiling object, but the
  package built successfully.
* `AscendC::SoftmaxFlashV2` is **not exposed** for the generated custom-op
  `ASCEND_COMPUTE_UNIT=ascend910` compile path. The generated project targets the
  old 910A `dav-m200` path, while `adv_api/activation/softmaxflashv2.h` only
  declares `SoftMaxFlashV2TilingFunc` / `SoftmaxFlashV2` behind guarded
  `__NPU_ARCH__` values such as `2002`/`2201`. In the actual generated
  `ascend910` build those names are undeclared, so this project must not depend
  on `SoftmaxFlashV2`.

Therefore the kernel is not pure CANN glue. The correct 910A path is:

1. Use CANN Matmul for the heavy GEMMs (`Q@K^T` and `P@V`) because the generated
   `ascend910` compile/link probe passed.
2. Self-write the online-softmax recurrence in AscendC vector/scalar code with
   fp32 row max, row sum, and output accumulator.
3. Keep a vectorized/tiled dot-product fallback plan only if Matmul runtime
   validation fails despite compiling.

Recommended first tile shape for 910A bring-up:

| Parameter | Initial values | Notes |
|-----------|----------------|-------|
| `BLOCK_M` | `1`, then `4`/`8` | Query rows per program. Start with `1` to isolate recurrence correctness, then increase after correctness. |
| `BLOCK_N` | `32`/`64` | Key/value rows per tile. Tune separately for `D=64` and `D=128`. |
| `D` | `64`, `128` | Keep current dispatch contract; specialize branches by `head_dim`. |
| Accumulator | fp32 | Row max, row sum, scores/probs, and output accumulator stay fp32 before final fp16 store. |

Program mapping:

1. Map each program to one `(batch, head, q_block)` region.
2. Load a contiguous `Q[BLOCK_M, D]` tile into UB/L1.
3. Initialize per-query-row `m = -inf`, `l = 0`, and `acc[BLOCK_M, D] = 0` in fp32.
4. For each `K/V` tile `n0:n0+BLOCK_N`:
   - load `K[BLOCK_N, D]` and `V[BLOCK_N, D]`;
   - compute `S = Q @ K^T * scale` with CANN Matmul if available, otherwise a
     vectorized dot-product loop;
   - self-write online softmax per query row:
     - `m_new = max(m, rowmax(S))`
     - `alpha = exp(m - m_new)`
     - `P = exp(S - m_new)`
     - `l_new = l * alpha + rowsum(P)`
     - `acc = acc * alpha + P @ V`
   - set `m = m_new`, `l = l_new`.
5. Store `(acc / l).to(fp16)` to the original BNSD output location.

The scalar kernel remains the reference/native smoke path until this tiled path
passes validation.

## Host tiling changes

Extend `StreamingAttentionTilingData` beyond the current shape-only fields:

- `blockM`, `blockN` — selected tile sizes;
- `qBlockCount` — `ceil(seqLen / blockM)`;
- optional `TCubeTiling` fields for `Q@K^T` and `P@V` if the Matmul path is used;
- optional per-shape flags for `headDim == 64` vs `128`;
- workspace size if Matmul/KFC needs it.

Initial host selection should be conservative and deterministic:

```text
D=64:  BLOCK_M=1, BLOCK_N=64    # V2a recurrence bring-up
D=128: BLOCK_M=1, BLOCK_N=32    # V2a recurrence bring-up
```

After V2a correctness, increase `BLOCK_M` to `4`/`8` and retune `BLOCK_N`. Keep
Python env hints `LTX2_ASCEND_STREAMING_ATTN_BLOCK_M` and
`LTX2_ASCEND_STREAMING_ATTN_BLOCK_N`, but validate values in host tiling and fall
back to safe defaults if unsupported.

## Bring-up order

1. **Rejected V2a: ad-hoc vector-load dot path**
   - Tried replacing scalar q/k `GetValue` loads with UB `DataCopy` + `Cast` +
     vector `Mul`, followed by scalar reduction over the UB product. The kernel
     compiled and some single calls matched the reference, but repeated native
     calls produced unstable outliers (`max_abs` up to hundreds on tiny shapes).
   - This path is rejected and reverted. Do not reuse it without a deeper buffer /
     synchronization redesign.

2. **Next V2a: Matmul-assisted GEMMs**
   - Use the confirmed-compiling `lib/matmul_intf.h` path for `Q@K^T` and `P@V`.
   - Carry proper host-produced `TCubeTiling` in `StreamingAttentionTilingData`
     instead of the local probe tiling object.
   - Keep the self-written online-softmax recurrence and validate Matmul runtime
     output against the existing fp32-softmax/chunked NPU reference.

3. **V2c: multi-row tiled path**
   - Increase to `BLOCK_M=4` or `8`.
   - Keep online-softmax state per row in fp32.
   - Add tail handling for `seqLen % BLOCK_M` and `seqLen % BLOCK_N`.
   - Double-buffer K/V tiles if UB/L1 budget permits.

4. **V2d: production gating**
   - Run tiny/small native validation, then representative single-rank shapes:
     `B=1,H=8,T=512,D=128`, `B=1,H=8,T=2048,D=64`, and
     `B=1,H=8,T=8192,D=128`.
   - Run 4-card TP smoke with `LTX2_ASCEND_ATTENTION=streaming`.
   - Run HQ A/B against `LTX2_ASCEND_ATTENTION=eager` on the verified-good path.
   - Only consider default enablement after unchanged video/audio quality and
     repeatable speedup.

## Build wiring for available CANN primitives

The generated `op_kernel/CMakeLists.txt` already builds AscendC kernels via
`npu_op_kernel_*`. For Matmul, model the built-in legacy attention kernels:

```cpp
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
```

Use `REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.bmm, bmmTiling)` and carry
any needed `TCubeTiling` in `StreamingAttentionTilingData`. Do **not** include or
rely on `SoftmaxFlashV2` for `ascend910`/`dav-m200`; the generated build probe
showed the public API is guarded out for this target.

## Guardrails

- Do not use fp16 softmax accumulators or approximate low-precision shortcuts as
  the candidate default.
- Do not implement masked/cross-attention by dropping or booleanizing existing
  additive mask semantics; unsupported masks must continue to use chunked
  attention.
- Do not add HCCL or rank communication inside the custom op.
- Do not route inference tensors through CPU for validation or timing.
- Keep the native op optional and explicitly enabled until end-to-end quality and
  speed validation pass.

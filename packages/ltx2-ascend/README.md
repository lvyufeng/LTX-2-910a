# ltx2-ascend

Ascend NPU (910A / 910B-class, CANN 9.0.0) entry point and runtime tuning for
LTX-2 inference. The CLI (`python -m ltx2_ascend.cli`) wraps the `ltx-pipelines`
pipelines with NPU-aware device/dtype defaults and HCCL tensor parallelism.

## Verified good path (do not regress)

4-card HCCL tensor parallel, `two-stage-hq` (Res2s), `960x1664`, `121` frames,
`24` fps, `15` HQ steps. Transformer runs fp16; **embeddings processor, video
decoder, and audio decoder run fp32** (the CLI sets these fp32 defaults on NPU
automatically). `LTX2_GUIDANCE_FP32=1`. Attention base chunk defaults to `1536`
on the TP HQ path, with a scoped shape policy using chunk2048 only for the
validated stage-2 long-K self-attention shape.

Known-bad flags that corrupt output and must stay **off**: fp16/TP embeddings
processor (`LTX2_TP_EMBEDDINGS_PROCESSOR`,
`LTX2_ASCEND_EMBEDDINGS_FEATURE_EXTRACTOR_AUTOCAST`), fp16 video/audio decode
autocast (`LTX2_ASCEND_VIDEO_DECODER_AUTOCAST`,
`LTX2_ASCEND_AUDIO_DECODER_LOWRES_AUTOCAST`), the experimental-precision master
gate (`LTX2_ASCEND_EXPERIMENTAL_PRECISION`), rank0-only prompt
(`LTX2_TP_PROMPT_RANK0_ONLY`), and disabling the TP text encoder
(`LTX2_DISABLE_TP_TEXT_ENCODER`). Everything must run on NPU; no CPU execution
on the good path.

## Runtime env knobs

| Env var | Default | Effect |
|---------|---------|--------|
| `LTX2_GUIDANCE_FP32` | unset | Run guidance math in fp32 (part of the good path). |
| `LTX2_ASCEND_ATTENTION` | unset (auto) | `eager`/`math` = chunked attention; `streaming`/`custom` = optional AscendC streaming attention for supported no-mask shapes with chunked fallback; `fused` = `npu_fusion_attention` (not viable on this host — see below). |
| `LTX2_ASCEND_STREAMING_ATTN_MIN_T` | `1` | Minimum sequence length for the opt-in streaming attention dispatch. Smaller/unsupported shapes fall back to chunked attention. |
| `LTX2_ASCEND_STREAMING_ATTN_FULL_MATMUL` | unset (on inside native) | Native streaming bring-up uses full-sequence QK/PV Matmul with validated `blockM=32` and segmented whole-row UB softmax (`4096` scores/segment); set `0`/`off` or `LTX2_ASCEND_STREAMING_ATTN_BLOCKED=1` to force the slower blocked online path. |
| `LTX2_ASCEND_STREAMING_ATTN_BLOCK_M`, `LTX2_ASCEND_STREAMING_ATTN_BLOCK_N` | `32`/auto inside native | Optional native tile hints. `blockM=32` is the validated full-matmul default; `64/128` are experimental and exceeded tolerance on T8192/D128. |
| `LTX2_ASCEND_STREAMING_ATTN_SHAPE_WARMUP` | unset (on inside native) | Run a throwaway NPU native call after shape changes to avoid generated ACLNN/CANN Matmul first-call state pollution on 910A; set `0` for raw benchmarking only. |
| `LTX2_ASCEND_STREAMING_ATTN_MULTICORE_CORES` | `24` inside native | Row-parallel AiCore cap for the native custom op; 24 was faster than 16/32 on representative BM32 full-matmul shapes. |
| `LTX2_ASCEND_STREAMING_ATTN_STRICT` | unset | Validation mode: raise instead of falling back when streaming attention cannot dispatch. |
| `LTX2_ASCEND_ATTENTION_TRACE` | unset | Log attention metadata (shape/dtype/mask/backend/fallback) without tensor dumps. |
| `LTX2_ASCEND_ATTENTION_CHUNK` | `4096` (`1536` on TP HQ) | Query tiling for chunked attention; effective chunk is capped by the fp32 score working-set budget. Explicit env overrides are authoritative. |
| `LTX2_ASCEND_TP_HQ_SHAPE_POLICY` | unset (`1` only when the TP-HQ CLI applies its default chunk) | Scoped TP-HQ policy: keep stage-1 self-attention at chunk1536, but use chunk2048 for the validated stage-2 `B1/H8/Q=K=24960/D128` long-K self-attention shape. Set `0` to disable. |
| `LTX2_ASCEND_ATTENTION_EAGER_MAX_MB` | `1024` | fp32 full-attention working-set threshold for using the eager one-shot path instead of chunked attention. |
| `LTX2_ASCEND_ATTENTION_CHUNK_MAX_MB` | follows eager budget (`1600` on TP HQ) | fp32 per-chunk score working-set budget used to cap effective chunk size independently from the full-eager threshold. TP HQ raises this scoped default so the validated stage-2 long-K policy can use chunk2048 while keeping full-eager conservative. |
| `LTX2_ASCEND_SCALED_MASKED_SOFTMAX` | unset/off | Experimental opt-in only. The CANN `npu_scaled_masked_softmax` path is default-off because real TP-HQ sweeps found it non-bit-identical or incorrect on K=128/6240 shapes and slower than `torch.softmax(scores.float())`. |
| `LTX2_ASCEND_ROPE` | unset (`npu_rotary_mul`) | Default ON on NPU: rotary embedding via CANN `npu_rotary_mul`. Set `eager`/`off`/`0`/`false` to force PyTorch rope. See below. |
| `LTX2_ASCEND_ATTENTION_CAT_MAX_MB` | `32` | Use `torch.cat` chunk output assembly below this output-size threshold; set `0` to force indexed output writes. |
| `LTX2_ASCEND_PROFILE` | unset | Emit `[profile] <section> <secs>` timing lines. |
| `LTX2_ASCEND_PROFILE_DETAIL` | unset | Emit default-off rank0 aggregated TP/attention internals (`[profile-detail]` / `[profile-detail-attn]`); intended for diagnosis only because it inserts synchronizations. |

## Default-enabled TP optimizations

These optimizations are enabled by default because they are mathematically exact
and validated on the 4-card TP path. Per the project preference, performance
wins with unchanged quality/stability should be defaults, with explicit fallback
only where useful.

* **RoPE via CANN `npu_rotary_mul`** — default ON on NPU; force PyTorch fallback
  with `LTX2_ASCEND_ROPE=eager` (or `off`/`0`/`false`).
* **q/k paired RoPE frequency duplication** — when q and k share the same RoPE
  tensors, construct full-width `r1/r2` once and reuse them for both rotary
  calls.
* **TP q/k RMSNorm reduction fusion** — when q/k prefix shapes match, concatenate
  their local square sums and perform one HCCL `all_reduce` instead of two. Shape
  mismatch (e.g. cross-attention) falls back to the old exact two-reduction path.
* **Chunked attention long-sequence fast path** — general `AscendChunkedAttention`
  defaults to `LTX2_ASCEND_ATTENTION_CHUNK=4096` and
  `LTX2_ASCEND_ATTENTION_CAT_MAX_MB=32`. The effective chunk is capped by the fp32
  score working-set budget to avoid large transients. The 4-card TP-HQ CLI path
  keeps its scoped `1536` base chunk override, defaults
  `LTX2_ASCEND_ATTENTION_CHUNK_MAX_MB=1600`, and enables
  `LTX2_ASCEND_TP_HQ_SHAPE_POLICY=1` only when applying that default. The policy is
  shape-aware: stage-1 self-attention (`Q=K=6240,D=128`) stays at chunk1536, while
  the dominant stage-2 self-attention (`Q=K=24960,D=128`) uses chunk2048. Focused
  NPU sweeps measured both policies bit-identical; stage2 chunk2048 was consistently
  faster than chunk1536, while stage1 regressed at chunk2048. Cross-attention shapes
  stay below the full-eager threshold and already use the fastest eager path.
* **Scaled-masked-softmax is not a default** — CANN `npu_scaled_masked_softmax` is
  now explicit opt-in only via `LTX2_ASCEND_SCALED_MASKED_SOFTMAX=1`. Real TP-HQ
  sweeps found it non-bit-identical on K=128 and badly incorrect on K=6240, so the
  safe default remains `torch.softmax(scores.float())`.

### Custom-operator track: streaming attention

The next custom-op target is a FlashAttention-like AscendC streaming attention
kernel. It is **not default-enabled yet**: `LTX2_ASCEND_ATTENTION=streaming` (or
`custom`) selects the wrapper only for validation. Until the native op is built
and installed, or whenever a shape is unsupported, the wrapper falls back to the
unchanged `AscendChunkedAttention` path.

Current v1 dispatch contract:

* inputs stay on NPU and use fp16 `(B, T, H*D)` tensors;
* the wrapper reshapes to BNSD `(B, H, T, D)` for the native call;
* only no-mask, square `q_len == k_len`, `head_dim` 64/128 cases are candidates;
* masks, non-square/cross attention, unsupported dtypes/devices, missing native
  symbols, and native runtime errors all use chunked attention;
* `LTX2_ASCEND_STREAMING_ATTN_STRICT=1` raises instead of falling back for tests.

The optional scaffold lives in `packages/ltx2-ascend-ops/`. It now builds an
AscendC/CANN-Matmul ACLNN prototype and an optional
`torch.ops.ltx2_ascend.streaming_attention` binding scaffold. This is still a
native bring-up path only, not a model default: the current native-internal default
is full-sequence QK/PV Matmul with `blockM=32`, plus a shape-change warmup guard to
avoid 910A generated ACLNN/CANN Matmul first-call state pollution. It preserves the
fp32-softmax contract and passed representative single-rank native checks for
T512/D128, T2048/D64, and T8192/D128, but remains slower than the validated
`AscendChunkedAttention` path. The current full-matmul native path keeps softmax
score rows in UB in up-to-4096-score segments, uses 32B-aligned fp32 vector-tree
reductions (with scalar tail before any unaligned fp32 source offset), and writes
probabilities back once per segment. This cut T8192/D128 direct native from
~1.07s -> ~0.49s with the first aligned reducer, then to ~54ms with segmented
whole-row UB softmax; chunked is still ~25ms, so the backend remains opt-in. A
full-PV Matmul A-from-UB variant is not a viable long-sequence landing path:
BM32/T8192 would need ~512 KiB of UB just for the probability tile, and a
temporary matmul-only build measured QK+PV alone at ~39.8ms, already slower than
chunked. The slower blocked online path remains available with
`LTX2_ASCEND_STREAMING_ATTN_FULL_MATMUL=0` or
`LTX2_ASCEND_STREAMING_ATTN_BLOCKED=1`; `blockM=64/128` are explicit experiments
only because they exceeded tolerance on T8192/D128.

Build/install is out-of-band with CANN `msopgen`/AscendC templates, using vendor
`ltx2_ascend`; `--probe-only` reports `custom_streaming_attention` so the runtime
can confirm whether native symbols are visible. For local, uninstalled package
testing, point `ASCEND_CUSTOM_OPP_PATH` at the packaged vendor root
`.../custom_opp_ubuntu_aarch64.run/packages/vendors/ltx2_ascend` and load the
matching `op_api/lib/libcust_opapi.so`; pointing at the raw `build_out` or
`packages` parent can make ACLNN symbol lookup succeed while every
`GetWorkspaceSize` call fails. The focused native validation entry point is
`scripts/validate_streaming_attention_native.py`; it requires the explicit native
enable env and compares direct BNSD native output plus the model wrapper against
the fp32-softmax/chunked NPU reference. The focused benchmark entry point is
`scripts/bench_streaming_attention.py`; it labels whether it is timing native
dispatch or fallback-wrapper dispatch and keeps representative large shapes behind
`--include-representative`. Do not make this backend a model default until op-level
tolerances, 4-card TP smoke, and full HQ A/B pass with unchanged visual/audio
quality and repeatable end-to-end speedup.

## Custom-operator track: RoPE via `npu_rotary_mul`

**Decision (Phase 1): wrap the CANN op; no self-written AscendC kernel needed.**

The directive was: use CANN's big operators where they exist and work; self-write
AscendC only for what CANN lacks or cannot run on this 910A-class host. For
rotary positional embedding (RoPE), CANN already ships a working op.

`torch_npu.npu_rotary_mul(x, r1, r2, "half")` computes
`out = r1*x + r2*cat(-x2, x1)`. LTX `apply_split_rotary_emb` (the default SPLIT
rope) is algebraically identical when `r1 = cat([cos, cos], -1)` and
`r2 = cat([sin, sin], -1)` over the half-length `D/2` cos/sin, using the same
`(B, T, H*D) -> (B, H, T, D)` layout adapter LTX already applies internally.

Measured on-device (fp16, env `ltx2-npu`, real LTX shapes — video `H=32 D=128`,
audio `H=32 D=64`, TP shard `H=8 D=128`, `T` up to 8192): **bit-for-bit
identical** to `apply_split_rotary_emb` (max abs diff `0.000e+00`, same
fp32-oracle error) and **~1.3–1.6× faster**. So there is no numerical reason to
hand-write a kernel; wrapping the CANN op is strictly better on maintenance.

Implementation: `ltx_core.model.transformer.rope_npu.apply_rotary_emb_backend` and
`apply_rotary_emb_pair_backend`, drop-ins for `apply_rotary_emb`. The pair helper
is used at both attention call sites and, when q/k share the same RoPE tensors,
constructs the duplicated full-width `r1 = cat([cos, cos])` /
`r2 = cat([sin, sin])` only once and reuses it for both `npu_rotary_mul` calls.
It uses the NPU op by default on NPU when the rope type is SPLIT and the input
tensor has a supported layout; set `LTX2_ASCEND_ROPE=eager` (or `off`/`0`/
`false`) to force the PyTorch fallback. Any other unsupported case (INTERLEAVED
legacy rope, non-NPU tensor, unexpected shape, or runtime error) silently falls
back to the unchanged PyTorch path. Wired into both rope call sites: `ops.py`
`PytorchPreAttention` (non-TP) and `ascend_tensor_parallel.py`
`HCCLTensorParallelAttention` (TP, after `_slice_rope`).

The backend is **default ON** on NPU after the profiled HQ A/B at
`960x1664`, `121` frames, 5 steps showed positive timing with unchanged output
semantics: stage-1 loop `79.899s -> 78.961s` (+1.19%), stage-2 loop
`133.232s -> 132.581s` (+0.49%), loop sum +0.75%, and peak RSS changed by only
~0.06 GiB. The q/k pair helper is also bit-exact and microbenchmarks faster
than two separate NPU RoPE calls: +9.13% on a TP video shard (`H=8,T=8192,D=128`),
+11.61% on full video heads (`H=32,T=8192,D=128`), and +21.11% on audio
(`H=32,T=2048,D=64`). The TP q/k RMSNorm reduction fusion is bit-exact and
microbenchmarks faster than two separate HCCL reductions: +51.54% on the TP video
shard (`T=8192`) and +40.99% on the smoke shape (`T=512`). TP QKV/KV projection
fusion is bit-exact and microbenchmarks faster than separate projection launches:
self-attention +95.18% on `T=512` and +49.07% on `T=8192`; cross-attention KV
fusion is small positive/neutral (+3.75% text-ish, +0.17% stage2-ish). Equivalence
is pinned by `tests/test_rope_npu_backend.py` plus the 4-rank RMSNorm
microbenchmark (`/tmp/test_tp_rmsnorm_pair.py`) and QKV microbenchmark
(`/tmp/bench_tp_qkv_fusion.py`).

### Operators evaluated but NOT used on the good path

* `npu_fusion_attention` — `FlashAttentionScore InferShape failed / Op has no
  infershape func` on this host even after the kernel-include fix. Stays off;
  the good path uses `AscendChunkedAttention`.
* `npu_mm_all_reduce_base` (MC2 fused matmul+all-reduce) — `executor is nullptr`
  on the 4-rank HCCL setup. Output projection stays `Linear` + `all_reduce`.

### Toolchain note (AscendC JIT include fix)

Some CANN ops (e.g. `npu_rms_norm`) JIT-compile a kernel on first use. On this
image that failed with `fatal error: 'cstdint' file not found` until the system
C++ include dirs were made visible to the compiler. `ltx_core.accelerator`
(`configure_ascend_kernel_includes`, called from `configure_npu_runtime`) now
discovers and exports those paths automatically; `--probe-only` reports the
resolved `ascendc_kernel_includes`.

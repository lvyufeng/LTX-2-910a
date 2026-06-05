# ltx2-ascend-ops

Optional custom AscendC operators for LTX-2 Ascend inference.

This package is a scaffold for the streaming attention operator. It is kept
separate from the core Python packages so a missing native build never breaks the
verified-good path. The model code imports `ltx2_ascend_ops` lazily only when
`LTX2_ASCEND_ATTENTION=streaming` is selected, and falls back to
`AscendChunkedAttention` if the native op is unavailable.

## Streaming attention target

External model tensors use `(B, T, H*D)`. The Python wrapper in `ltx-core`
converts them to BNSD `(B, H, T, D)` and calls:

```python
ltx2_ascend_ops.streaming_attention.streaming_attention(q, k, v, scale, block_m, block_n)
```

The native op is expected to compute no-mask attention equivalent to:

```python
scores = (q @ k.transpose(-2, -1)) * scale
probs = torch.softmax(scores.float(), dim=-1).to(v.dtype)
out = probs @ v
```

without materializing the full score/prob matrix. It should support fp16 q/k/v,
fp16 output, and high-precision online softmax accumulation.

## Generated AscendC skeleton

`ascendc/streaming_attention/StreamingAttention.json` is the fp16 ND op schema.
`generate.sh` calls CANN 9.0.0 `msopgen` through the active Python interpreter
(the installed `msopgen` shebang can point at `/root/miniconda3`) and writes the
project to `ascendc/streaming_attention/generated/`.

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
conda activate ltx2-npu
packages/ltx2-ascend-ops/ascendc/streaming_attention/generate.sh
packages/ltx2-ascend-ops/ascendc/streaming_attention/build.sh
```

`build.sh` also exports the host C++ include paths needed by the AscendC compiler
(`cstdint`/`stdlib.h` fix) and has been verified to build the generated project
into `custom_opp_ubuntu_aarch64.run`.

The current kernel is a scalar/one-row-per-core online-softmax bring-up
prototype for fp16 BNSD `(B, H, T, D)` tensors with `head_dim` 64 or 128 and
square no-mask attention. It is intended to validate ACLNN wiring and numerical
semantics, not performance. It computes the same scale from `head_dim` that the
Python wrapper passes (`dim_head**-0.5`) and keeps row max/sum/output
accumulation in fp32 before writing fp16 output. A tiny NPU smoke test using the
unpacked local OPP artifacts passed against the fp32-softmax reference with
`max_abs=9.77e-4` and `mean_abs=7.4e-5` for `(1, 1, 8, 64)`. Larger shape tests,
4-card TP smoke, and HQ A/B validation are still required before this can become
a performance candidate or default path.

CANN generates ACLNN C API artifacts such as `aclnn_streaming_attention.h` and
`libcust_opapi.so`, but it does not generate a PyTorch binding automatically.
This package therefore includes an optional out-of-band binding scaffold:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
conda activate ltx2-npu
packages/ltx2-ascend-ops/csrc/build_streaming_attention_binding.sh
```

The binding registers `torch.ops.ltx2_ascend.streaming_attention(q, k, v, scale,
block_m, block_n)` and calls the generated
`aclnnStreamingAttentionGetWorkspaceSize` / `aclnnStreamingAttention` API on the
current torch_npu stream. The binding is lazily imported only when
`LTX2_ASCEND_STREAMING_ATTN_ENABLE_NATIVE=1` is set. Its availability check also
requires `libcust_opapi.so` to be resolvable; otherwise the main model path keeps
falling back to `AscendChunkedAttention`.

After building the binding but before installing the OPP system-wide, local
bring-up can point directly at the unpacked artifacts:

```bash
export LTX2_ASCEND_STREAMING_ATTN_LIB="$PWD/packages/ltx2-ascend-ops/ascendc/streaming_attention/generated/build_out/op_host/libcust_opapi.so"
export ASCEND_CUSTOM_OPP_PATH="$PWD/packages/ltx2-ascend-ops/ascendc/streaming_attention/generated/build_out/_CPack_Packages/Linux/External/custom_opp_ubuntu_aarch64.run/packages/vendors/ltx2_ascend:${ASCEND_CUSTOM_OPP_PATH:-}"
```

For an installed OPP, use the paths printed by the generated installer instead.

Until those symbols are installed, `availability_report()` reports that the
native operator is unavailable and the main model path falls back automatically.
After the native OPP and PyTorch binding are installed, set
`LTX2_ASCEND_STREAMING_ATTN_ENABLE_NATIVE=1` to allow dispatch; this prevents an
accidental `torch.ops` name collision from changing inference behavior.

A focused validation harness is available at
`../../scripts/validate_streaming_attention_native.py` from the repo root. It
runs only when native dispatch is explicitly available, checks direct BNSD native
output and the `AscendStreamingAttention` wrapper against NPU fp32-softmax /
chunked references, and keeps the larger scalar-prototype cases behind
`--include-large` because this bring-up kernel is not a performance candidate yet.
For timing, use `../../scripts/bench_streaming_attention.py`; it reports the
native availability string, labels native-vs-fallback wrapper timing explicitly,
and keeps representative TP/audio/HQ shapes behind `--include-representative`.
The next native implementation step is documented in
`ascendc/streaming_attention/VECTORIZED_KERNEL_PLAN.md`: replace the scalar
one-row kernel with a tiled online-softmax path while preserving fp32 softmax
accumulation and exact fallback semantics. A reversible build probe showed CANN
`SoftmaxFlashV2` is guarded out for the generated `ascend910`/`dav-m200` build,
so the online-softmax recurrence must stay self-written; CANN `Matmul` does
compile/link and remains the preferred building block for `Q@K^T` and `P@V`. An
ad-hoc vector-load dot path was also rejected: it compiled, but repeated native
calls produced unstable outliers, so the source was reverted to the validated
scalar bring-up kernel.

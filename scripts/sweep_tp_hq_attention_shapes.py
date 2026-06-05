"""Shape-policy sweep for the TP-HQ rank-local attention shapes.

Goal: confirm whether the chunk/eager policy for the attention shapes that
actually dominate the 4-card TP-HQ (960x1664, 121f) run can still be tuned
*losslessly*.  All tensor compute stays on NPU; only timing/metric scalars are
host-side.

The shapes below come from the 4-card TP-HQ detail profile
(``LTX2_ASCEND_PROFILE_DETAIL=1``), ranked by total attention time:

    B1 H8 Q24960 K24960 D128  -- stage-2 video self-attention (dominant)
    B1 H8 Q 6240 K 6240 D128  -- stage-1 video self-attention
    B1 H8 Q24960 K  128 D128  -- cross-attention (small K, already eager)
    B1 H8 Q24960 K  126 D 64  -- cross-attention (small K, already eager)
    B1 H8 Q  126 K24960 D 64  -- cross-attention (small Q, large K, already eager)

For each shape we time a set of *policies* (eager vs chunked at several chunk
sizes, with/without the fp32 ``npu_scaled_masked_softmax`` fast path) and compare
every policy's output against a canonical fp32-softmax reference (small chunk,
scaled-masked-softmax disabled).  A policy is only a candidate default if its
output is bit-identical to that reference (max_abs == 0) -- the project forbids
any quality-reducing change.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from ltx_core.accelerator import configure_npu_runtime
from ltx_core.model.transformer.attention import AscendChunkedAttention

_FP32_BYTES = 4


@dataclass(frozen=True)
class Shape:
    name: str
    batch: int
    heads: int
    q_len: int
    k_len: int
    head_dim: int
    iters: int


@dataclass(frozen=True)
class Policy:
    label: str
    eager: bool
    chunk: int  # query chunk size when not eager
    sms: bool  # allow npu_scaled_masked_softmax fast path


@dataclass(frozen=True)
class Timing:
    median_ms: float
    min_ms: float
    max_ms: float


def _shapes() -> list[Shape]:
    return [
        Shape("stage2-self", 1, 8, 24960, 24960, 128, 3),
        Shape("stage1-self", 1, 8, 6240, 6240, 128, 5),
        Shape("cross-q24960-k128-d128", 1, 8, 24960, 128, 128, 5),
        Shape("cross-q24960-k126-d64", 1, 8, 24960, 126, 64, 5),
        Shape("cross-q126-k24960-d64", 1, 8, 126, 24960, 64, 10),
    ]


def _full_score_bytes(s: Shape) -> int:
    return s.batch * s.heads * s.q_len * s.k_len * _FP32_BYTES


def _chunk_transient_bytes(s: Shape, chunk: int) -> int:
    return s.batch * s.heads * min(chunk, s.q_len) * s.k_len * _FP32_BYTES


def _sms_eligible(s: Shape, rows: int) -> bool:
    # Mirror AscendChunkedAttention._should_use_scaled_masked_softmax contract:
    # last dim (k_len) in [32, 8192] divisible by 32; rows and k_len >= min_t.
    return 32 <= s.k_len <= 8192 and s.k_len % 32 == 0 and rows >= 1


def _policies(s: Shape, max_transient_gb: float) -> list[Policy]:
    max_transient = int(max_transient_gb * (1024**3))
    policies: list[Policy] = []

    # Eager (full fp32 score) only if the transient fits the cap.
    if _full_score_bytes(s) <= max_transient:
        policies.append(Policy("eager", eager=True, chunk=s.q_len, sms=False))
        if _sms_eligible(s, s.q_len):
            policies.append(Policy("eager+sms", eager=True, chunk=s.q_len, sms=True))

    for chunk in (512, 768, 1024, 1344, 1536, 2048, 3072, 4096):
        if chunk >= s.q_len:
            continue
        if _chunk_transient_bytes(s, chunk) > max_transient:
            continue
        policies.append(Policy(f"chunk{chunk}", eager=False, chunk=chunk, sms=False))
        # Only add an explicit +sms variant where the chunk rows actually make it
        # eligible (so we can measure the fast path's effect, not a no-op).
        if _sms_eligible(s, min(chunk, s.q_len)):
            policies.append(Policy(f"chunk{chunk}+sms", eager=False, chunk=chunk, sms=True))

    return policies


def _make_attn(p: Policy) -> AscendChunkedAttention:
    if p.eager:
        # eager_max_mb huge -> always take the full-eager branch.
        attn = AscendChunkedAttention(query_chunk_size=max(p.chunk, 1), eager_max_mb=1 << 30)
    else:
        # Force chunked and lift the chunk cap so the requested chunk is honored
        # (we gate transient size separately in _policies).
        attn = AscendChunkedAttention(query_chunk_size=p.chunk, eager_max_mb=0)
        attn.chunk_max_bytes = 1 << 62
    attn._scaled_masked_softmax_disabled = not p.sms
    if p.sms:
        attn._scaled_masked_softmax_min_t = 1
    else:
        # Force the plain fp32 torch.softmax path regardless of size.
        attn._scaled_masked_softmax_min_t = 1 << 30
    return attn


def _reference_attn() -> AscendChunkedAttention:
    # Canonical fp32-softmax reference: small chunk, scaled-masked-softmax OFF.
    attn = AscendChunkedAttention(query_chunk_size=256, eager_max_mb=0)
    attn.chunk_max_bytes = 1 << 62
    attn._scaled_masked_softmax_disabled = True
    attn._scaled_masked_softmax_min_t = 1 << 30
    return attn


def _time_ms(fn, *, warmup: int, repeats: int, iters: int) -> Timing:
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.npu.synchronize()
        samples.append((time.perf_counter() - start) / iters * 1e3)
    return Timing(statistics.median(samples), min(samples), max(samples))


def _diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    delta = (actual.float() - expected.float()).abs()
    return float(delta.max().item()), float(delta.mean().item())


def _run_shape(s: Shape, args, device: torch.device) -> None:
    inner = s.heads * s.head_dim
    q = torch.randn((s.batch, s.q_len, inner), device=device, dtype=torch.float16)
    k = torch.randn((s.batch, s.k_len, inner), device=device, dtype=torch.float16)
    v = torch.randn_like(k)

    ref = _reference_attn()
    expected = ref(q, k, v, s.heads)
    torch.npu.synchronize()

    print(
        f"\n[{s.name}] B={s.batch} H={s.heads} Q={s.q_len} K={s.k_len} D={s.head_dim} "
        f"full_score={_full_score_bytes(s) / 1024**3:.2f}GiB iters={s.iters}"
    )
    rows = []
    for p in _policies(s, args.max_transient_gb):
        attn = _make_attn(p)
        out = attn(q, k, v, s.heads)
        torch.npu.synchronize()
        max_abs, mean_abs = _diff(out, expected)
        t = _time_ms(lambda: attn(q, k, v, s.heads), warmup=args.warmup, repeats=args.repeats, iters=s.iters)
        bit_identical = max_abs == 0.0
        rows.append((p.label, t.median_ms, t.min_ms, t.max_ms, max_abs, mean_abs, bit_identical))
        del attn

    rows.sort(key=lambda r: r[1])
    best_identical = next((r for r in rows if r[6]), None)
    for label, med, mn, mx, max_abs, mean_abs, ident in rows:
        flag = "BIT-IDENTICAL" if ident else f"diff(max={max_abs:.2e},mean={mean_abs:.2e})"
        marker = ""
        if best_identical is not None and label == best_identical[0]:
            marker = "  <== best lossless"
        print(f"  {label:18s} {med:8.3f}ms [{mn:7.3f},{mx:7.3f}]  {flag}{marker}")

    del q, k, v, expected
    torch.npu.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--max-transient-gb",
        type=float,
        default=2.0,
        help="Skip eager/chunk policies whose fp32 score transient exceeds this (memory safety).",
    )
    parser.add_argument("--only", type=str, default="", help="Comma-separated shape names to restrict the sweep.")
    args = parser.parse_args()

    configure_npu_runtime(args.device)
    torch.manual_seed(args.seed)
    device = torch.device("npu", args.device)

    print("TP-HQ attention shape-policy sweep")
    print("fp32-softmax reference = chunk256, scaled_masked_softmax OFF; all compute on NPU")
    print(f"max_transient_gb={args.max_transient_gb}")

    only = {name.strip() for name in args.only.split(",") if name.strip()}
    for s in _shapes():
        if only and s.name not in only:
            continue
        _run_shape(s, args, device)


if __name__ == "__main__":
    main()

"""Numeric + timing validation for the optional native LongKSoftmax op.

Compares ``ltx2_ascend_ops.long_k_softmax`` against the canonical
``torch.softmax(scores.float(), dim=-1).to(fp16)`` reference on the NPU for the
real TP-HQ long-K self-attention chunk shapes.  All tensor compute stays on the
NPU; only timing/metric scalars are read on the host.

The native op is intentionally opt-in.  This script enables it explicitly and
fails loudly if it is unavailable so a silent fallback cannot be mistaken for a
passing native run.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from ltx2_ascend_ops.long_k_softmax import availability_report, is_available, long_k_softmax
from ltx_core.accelerator import configure_npu_runtime


@dataclass(frozen=True)
class Shape:
    name: str
    batch: int
    heads: int
    q_len: int
    k_len: int
    iters: int


def _shapes() -> list[Shape]:
    return [
        Shape("small-k512", 1, 8, 64, 512, 20),
        Shape("k4096", 1, 8, 64, 4096, 10),
        Shape("k8192", 1, 8, 64, 8192, 10),
        # Real TP-HQ stage-2 self-attention K with the validated chunk Q rows.
        Shape("tp-hq-chunk2048-k24960", 1, 8, 2048, 24960, 3),
        Shape("tp-hq-row1-k24960", 1, 8, 1, 24960, 5),
    ]


def _reference(scores: torch.Tensor) -> torch.Tensor:
    return torch.softmax(scores.float(), dim=-1).to(scores.dtype)


def _diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, bool]:
    delta = (actual.float() - expected.float()).abs()
    finite = bool(torch.isfinite(actual).all().item())
    return float(delta.max().item()), float(delta.mean().item()), finite


def _time_ms(fn, *, warmup: int, repeats: int, iters: int) -> float:
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
    return statistics.median(samples)


def _run_shape(s: Shape, device: torch.device, args) -> bool:
    scores = torch.randn((s.batch, s.heads, s.q_len, s.k_len), device=device, dtype=torch.float16)
    expected = _reference(scores)
    actual = long_k_softmax(scores)
    torch.npu.synchronize()
    max_abs, mean_abs, finite = _diff(actual, expected)
    # fp16 softmax round-off: probabilities are tiny (1/K) so absolute tolerance is
    # generous but mean must be near zero.  This is a quality gate, not a perf gate.
    ok = finite and max_abs <= 2.0e-3 and mean_abs <= 2.0e-5

    ref_ms = _time_ms(lambda: _reference(scores), warmup=args.warmup, repeats=args.repeats, iters=s.iters)
    native_ms = _time_ms(lambda: long_k_softmax(scores), warmup=args.warmup, repeats=args.repeats, iters=s.iters)
    speedup = (ref_ms / native_ms - 1.0) * 100.0 if native_ms > 0 else float("nan")
    print(
        f"[{s.name}] B={s.batch} H={s.heads} Q={s.q_len} K={s.k_len} "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} finite={finite} "
        f"ref={ref_ms:.3f}ms native={native_ms:.3f}ms speedup={speedup:.2f}% ok={ok}"
    )
    del scores, expected, actual
    torch.npu.empty_cache()
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--only", type=str, default="")
    args = parser.parse_args()

    configure_npu_runtime(args.device)
    torch.manual_seed(args.seed)
    device = torch.device("npu", args.device)

    print(f"LongKSoftmax native validation: {availability_report()}")
    if not is_available():
        raise SystemExit("native LongKSoftmax op is required but unavailable")

    only = {name.strip() for name in args.only.split(",") if name.strip()}
    all_ok = True
    for s in _shapes():
        if only and s.name not in only:
            continue
        all_ok = _run_shape(s, device, args) and all_ok

    if not all_ok:
        raise SystemExit("one or more LongKSoftmax shapes failed the quality gate")
    print("all LongKSoftmax shapes passed the quality gate")


if __name__ == "__main__":
    main()

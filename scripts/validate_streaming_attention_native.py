from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import torch

from ltx2_ascend_ops.streaming_attention import availability_report, is_available, streaming_attention
from ltx_core.accelerator import configure_npu_runtime
from ltx_core.model.transformer.attention import AscendChunkedAttention, AscendStreamingAttention


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    tolerance_max: float = 1.0e-2
    tolerance_mean: float = 1.0e-3


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the optional native AscendC streaming attention op against "
            "the NPU fp32-softmax reference and the model wrapper fallback."
        )
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device index to use")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="Also run larger prototype shapes. The current scalar kernel can be slow.",
    )
    parser.add_argument(
        "--wrapper-only",
        action="store_true",
        help="Skip direct BNSD native checks and only validate the AscendStreamingAttention wrapper.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeat each case with fresh NPU inputs to catch repeated-call native state instability.",
    )
    parser.add_argument(
        "--alternating-repeats",
        type=int,
        default=0,
        help=(
            "Run an additional alternating-shape direct+wrapper stress loop. Useful after native changes "
            "because previous Matmul/UB issues appeared only across sequential calls."
        ),
    )
    return parser.parse_args()


def _require_native_available() -> None:
    report = availability_report()
    if is_available():
        print(f"native streaming attention: {report}")
        return
    print(f"native streaming attention unavailable: {report}", file=sys.stderr)
    print(
        "Build/export the optional OPP and binding first, then set:\n"
        "  LTX2_ASCEND_STREAMING_ATTN_ENABLE_NATIVE=1\n"
        "  LTX2_ASCEND_ATTENTION=streaming\n"
        "  LTX2_ASCEND_STREAMING_ATTN_LIB=/path/to/libcust_opapi.so  # for local bring-up\n"
        "  ASCEND_CUSTOM_OPP_PATH=/path/to/vendors/ltx2_ascend:${ASCEND_CUSTOM_OPP_PATH:-}",
        file=sys.stderr,
    )
    raise SystemExit(2)


def _sync_time_ms(fn) -> tuple[torch.Tensor, float]:
    torch.npu.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.npu.synchronize()
    return out, (time.perf_counter() - start) * 1e3


def _diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, bool]:
    delta = (actual.float() - expected.float()).abs()
    finite = bool(torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item())
    return float(delta.max().item()), float(delta.mean().item()), finite


def _reference_bnsd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    probs = torch.softmax(scores, dim=-1).to(v.dtype)
    return torch.matmul(probs, v)


def _validate_direct(case: Case, device: torch.device, *, prefix: str = "direct") -> None:
    scale = case.head_dim**-0.5
    q = torch.randn((case.batch, case.heads, case.seq_len, case.head_dim), device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    expected, ref_ms = _sync_time_ms(lambda: _reference_bnsd(q, k, v, scale))
    actual, native_ms = _sync_time_ms(lambda: streaming_attention(q, k, v, scale=scale))
    max_abs, mean_abs, finite = _diff(actual, expected)
    ok = finite and max_abs <= case.tolerance_max and mean_abs <= case.tolerance_mean
    print(
        f"{prefix} {case.name}: B={case.batch} H={case.heads} T={case.seq_len} D={case.head_dim} "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} finite={finite} "
        f"ref={ref_ms:.3f}ms native={native_ms:.3f}ms ok={ok}"
    )
    if not ok:
        raise AssertionError(f"direct native validation failed for {case.name}")


def _validate_wrapper(case: Case, device: torch.device, *, prefix: str = "wrapper") -> None:
    scale = case.head_dim**-0.5
    del scale  # The wrapper derives scale from heads/head_dim internally.
    q = torch.randn((case.batch, case.seq_len, case.heads * case.head_dim), device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    chunked = AscendChunkedAttention(query_chunk_size=1536, eager_max_mb=0)
    streaming = AscendStreamingAttention()

    expected, ref_ms = _sync_time_ms(lambda: chunked(q, k, v, case.heads))
    actual, native_ms = _sync_time_ms(lambda: streaming(q, k, v, case.heads))
    max_abs, mean_abs, finite = _diff(actual, expected)
    ok = finite and max_abs <= case.tolerance_max and mean_abs <= case.tolerance_mean
    print(
        f"{prefix} {case.name}: B={case.batch} H={case.heads} T={case.seq_len} D={case.head_dim} "
        f"max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} finite={finite} "
        f"chunked={ref_ms:.3f}ms streaming={native_ms:.3f}ms ok={ok}"
    )
    if not ok:
        raise AssertionError(f"wrapper native validation failed for {case.name}")


def _run_alternating_stress(cases: list[Case], device: torch.device, repeats: int, wrapper_only: bool) -> None:
    if repeats <= 0:
        return
    print(f"alternating-shape stress {repeats} cycle(s)")
    for repeat in range(repeats):
        for case in cases:
            prefix = f"stress {repeat + 1}/{repeats}"
            if not wrapper_only:
                _validate_direct(case, device, prefix=f"{prefix} direct")
            _validate_wrapper(case, device, prefix=f"{prefix} wrapper")


def main() -> None:
    args = _parse_args()
    configure_npu_runtime(args.device)
    torch.manual_seed(args.seed)
    os.environ.setdefault("LTX2_ASCEND_ATTENTION", "streaming")
    os.environ.setdefault("LTX2_ASCEND_STREAMING_ATTN_STRICT", "1")
    _require_native_available()

    device = torch.device("npu", args.device)
    cases = [
        Case("tiny-d64", 1, 1, 16, 64),
        Case("small-d64", 1, 2, 32, 64),
        Case("small-d128", 1, 2, 32, 128),
    ]
    if args.include_large:
        cases.extend(
            [
                Case("tp-smoke-scalar-prototype", 1, 8, 128, 128),
                Case("audio-scalar-prototype", 1, 8, 256, 64),
            ]
        )

    repeats = max(1, args.repeats)
    for repeat in range(repeats):
        if repeats > 1:
            print(f"repeat {repeat + 1}/{repeats}")
        for case in cases:
            if not args.wrapper_only:
                _validate_direct(case, device)
            _validate_wrapper(case, device)

    _run_alternating_stress(cases, device, args.alternating_repeats, args.wrapper_only)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import importlib
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch

from ltx_core.accelerator import configure_npu_runtime
from ltx_core.model.transformer.attention import AscendChunkedAttention, AscendStreamingAttention

streaming_attention_ops = importlib.import_module("ltx2_ascend_ops.streaming_attention")

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Case:
    name: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    iters: int
    tolerance_max: float = 1.0e-2
    tolerance_mean: float = 1.0e-3


@dataclass(frozen=True)
class Timing:
    median_ms: float
    min_ms: float
    max_ms: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark AscendChunkedAttention against the optional AscendC streaming "
            "attention wrapper and direct BNSD native op. The native kernel remains "
            "opt-in until it is faster and TP/HQ validated."
        )
    )
    parser.add_argument("--device", type=int, default=0, help="NPU device index to use")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations per measured function")
    parser.add_argument("--repeats", type=int, default=3, help="Repeated timing samples")
    parser.add_argument("--iters", type=int, default=0, help="Override per-case iterations when > 0")
    parser.add_argument("--chunk-size", type=int, default=1536, help="AscendChunkedAttention query chunk size")
    parser.add_argument(
        "--include-representative",
        action="store_true",
        help=(
            "Also run representative TP/audio/HQ shapes. With early native prototypes "
            "this can be slow; use mainly after small cases pass."
        ),
    )
    parser.add_argument(
        "--enable-native",
        action="store_true",
        help="Set LTX2_ASCEND_STREAMING_ATTN_ENABLE_NATIVE=1 in this process before resolving the op.",
    )
    parser.add_argument(
        "--require-native",
        action="store_true",
        help=(
            "Require the optional native op to be available and enable strict mode so "
            "fallbacks fail instead of being timed as native."
        ),
    )
    return parser.parse_args()


def _clear_native_cache() -> None:
    streaming_attention_ops._resolve_native_op.cache_clear()


def _truthy_env(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY_ENV_VALUES


def _env_disabled(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in {"0", "false", "no", "off"}


def _variant_label(base: str) -> str:
    flags: list[str] = [base]
    if _truthy_env("LTX2_ASCEND_STREAMING_ATTN_ASYNC"):
        flags.append("async")
    if _truthy_env("LTX2_ASCEND_STREAMING_ATTN_MULTICORE"):
        flags.append("multicore")
    if not _env_disabled("LTX2_ASCEND_STREAMING_ATTN_FULL_MATMUL") and not _truthy_env("LTX2_ASCEND_STREAMING_ATTN_BLOCKED"):
        flags.append("fullmatmul")
    else:
        flags.append("blocked")
    return "+".join(flags)


def _time_ms(fn: Callable[[], object], *, warmup: int, repeats: int, iters: int) -> Timing:
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
    return Timing(
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
    )


def _diff(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float, bool]:
    delta = (actual.float() - expected.float()).abs()
    finite = bool(torch.isfinite(actual).all().item() and torch.isfinite(expected).all().item())
    return float(delta.max().item()), float(delta.mean().item()), finite


def _to_bnsd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, seq_len, inner = q.shape
    head_dim = inner // heads
    q4d = q.view(batch, seq_len, heads, head_dim).transpose(1, 2).contiguous()
    k4d = k.view(batch, seq_len, heads, head_dim).transpose(1, 2).contiguous()
    v4d = v.view(batch, seq_len, heads, head_dim).transpose(1, 2).contiguous()
    return q4d, k4d, v4d


def _from_bnsd(out: torch.Tensor) -> torch.Tensor:
    batch, heads, seq_len, head_dim = out.shape
    return out.transpose(1, 2).contiguous().view(batch, seq_len, heads * head_dim)


def _cases(include_representative: bool, iters_override: int) -> list[Case]:
    cases = [
        Case("tiny-d64", 1, 1, 16, 64, 50),
        Case("small-d64", 1, 2, 32, 64, 30),
        Case("small-d128", 1, 2, 32, 128, 30),
    ]
    if include_representative:
        cases.extend(
            [
                Case("tp-smoke", 1, 8, 512, 128, 10),
                Case("audio", 1, 8, 2048, 64, 6),
                Case("tp-hq", 1, 8, 8192, 128, 3),
            ]
        )
    if iters_override > 0:
        cases = [Case(c.name, c.batch, c.heads, c.seq_len, c.head_dim, iters_override, c.tolerance_max, c.tolerance_mean) for c in cases]
    return cases


def _case(case: Case, device: torch.device, args: argparse.Namespace, backend_label: str, native_available: bool) -> None:
    q = torch.randn((case.batch, case.seq_len, case.heads * case.head_dim), device=device, dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    chunked = AscendChunkedAttention(query_chunk_size=args.chunk_size, eager_max_mb=0)
    streaming = AscendStreamingAttention()

    expected = chunked(q, k, v, case.heads)
    actual = streaming(q, k, v, case.heads)
    torch.npu.synchronize()
    max_abs, mean_abs, finite = _diff(actual, expected)
    ok = finite and max_abs <= case.tolerance_max and mean_abs <= case.tolerance_mean
    if not ok:
        raise AssertionError(
            f"{case.name} wrapper exceeded tolerance: max_abs={max_abs:.3e}, "
            f"mean_abs={mean_abs:.3e}, finite={finite}"
        )

    t_chunked = _time_ms(lambda: chunked(q, k, v, case.heads), warmup=args.warmup, repeats=args.repeats, iters=case.iters)
    t_streaming = _time_ms(lambda: streaming(q, k, v, case.heads), warmup=args.warmup, repeats=args.repeats, iters=case.iters)
    t_layout_pack = _time_ms(lambda: _to_bnsd(q, k, v, case.heads), warmup=args.warmup, repeats=args.repeats, iters=case.iters)

    direct_fragment = "direct_native=skipped"
    if native_available:
        q4d, k4d, v4d = _to_bnsd(q, k, v, case.heads)
        expected4d = expected.view(case.batch, case.seq_len, case.heads, case.head_dim).transpose(1, 2).contiguous()
        scale = case.head_dim**-0.5
        direct = streaming_attention_ops.streaming_attention(q4d, k4d, v4d, scale=scale)
        torch.npu.synchronize()
        direct_max_abs, direct_mean_abs, direct_finite = _diff(direct, expected4d)
        direct_ok = direct_finite and direct_max_abs <= case.tolerance_max and direct_mean_abs <= case.tolerance_mean
        if not direct_ok:
            raise AssertionError(
                f"{case.name} direct native exceeded tolerance: max_abs={direct_max_abs:.3e}, "
                f"mean_abs={direct_mean_abs:.3e}, finite={direct_finite}"
            )
        t_direct = _time_ms(
            lambda: streaming_attention_ops.streaming_attention(q4d, k4d, v4d, scale=scale),
            warmup=args.warmup,
            repeats=args.repeats,
            iters=case.iters,
        )
        t_layout_unpack = _time_ms(
            lambda: _from_bnsd(direct),
            warmup=args.warmup,
            repeats=args.repeats,
            iters=case.iters,
        )
        direct_speedup = (t_chunked.median_ms / t_direct.median_ms - 1.0) * 100.0
        direct_fragment = (
            f"direct_native={t_direct.median_ms:.3f}ms[{t_direct.min_ms:.3f},{t_direct.max_ms:.3f}] "
            f"direct_speedup={direct_speedup:.2f}% "
            f"direct_max_abs={direct_max_abs:.3e} direct_mean_abs={direct_mean_abs:.3e} "
            f"layout_unpack={t_layout_unpack.median_ms:.3f}ms"
        )

    wrapper_speedup = (t_chunked.median_ms / t_streaming.median_ms - 1.0) * 100.0
    print(
        f"{case.name}: backend={backend_label} B={case.batch} H={case.heads} T={case.seq_len} D={case.head_dim} "
        f"iters={case.iters} max_abs={max_abs:.3e} mean_abs={mean_abs:.3e} "
        f"chunked={t_chunked.median_ms:.3f}ms[{t_chunked.min_ms:.3f},{t_chunked.max_ms:.3f}] "
        f"wrapper_streaming={t_streaming.median_ms:.3f}ms[{t_streaming.min_ms:.3f},{t_streaming.max_ms:.3f}] "
        f"wrapper_speedup={wrapper_speedup:.2f}% "
        f"layout_pack={t_layout_pack.median_ms:.3f}ms "
        f"{direct_fragment}"
    )


def main() -> None:
    args = _parse_args()
    configure_npu_runtime(args.device)
    torch.manual_seed(args.seed)

    os.environ.setdefault("LTX2_ASCEND_ATTENTION", "streaming")
    os.environ.setdefault("LTX2_ASCEND_STREAMING_ATTN_LOG_FALLBACK", "1")
    if args.enable_native or args.require_native:
        os.environ["LTX2_ASCEND_STREAMING_ATTN_ENABLE_NATIVE"] = "1"
    if args.require_native:
        os.environ["LTX2_ASCEND_STREAMING_ATTN_STRICT"] = "1"
    else:
        os.environ.setdefault("LTX2_ASCEND_STREAMING_ATTN_STRICT", "0")
    _clear_native_cache()

    report = streaming_attention_ops.availability_report()
    native_available = streaming_attention_ops.is_available()
    if args.require_native and not native_available:
        raise SystemExit(f"native streaming attention is required but unavailable: {report}")

    base_label = "native-strict" if args.require_native else ("native-available-wrapper" if native_available else "fallback-wrapper")
    backend_label = _variant_label(base_label)
    print("streaming attention benchmark")
    print("note: native AscendC streaming attention remains opt-in; do not default-enable from microbench numbers alone")
    print(f"availability: {report}")
    print(f"backend label: {backend_label}")

    device = torch.device("npu", args.device)
    for case in _cases(args.include_representative, args.iters):
        _case(case, device, args, backend_label, native_available)


if __name__ == "__main__":
    main()

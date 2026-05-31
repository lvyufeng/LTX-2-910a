#!/usr/bin/env python
"""Compare debug tensor dumps from two reference runs (e.g. cpu vs ascend).

Each run writes ``<prefix>_<name>.pt`` payloads via ``ltx_core.debug.dump_tensor``.
This pairs files that share the same ``<name>`` across two prefixes and reports
per-dump-point absolute differences, so divergence can be localized to a specific
pipeline boundary (initial latent -> denoised latent -> decoder input -> first chunk).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _strip_prefix(stem: str, prefix: str) -> str | None:
    head = f"{prefix}_"
    if not stem.startswith(head):
        return None
    return stem[len(head) :]


def _collect(dump_dir: Path, prefix: str) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for path in sorted(dump_dir.glob(f"{prefix}_*.pt")):
        name = _strip_prefix(path.stem, prefix)
        if name is not None:
            out[name] = path
    return out


def _compare_one(a_payload: dict, b_payload: dict) -> dict[str, object]:
    a = a_payload["tensor"].float()
    b = b_payload["tensor"].float()
    info: dict[str, object] = {
        "a_shape": tuple(a_payload["shape"]),
        "b_shape": tuple(b_payload["shape"]),
        "a_dtype": a_payload["dtype"],
        "b_dtype": b_payload["dtype"],
    }
    if a.shape != b.shape:
        info["shape_mismatch"] = True
        return info
    diff = (a - b).abs()
    denom = b.abs().mean().item()
    info.update(
        {
            "mean_abs_diff": diff.mean().item(),
            "max_abs_diff": diff.max().item(),
            "rmse": diff.pow(2).mean().sqrt().item(),
            "rel_mean_abs_diff": (diff.mean().item() / denom) if denom > 0 else float("nan"),
            "allclose_1e-3": bool(torch.allclose(a, b, atol=1e-3, rtol=1e-3)),
        }
    )
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump_dir", type=Path, help="Directory containing the .pt dumps")
    parser.add_argument("--a-prefix", default="cpu")
    parser.add_argument("--b-prefix", default="ascend")
    args = parser.parse_args()

    a_files = _collect(args.dump_dir, args.a_prefix)
    b_files = _collect(args.dump_dir, args.b_prefix)

    names = sorted(set(a_files) | set(b_files))
    if not names:
        raise SystemExit(f"No dumps found in {args.dump_dir} for prefixes {args.a_prefix!r}/{args.b_prefix!r}")

    for name in names:
        if name not in a_files:
            print(f"[{name}] MISSING in {args.a_prefix!r}")
            continue
        if name not in b_files:
            print(f"[{name}] MISSING in {args.b_prefix!r}")
            continue
        info = _compare_one(_load(a_files[name]), _load(b_files[name]))
        if info.get("shape_mismatch"):
            print(f"[{name}] SHAPE MISMATCH {info['a_shape']} vs {info['b_shape']}")
            continue
        print(
            f"[{name}] "
            f"dtype={info['a_dtype']}/{info['b_dtype']} "
            f"shape={info['a_shape']} "
            f"mean_abs={info['mean_abs_diff']:.6g} "
            f"max_abs={info['max_abs_diff']:.6g} "
            f"rmse={info['rmse']:.6g} "
            f"rel_mean={info['rel_mean_abs_diff']:.6g} "
            f"allclose1e-3={info['allclose_1e-3']}"
        )


if __name__ == "__main__":
    main()

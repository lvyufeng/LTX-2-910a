from __future__ import annotations

import argparse
import math
from pathlib import Path

import av
import numpy as np


def _read_video(path: Path) -> np.ndarray:
    frames: list[np.ndarray] = []
    with av.open(str(path), mode="r") as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError(f"{path} has no video stream")
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"{path} decoded zero video frames")
    return np.stack(frames, axis=0)


def _psnr(rmse: float) -> float:
    if rmse == 0:
        return math.inf
    return 20.0 * math.log10(255.0 / rmse)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare decoded RGB frames from two reference MP4 outputs.")
    parser.add_argument("--expected", type=Path, required=True, help="reference video path")
    parser.add_argument("--actual", type=Path, required=True, help="video path to compare")
    parser.add_argument(
        "--max-frame-mean-abs-diff",
        type=_positive_float,
        default=None,
        help="fail if decoded-frame mean absolute RGB difference exceeds this value",
    )
    parser.add_argument(
        "--max-frame-max-abs-diff",
        type=_positive_float,
        default=None,
        help="fail if decoded-frame max absolute RGB difference exceeds this value",
    )
    parser.add_argument("--require-same-shape", action="store_true", help="fail if decoded video arrays differ in shape")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    expected = _read_video(args.expected)
    actual = _read_video(args.actual)

    if args.require_same_shape and expected.shape != actual.shape:
        raise SystemExit(f"shape mismatch: expected {expected.shape}, actual {actual.shape}")

    common_shape = tuple(min(a, b) for a, b in zip(expected.shape, actual.shape, strict=True))
    expected_crop = expected[tuple(slice(0, dim) for dim in common_shape)].astype(np.float32)
    actual_crop = actual[tuple(slice(0, dim) for dim in common_shape)].astype(np.float32)
    diff = np.abs(expected_crop - actual_crop)
    mean_abs = float(diff.mean())
    max_abs = float(diff.max())
    rmse = float(np.sqrt(np.mean(np.square(expected_crop - actual_crop))))

    print(f"expected_shape={expected.shape}")
    print(f"actual_shape={actual.shape}")
    print(f"compared_shape={common_shape}")
    print(f"mean_abs_diff={mean_abs:.6f}")
    print(f"max_abs_diff={max_abs:.6f}")
    print(f"rmse={rmse:.6f}")
    print(f"psnr={_psnr(rmse):.6f}")

    if args.max_frame_mean_abs_diff is not None and mean_abs > args.max_frame_mean_abs_diff:
        raise SystemExit(
            f"mean abs diff {mean_abs:.6f} exceeds threshold {args.max_frame_mean_abs_diff:.6f}"
        )
    if args.max_frame_max_abs_diff is not None and max_abs > args.max_frame_max_abs_diff:
        raise SystemExit(f"max abs diff {max_abs:.6f} exceeds threshold {args.max_frame_max_abs_diff:.6f}")


if __name__ == "__main__":
    main()

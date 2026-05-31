from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import torch

_COUNTERS: defaultdict[str, int] = defaultdict(int)
_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")


def _dump_dir() -> Path | None:
    value = os.environ.get("LTX2_DEBUG_DUMP_DIR")
    if not value:
        return None
    return Path(value).expanduser().resolve()


def _is_rank0() -> bool:
    return os.environ.get("RANK", "0") in {"", "0"}


def should_dump_tensor(name: str) -> bool:
    if _dump_dir() is None or not _is_rank0():
        return False
    filters = os.environ.get("LTX2_DEBUG_DUMP_FILTER")
    if not filters:
        return True
    return any(part and part in name for part in filters.split(","))


def dump_tensor(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is None or not should_dump_tensor(name):
        return

    dump_dir = _dump_dir()
    if dump_dir is None:
        return

    safe_name = _SAFE_CHARS.sub("_", name).strip("_") or "tensor"
    count = _COUNTERS[safe_name]
    _COUNTERS[safe_name] += 1

    prefix = os.environ.get("LTX2_DEBUG_DUMP_PREFIX", "")
    prefix = _SAFE_CHARS.sub("_", prefix).strip("_")
    stem = safe_name if count == 0 else f"{safe_name}_{count:03d}"
    if prefix:
        stem = f"{prefix}_{stem}"

    dump_dir.mkdir(parents=True, exist_ok=True)
    detached = tensor.detach()
    torch.save(
        {
            "name": name,
            "shape": tuple(detached.shape),
            "dtype": str(detached.dtype),
            "device": str(detached.device),
            "tensor": detached.to("cpu"),
        },
        dump_dir / f"{stem}.pt",
    )

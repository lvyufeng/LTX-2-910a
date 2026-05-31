from __future__ import annotations

import gc
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

try:
    import torch_npu  # noqa: F401
except Exception:
    torch_npu = None

HCCS_GROUPS: tuple[tuple[int, ...], ...] = ((0, 1, 2, 3), (4, 5, 6, 7))
DEFAULT_DTYPE = torch.float16
_COMPILE_MODE_CONFIGURED = False
_REMEDIATION = (
    "Source the CANN environment for this installation, for example "
    "`source /usr/local/Ascend/cann-9.0.0/set_env.sh`, and ensure torch_npu "
    "is installed in the active Python environment. Keep the CANN entries that "
    "set_env.sh adds to PYTHONPATH; do not replace PYTHONPATH with project paths only. "
    "The runtime explicitly disables NPU JIT compilation with "
    "torch.npu.set_compile_mode(jit_compile=False)."
)


@dataclass(frozen=True)
class AcceleratorInfo:
    kind: str
    available: bool
    device_count: int


def configure_npu_runtime(device_index: int | None = None) -> None:
    global _COMPILE_MODE_CONFIGURED
    if torch_npu is None or not hasattr(torch, "npu"):
        return
    os.environ.setdefault("ENABLE_ACLNN", "true")
    if not _COMPILE_MODE_CONFIGURED:
        torch.npu.set_compile_mode(jit_compile=False)
        _COMPILE_MODE_CONFIGURED = True
    if device_index is not None and torch.npu.is_available():
        torch.npu.set_device(f"npu:{device_index}")


def _env_snippet(name: str, *, limit: int = 5) -> str:
    value = os.environ.get(name, "")
    if not value:
        return ""
    parts = value.split(os.pathsep)
    if len(parts) <= limit:
        return value
    return os.pathsep.join(parts[:limit]) + os.pathsep + "..."


def _format_exc(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _pythonpath_has(path: str) -> bool:
    target = Path(path).resolve()
    for value in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if not value:
            continue
        try:
            if Path(value).resolve() == target:
                return True
        except OSError:
            if value == path:
                return True
    return False


def _tbe_pythonpath_status() -> tuple[str, str | None]:
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home:
        candidates = [
            Path(ascend_home) / "python" / "site-packages",
            Path(ascend_home) / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe",
        ]
    else:
        candidates = [
            Path("/usr/local/Ascend/cann-9.0.0/python/site-packages"),
            Path("/usr/local/Ascend/cann-9.0.0/opp/built-in/op_impl/ai_core/tbe"),
        ]

    existing = list(dict.fromkeys(path.resolve() for path in candidates if path.exists()))
    if not existing:
        return "WARN", "no known CANN Python paths found"
    missing = [str(path) for path in existing if not _pythonpath_has(str(path))]
    if missing:
        return "WARN", "missing from PYTHONPATH: " + ", ".join(missing)
    return "OK", None


def npu_runtime_diagnostics(device_index: int = 0) -> list[str]:
    """Return human-readable diagnostics for the Ascend NPU runtime."""
    lines: list[str] = []
    if torch_npu is None:
        lines.append("ERROR torch_npu=not importable")
    else:
        lines.append(f"OK torch_npu={getattr(torch_npu, '__version__', 'unknown')}")

    if hasattr(torch, "npu"):
        lines.append("OK torch.npu=present")
    else:
        lines.append("ERROR torch.npu=missing")

    for name in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ENABLE_ACLNN"):
        value = os.environ.get(name, "")
        status = "OK" if value else "WARN"
        lines.append(f"{status} {name}={value}")
    lines.append(f"INFO LD_LIBRARY_PATH={_env_snippet('LD_LIBRARY_PATH')}")
    lines.append(f"INFO PYTHONPATH={_env_snippet('PYTHONPATH')}")
    tbe_path_status, tbe_path_detail = _tbe_pythonpath_status()
    tbe_path_message = "CANN Python paths preserved" if tbe_path_detail is None else tbe_path_detail
    lines.append(f"{tbe_path_status} cann_pythonpath={tbe_path_message}")

    lines.append("INFO npu_jit_compile=false")

    if torch_npu is not None and hasattr(torch, "npu"):
        try:
            available = torch.npu.is_available()
            lines.append(f"OK torch.npu.is_available={available}")
        except Exception as exc:
            lines.append(f"ERROR torch.npu.is_available={_format_exc(exc)}")

        try:
            configure_npu_runtime(device_index)
            lines.append(f"OK configure_npu_runtime=device {device_index}")
        except Exception as exc:
            lines.append(f"ERROR configure_npu_runtime={_format_exc(exc)}")

        try:
            torch.npu.set_device(f"npu:{device_index}")
            lines.append(f"OK torch.npu.set_device=npu:{device_index}")
        except Exception as exc:
            lines.append(f"ERROR torch.npu.set_device={_format_exc(exc)}")

    return lines


def require_npu_runtime(device_index: int = 0) -> None:
    """Raise a helpful error if the Ascend NPU runtime is not ready."""
    diagnostics = npu_runtime_diagnostics(device_index)
    failures = [line for line in diagnostics if line.startswith("ERROR")]
    if failures:
        details = "\n".join(f"  - {line}" for line in failures)
        raise RuntimeError(f"Ascend NPU runtime is not ready:\n{details}\n{_REMEDIATION}")


def get_accelerator() -> AcceleratorInfo:
    if torch_npu is not None and hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                configure_npu_runtime()
                return AcceleratorInfo("npu", True, torch.npu.device_count())
        except Exception:
            pass
    if torch.cuda.is_available():
        return AcceleratorInfo("cuda", True, torch.cuda.device_count())
    return AcceleratorInfo("cpu", True, 0)


def get_default_device(index: int = 0) -> torch.device:
    accelerator = get_accelerator()
    if accelerator.kind == "npu" and accelerator.device_count > 0:
        configure_npu_runtime(index)
        return torch.device("npu", index)
    if accelerator.kind == "cuda" and accelerator.device_count > 0:
        torch.cuda.set_device(index)
        return torch.device("cuda", index)
    return torch.device("cpu")


def synchronize(device: torch.device | None = None) -> None:
    accelerator = get_accelerator()
    if accelerator.kind == "npu":
        torch.npu.synchronize(device)
    elif accelerator.kind == "cuda":
        torch.cuda.synchronize(device)


def empty_cache() -> None:
    accelerator = get_accelerator()
    if accelerator.kind == "npu":
        torch.npu.empty_cache()
    elif accelerator.kind == "cuda":
        torch.cuda.empty_cache()


def cleanup_memory() -> None:
    gc.collect()
    empty_cache()
    synchronize()
    try:
        if hasattr(torch._C, "_host_emptyCache"):
            torch._C._host_emptyCache()
    except Exception:
        pass


def is_npu_device(device: torch.device | str | None) -> bool:
    if device is None:
        return False
    return torch.device(device).type == "npu"


def require_fp16(dtype: torch.dtype | None = None) -> torch.dtype:
    if dtype is None:
        return DEFAULT_DTYPE
    if dtype is not torch.float16:
        raise ValueError("Ascend 910A inference supports float16 only in this runtime.")
    return dtype


def hccs_group_for_device(device_index: int) -> tuple[int, ...]:
    for group in HCCS_GROUPS:
        if device_index in group:
            return group
    raise ValueError(f"device {device_index} is not in known HCCS groups {HCCS_GROUPS}")


def _safe_report(label: str, fn) -> str:
    try:
        return f"{label}={fn()}"
    except Exception as exc:
        return f"{label}_error={_format_exc(exc)}"


def environment_report(device_index: int = 0) -> str:
    lines = [
        f"python={sys.version.split()[0]}",
        f"torch={getattr(torch, '__version__', 'unknown')}",
        f"torch_npu={getattr(torch_npu, '__version__', 'not installed') if torch_npu is not None else 'not installed'}",
        _safe_report("accelerator", get_accelerator),
        _safe_report("default_device", lambda: get_default_device(device_index)),
        f"default_dtype={DEFAULT_DTYPE}",
        f"hccs_groups={HCCS_GROUPS}",
        f"ASCEND_HOME_PATH={os.environ.get('ASCEND_HOME_PATH', '')}",
        f"ASCEND_OPP_PATH={os.environ.get('ASCEND_OPP_PATH', '')}",
        f"ENABLE_ACLNN={os.environ.get('ENABLE_ACLNN', '')}",
    ]
    npu_smi = shutil.which("npu-smi")
    if npu_smi:
        try:
            proc = subprocess.run([npu_smi, "info"], text=True, capture_output=True, timeout=20, check=False)
            lines.append("npu-smi info:\n" + proc.stdout.strip())
            if proc.stderr.strip():
                lines.append("npu-smi stderr:\n" + proc.stderr.strip())
        except Exception as exc:
            lines.append(f"npu-smi error={exc!r}")
    else:
        lines.append("npu-smi=not found")
    return "\n".join(lines)


def parse_device_list(spec: str | None, *, default_count: int | None = 4) -> list[torch.device]:
    accelerator = get_accelerator()
    if spec:
        ids = [int(part.strip()) for part in spec.split(",") if part.strip()]
    elif accelerator.kind == "npu" and accelerator.device_count > 0:
        count = accelerator.device_count if default_count is None else min(default_count, accelerator.device_count)
        ids = list(range(count))
    elif accelerator.kind == "cuda" and accelerator.device_count > 0:
        count = accelerator.device_count if default_count is None else min(default_count, accelerator.device_count)
        ids = list(range(count))
    else:
        return [torch.device("cpu")]
    return [torch.device(accelerator.kind, idx) for idx in ids]


def balanced_block_device_map(num_blocks: int, devices: list[torch.device]) -> dict[int, torch.device]:
    if num_blocks <= 0:
        return {}
    if not devices:
        devices = [get_default_device()]
    return {idx: devices[min(idx * len(devices) // num_blocks, len(devices) - 1)] for idx in range(num_blocks)}

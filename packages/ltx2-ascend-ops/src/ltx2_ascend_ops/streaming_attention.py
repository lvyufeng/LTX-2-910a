from __future__ import annotations

import functools
import importlib
import os
from collections.abc import Callable

import torch

_CUSTOM_OP_ENV = "LTX2_ASCEND_STREAMING_ATTN_OP"
_ENABLE_NATIVE_ENV = "LTX2_ASCEND_STREAMING_ATTN_ENABLE_NATIVE"
_SHAPE_WARMUP_ENV = "LTX2_ASCEND_STREAMING_ATTN_SHAPE_WARMUP"
_LONG_SHAPE_BM16_WARMUP_ENV = "LTX2_ASCEND_STREAMING_ATTN_LONG_BM16_WARMUP"
_BLOCK_M_ENV = "LTX2_ASCEND_STREAMING_ATTN_BLOCK_M"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSY_ENV_VALUES = {"0", "false", "no", "off"}
_DEFAULT_OP_CANDIDATES = (
    "ltx2_ascend.streaming_attention",
    "ltx2_ascend_ops.streaming_attention",
    "ltx2_ascend_ops.streaming_attention_forward",
)


class NativeOpUnavailable(RuntimeError):
    """Raised when the optional AscendC streaming attention op is not installed."""


def _try_import_torch_binding() -> str | None:
    """Import the optional C++ extension that registers torch.ops symbols."""
    try:
        importlib.import_module("ltx2_ascend_ops._streaming_attention_binding")
    except Exception as exc:
        return f"binding import failed: {type(exc).__name__}: {exc}"
    return None


def _candidate_names() -> tuple[str, ...]:
    override = os.getenv(_CUSTOM_OP_ENV, "").strip()
    if override:
        return (override,)
    return _DEFAULT_OP_CANDIDATES


def _lookup_torch_op(name: str) -> Callable | None:
    namespace, sep, op_name = name.partition(".")
    if not sep or not namespace or not op_name:
        return None
    try:
        return getattr(getattr(torch.ops, namespace), op_name)
    except (AttributeError, RuntimeError):
        return None


def _native_op_runtime_error(name: str) -> str | None:
    """Return a runtime availability error for registered binding-backed ops."""
    if name != "ltx2_ascend.streaming_attention":
        return None
    checker = _lookup_torch_op("ltx2_ascend.streaming_attention_is_available")
    if checker is None:
        return "binding availability checker missing"
    try:
        if bool(checker()):
            return None
    except Exception as exc:
        return f"binding availability check failed: {type(exc).__name__}: {exc}"
    return "binding loaded but libcust_opapi.so/aclnnStreamingAttention is unavailable"


@functools.cache
def _resolve_native_op() -> tuple[Callable | None, str]:
    if os.getenv(_ENABLE_NATIVE_ENV, "").strip().lower() not in _TRUTHY_ENV_VALUES:
        return None, f"disabled; set {_ENABLE_NATIVE_ENV}=1 after installing the AscendC op and PyTorch binding"
    binding_error = _try_import_torch_binding()
    checked: list[str] = []
    for name in _candidate_names():
        checked.append(name)
        op = _lookup_torch_op(name)
        if op is None:
            continue
        runtime_error = _native_op_runtime_error(name)
        if runtime_error is None:
            return op, name
        checked[-1] = f"{name} ({runtime_error})"
    detail = "checked " + ", ".join(checked)
    if binding_error is not None:
        detail = f"{detail}; {binding_error}"
    return None, detail


def is_available() -> bool:
    op, _ = _resolve_native_op()
    return op is not None


def availability_report() -> str:
    op, detail = _resolve_native_op()
    if op is None:
        return f"unavailable ({detail})"
    return f"available ({detail})"


_last_shape_warmup_key: tuple[object, ...] | None = None


def _shape_warmup_enabled() -> bool:
    return os.getenv(_SHAPE_WARMUP_ENV, "1").strip().lower() not in _FALSY_ENV_VALUES


def _long_shape_bm16_warmup_enabled() -> bool:
    # 910A CANN Matmul state for very long full-matmul attention is sensitive to
    # the first blockM used for a shape.  A throwaway BM16 call before BM32/BM64
    # keeps T24960 numerics within the fp32-softmax reference tolerance.  This is
    # inside the already opt-in native backend and can be disabled for diagnosis.
    return os.getenv(_LONG_SHAPE_BM16_WARMUP_ENV, "1").strip().lower() not in _FALSY_ENV_VALUES


def _dispatch_key(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, block_m: int, block_n: int) -> tuple[object, ...]:
    return (
        tuple(q.shape),
        tuple(k.shape),
        tuple(v.shape),
        str(q.dtype),
        str(k.dtype),
        str(v.dtype),
        str(q.device),
        int(block_m),
        int(block_n),
        float(scale),
    )


def _dispatch_desc(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, block_m: int, block_n: int) -> str:
    rank = os.getenv("RANK", "?")
    local_rank = os.getenv("LOCAL_RANK", "?")
    return (
        f"rank={rank} local_rank={local_rank} device={q.device} "
        f"q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} "
        f"dtype={q.dtype} scale={scale:.8g} block_m={int(block_m)} block_n={int(block_n)}"
    )


def _synchronize_device(tensor: torch.Tensor) -> None:
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "synchronize"):
        return
    device = tensor.device
    try:
        npu.synchronize(device)
    except TypeError:
        npu.synchronize()


def _ensure_current_device(tensor: torch.Tensor) -> None:
    # ACLNN custom-op launch uses the process' current NPU context in addition to
    # the stream/tensor descriptors.  In the TP pipeline, other subsystems can touch
    # device context before attention dispatch; re-pin to the tensor device so rank
    # local native launches do not fail with aclnn runtime status 361001.
    if tensor.device.type != "npu":
        return
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "set_device"):
        return
    try:
        current = npu.current_device() if hasattr(npu, "current_device") else None
        target = tensor.device.index
        if target is not None and current != target:
            npu.set_device(tensor.device)
    except Exception:
        # Let the actual native call report the concrete failure; this helper should
        # never hide the original fallback/strict behavior.
        return


def _maybe_warmup_shape(op: Callable, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, block_m: int, block_n: int) -> None:
    # The generated ACLNN/CANN Matmul path on 910A has shown a first-call-after-
    # shape-change stale-state hazard under mixed representative timing loops.  A
    # throwaway native call plus stream sync keeps all compute on NPU while forcing
    # the executor/workspace state for the new shape to be initialized before the
    # output that callers observe is produced.  This is only inside the already
    # opt-in native backend; set LTX2_ASCEND_STREAMING_ATTN_SHAPE_WARMUP=0 for
    # diagnosis/benchmarking of raw first-call behavior.
    if not _shape_warmup_enabled():
        return
    global _last_shape_warmup_key
    key = _dispatch_key(q, k, v, scale, block_m, block_n)
    if key == _last_shape_warmup_key:
        return
    if _long_shape_bm16_warmup_enabled() and q.ndim == 4 and int(q.shape[2]) > 8192:
        previous_block_m = os.environ.get(_BLOCK_M_ENV)
        os.environ[_BLOCK_M_ENV] = "16"
        try:
            _ = op(q, k, v, scale, 16, block_n)
            _synchronize_device(q)
        except Exception as exc:
            desc = _dispatch_desc(q, k, v, scale, 16, block_n)
            raise RuntimeError(f"shape-warmup block_m=16 failed ({desc}): {exc}") from exc
        finally:
            if previous_block_m is None:
                os.environ.pop(_BLOCK_M_ENV, None)
            else:
                os.environ[_BLOCK_M_ENV] = previous_block_m
    else:
        try:
            _ = op(q, k, v, scale, block_m, block_n)
            _synchronize_device(q)
        except Exception as exc:
            desc = _dispatch_desc(q, k, v, scale, block_m, block_n)
            raise RuntimeError(f"shape-warmup failed ({desc}): {exc}") from exc
    _last_shape_warmup_key = key


def streaming_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    block_m: int = 0,
    block_n: int = 0,
) -> torch.Tensor:
    """Call the optional native no-mask streaming attention op.

    Inputs and output use BNSD layout ``(B, heads, T, dim_head)``.  The native
    kernel is expected to return a tensor with the same shape, dtype, and device
    as ``q``/``v``.  This function performs no fallback itself; callers in
    ``ltx_core`` catch :class:`NativeOpUnavailable` or native runtime errors and
    fall back to ``AscendChunkedAttention``.
    """
    op, detail = _resolve_native_op()
    if op is None:
        raise NativeOpUnavailable(f"Ascend streaming attention native op unavailable: {detail}")
    _ensure_current_device(q)
    _maybe_warmup_shape(op, q, k, v, scale, block_m, block_n)
    try:
        _ensure_current_device(q)
        return op(q, k, v, scale, block_m, block_n)
    except Exception as exc:
        desc = _dispatch_desc(q, k, v, scale, block_m, block_n)
        raise RuntimeError(f"native call failed ({desc}): {exc}") from exc

from __future__ import annotations

import functools
import importlib
import os
from collections.abc import Callable

import torch

_CUSTOM_OP_ENV = "LTX2_ASCEND_LONGK_SOFTMAX_OP"
_ENABLE_NATIVE_ENV = "LTX2_ASCEND_LONGK_SOFTMAX_NATIVE"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_DEFAULT_OP_CANDIDATES = (
    "ltx2_ascend_longk.long_k_softmax",
    "ltx2_ascend_ops.long_k_softmax",
    "ltx2_ascend_ops.long_k_softmax_forward",
)


class NativeOpUnavailable(RuntimeError):
    """Raised when the optional AscendC long-K softmax op is not installed."""


def _try_import_torch_binding() -> str | None:
    """Import the optional C++ extension that registers torch.ops symbols."""
    try:
        importlib.import_module("ltx2_ascend_ops._long_k_softmax_binding")
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
    if name != "ltx2_ascend_longk.long_k_softmax":
        return None
    checker = _lookup_torch_op("ltx2_ascend_longk.long_k_softmax_is_available")
    if checker is None:
        return "binding availability checker missing"
    try:
        if bool(checker()):
            return None
    except Exception as exc:
        return f"binding availability check failed: {type(exc).__name__}: {exc}"
    return "binding loaded but libcust_opapi.so/aclnnLongKSoftmax is unavailable"


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


def long_k_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Call the optional native long-K softmax op.

    ``scores`` must be a contiguous fp16 tensor in ``(B, heads, Q, K)`` layout.
    The operator computes row max and row sum in fp32 and returns fp16
    probabilities, matching ``torch.softmax(scores.float(), dim=-1).to(scores.dtype)``.
    It is intentionally opt-in; callers should catch :class:`NativeOpUnavailable`
    or native runtime errors and fall back to the standard PyTorch path.
    """
    op, detail = _resolve_native_op()
    if op is None:
        raise NativeOpUnavailable(f"Ascend long-K softmax native op unavailable: {detail}")
    return op(scores)

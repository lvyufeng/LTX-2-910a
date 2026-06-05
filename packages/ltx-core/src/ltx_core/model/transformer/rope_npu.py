"""NPU RoPE backend with an exact PyTorch fallback.

Phase 1 of the Ascend custom-operator work proved that CANN's
``torch_npu.npu_rotary_mul`` (``half`` mode) reproduces LTX
:func:`apply_split_rotary_emb` **bit-for-bit** in fp16 across all real LTX
shapes (video H=32/D=128, audio H=32/D=64, the 4-card TP shard H=8/D=128, T up
to 8192), while running ~1.3-1.6x faster.  Half mode computes
``out = r1*x + r2*cat(-x2, x1)``; SPLIT rope is exactly that with
``r1 = cat([cos, cos])`` and ``r2 = cat([sin, sin])`` over the half-length
``D/2`` cos/sin.  No self-written AscendC kernel is required for RoPE.

This module exposes two drop-in helpers:

* :func:`apply_rotary_emb_backend` for a single tensor.
* :func:`apply_rotary_emb_pair_backend` for the q/k attention pair.  When q and
  k share the same RoPE tensors, it builds the duplicated full-width
  ``r1``/``r2`` tensors once and reuses them for both ``npu_rotary_mul`` calls.

The NPU op is used only when

* ``LTX2_ASCEND_ROPE`` does not explicitly disable it (default ON on NPU),
* the rope type is SPLIT (INTERLEAVED is legacy and not accelerated), and
* the input tensor lives on an NPU device with a supported layout.

Any other case -- disabled backend, non-NPU tensor, unexpected shape, or a
runtime error -- silently falls back to the unchanged PyTorch
:func:`apply_rotary_emb`, so the verified-good fallback is always available.
"""
from __future__ import annotations

import functools
import logging
import os

import torch

from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

logger = logging.getLogger(__name__)

try:
    import torch_npu  # noqa: F401
except Exception:  # pragma: no cover - torch_npu absent on CPU/GPU hosts
    torch_npu = None

_DISABLED_BACKENDS = {"eager", "math", "off", "disable", "disabled", "0", "false", "no"}
_PreparedRotary = tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]


@functools.cache
def _rope_backend() -> str:
    """Resolve the RoPE backend from ``LTX2_ASCEND_ROPE`` (cached once per process).

    Default is ``npu`` on NPU: Phase 4 HQ A/B showed the CANN op is bit-exact and
    faster.  ``eager``/``off``/``0``/``false`` force the PyTorch fallback for
    debugging; ``npu``/``rotary_mul``/``custom`` all select ``npu_rotary_mul``.
    """
    value = os.getenv("LTX2_ASCEND_ROPE", "").strip().lower()
    if value in _DISABLED_BACKENDS:
        return "eager"
    return "npu"


def _prepare_npu_rotary_input(
    input_tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> _PreparedRotary | None:
    """Map LTX SPLIT rope layout to ``npu_rotary_mul`` input layout.

    ``input_tensor`` may be 3D ``(B, T, H*D)`` (the attention call sites) or 4D
    ``(B, H, T, D)``; ``cos``/``sin`` are ``(B, H, T, D/2)``.  Returns ``None``
    on any shape it cannot map exactly.
    """
    if torch_npu is None:
        return None
    if cos.ndim != 4 or sin.shape != cos.shape:
        return None

    heads = cos.shape[1]
    seq = cos.shape[2]
    half = cos.shape[-1]
    head_dim = 2 * half

    if input_tensor.ndim == 4:
        if input_tensor.shape[1] != heads or input_tensor.shape[2] != seq:
            return None
        if input_tensor.shape[-1] != head_dim:
            return None
        x = input_tensor
        reshaped = False
    elif input_tensor.ndim == 3:
        if input_tensor.shape[1] != seq:
            return None
        if input_tensor.shape[-1] != heads * head_dim:
            return None
        # Match apply_split_rotary_emb's adapter exactly: (B,T,H*D) -> (B,H,T,D).
        x = input_tensor.unflatten(-1, (heads, head_dim)).transpose(1, 2)
        reshaped = True
    else:
        return None

    # LTX allows a broadcast cos/sin batch of 1; npu_rotary_mul wants matching dims.
    if cos.shape[0] == 1 and x.shape[0] != 1:
        cos = cos.expand(x.shape[0], -1, -1, -1)
        sin = sin.expand(x.shape[0], -1, -1, -1)
    elif cos.shape[0] != x.shape[0]:
        return None

    return x.contiguous(), cos, sin, reshaped


def _duplicated_freqs(cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.cat([cos, cos], dim=-1).contiguous(), torch.cat([sin, sin], dim=-1).contiguous()


def _apply_npu_rotary_prepared(
    prepared: _PreparedRotary,
    r1: torch.Tensor,
    r2: torch.Tensor,
) -> torch.Tensor:
    x, _, _, reshaped = prepared
    out = torch_npu.npu_rotary_mul(x, r1, r2, "half")
    if reshaped:
        out = out.transpose(1, 2).flatten(-2)
    return out


def _npu_rotary_split(
    input_tensor: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor | None:
    """SPLIT rope via ``npu_rotary_mul`` half mode, or ``None`` if unsupported."""
    prepared = _prepare_npu_rotary_input(input_tensor, cos, sin)
    if prepared is None:
        return None
    _, prepared_cos, prepared_sin, _ = prepared
    r1, r2 = _duplicated_freqs(prepared_cos, prepared_sin)
    return _apply_npu_rotary_prepared(prepared, r1, r2)


def apply_rotary_emb_backend(
    input_tensor: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
) -> torch.Tensor:
    """Drop-in :func:`apply_rotary_emb` with a default-on NPU fast path.

    Falls back to the exact PyTorch implementation for any non-SPLIT rope,
    non-NPU tensor, unsupported shape, or runtime error.
    """
    if _rope_backend() == "npu" and rope_type == LTXRopeType.SPLIT and input_tensor.device.type == "npu":
        cos, sin = freqs_cis
        try:
            out = _npu_rotary_split(input_tensor, cos, sin)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("npu_rotary_mul rope failed (%s); falling back to eager", exc)
            out = None
        if out is not None:
            return out
    return apply_rotary_emb(input_tensor, freqs_cis, rope_type)


def apply_rotary_emb_pair_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: tuple[torch.Tensor, torch.Tensor],
    k_freqs_cis: tuple[torch.Tensor, torch.Tensor] | None = None,
    rope_type: LTXRopeType = LTXRopeType.SPLIT,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to q/k, reusing duplicated frequencies when they share RoPE.

    Attention call sites usually apply the same ``pe`` to q and k.  The single
    backend must build ``cat([cos, cos])`` and ``cat([sin, sin])`` for each call;
    this pair helper builds them once when q/k share the same tensors, then uses
    the same full-width frequencies for both ``npu_rotary_mul`` invocations.
    """
    k_freqs_cis = freqs_cis if k_freqs_cis is None else k_freqs_cis
    if (
        _rope_backend() == "npu"
        and rope_type == LTXRopeType.SPLIT
        and q.device.type == "npu"
        and k.device.type == "npu"
    ):
        cos, sin = freqs_cis
        k_cos, k_sin = k_freqs_cis
        if cos is k_cos and sin is k_sin:
            try:
                q_prepared = _prepare_npu_rotary_input(q, cos, sin)
                k_prepared = _prepare_npu_rotary_input(k, cos, sin)
                if q_prepared is not None and k_prepared is not None:
                    _, q_cos, q_sin, _ = q_prepared
                    _, k_cos_prepared, k_sin_prepared, _ = k_prepared
                    if q_cos.shape == k_cos_prepared.shape and q_sin.shape == k_sin_prepared.shape:
                        r1, r2 = _duplicated_freqs(q_cos, q_sin)
                        return (
                            _apply_npu_rotary_prepared(q_prepared, r1, r2),
                            _apply_npu_rotary_prepared(k_prepared, r1, r2),
                        )
            except Exception as exc:  # pragma: no cover - defensive fallback
                logger.debug("paired npu_rotary_mul rope failed (%s); falling back", exc)

    return (
        apply_rotary_emb_backend(q, freqs_cis, rope_type),
        apply_rotary_emb_backend(k, k_freqs_cis, rope_type),
    )

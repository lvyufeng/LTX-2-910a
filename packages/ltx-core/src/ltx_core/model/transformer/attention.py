import atexit
import functools
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from ltx_core.accelerator import synchronize
from ltx_core.model.transformer.ops import (
    GatedAttentionCallable,
    PreAttentionCallable,
    PytorchGatedAttention,
    PytorchPreAttention,
)
from ltx_core.model.transformer.rope import LTXRopeType

logger = logging.getLogger(__name__)

_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_PROFILE_DETAIL_ENV = "LTX2_ASCEND_PROFILE_DETAIL"
_PROFILE_STATS: dict[str, list[float]] = {}
_PROFILE_REGISTERED = False
_ASCEND_STREAMING_ATTENTION_VALUES = {"streaming", "custom"}
_ASCEND_CHUNKED_ATTENTION_VALUES = {"eager", "math"}
_ASCEND_ATTENTION_ENV = "LTX2_ASCEND_ATTENTION"
_ASCEND_STREAMING_ATTN_MIN_T_ENV = "LTX2_ASCEND_STREAMING_ATTN_MIN_T"
_ASCEND_STREAMING_ATTN_BLOCK_M_ENV = "LTX2_ASCEND_STREAMING_ATTN_BLOCK_M"
_ASCEND_STREAMING_ATTN_BLOCK_N_ENV = "LTX2_ASCEND_STREAMING_ATTN_BLOCK_N"
_ASCEND_STREAMING_ATTN_STRICT_ENV = "LTX2_ASCEND_STREAMING_ATTN_STRICT"
_ASCEND_STREAMING_ATTN_LOG_FALLBACK_ENV = "LTX2_ASCEND_STREAMING_ATTN_LOG_FALLBACK"
_ASCEND_ATTENTION_EAGER_MAX_MB_ENV = "LTX2_ASCEND_ATTENTION_EAGER_MAX_MB"
_ASCEND_ATTENTION_CHUNK_MAX_MB_ENV = "LTX2_ASCEND_ATTENTION_CHUNK_MAX_MB"
_ASCEND_ATTENTION_TRACE_ENV = "LTX2_ASCEND_ATTENTION_TRACE"
_ASCEND_SCALED_MASKED_SOFTMAX_ENV = "LTX2_ASCEND_SCALED_MASKED_SOFTMAX"
_ASCEND_SCALED_MASKED_SOFTMAX_MIN_T_ENV = "LTX2_ASCEND_SCALED_MASKED_SOFTMAX_MIN_T"
_ASCEND_LONGK_SOFTMAX_NATIVE_ENV = "LTX2_ASCEND_LONGK_SOFTMAX_NATIVE"
_ASCEND_LONGK_SOFTMAX_MIN_K_ENV = "LTX2_ASCEND_LONGK_SOFTMAX_MIN_K"
_ASCEND_TP_HQ_SHAPE_POLICY_ENV = "LTX2_ASCEND_TP_HQ_SHAPE_POLICY"
_STREAMING_SUPPORTED_HEAD_DIMS = {64, 128}
_STREAMING_DEFAULT_MIN_T = 1


def _env_truthy(name: str) -> bool:
    return os.getenv(name, "").strip().lower() in _TRUTHY_ENV_VALUES


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("ignoring invalid %s=%r; expected integer", name, value)
        return default


def _detail_profile_enabled() -> bool:
    return os.getenv(_PROFILE_DETAIL_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES


def _record_profile(name: str, elapsed: float) -> None:
    stats = _PROFILE_STATS.setdefault(name, [0.0, 0.0])
    stats[0] += 1.0
    stats[1] += elapsed


def _emit_profile_summary() -> None:
    if not _PROFILE_STATS or os.environ.get("RANK", "0") != "0":
        return
    for name, (count, total) in sorted(_PROFILE_STATS.items(), key=lambda item: (-item[1][1], item[0])):
        avg = total / count if count else 0.0
        logger.info("[profile-detail-attn] %s count=%d total=%.3fs avg=%.6fs", name, int(count), total, avg)


def _ensure_profile_registered() -> None:
    global _PROFILE_REGISTERED
    if _PROFILE_REGISTERED:
        return
    atexit.register(_emit_profile_summary)
    _PROFILE_REGISTERED = True


@contextmanager
def _profile_detail(name: str, device: torch.device | None = None) -> Iterator[None]:
    if not _detail_profile_enabled():
        yield
        return
    _ensure_profile_registered()
    synchronize(device)
    start = time.perf_counter()
    try:
        yield
    finally:
        synchronize(device)
        _record_profile(name, time.perf_counter() - start)


def _shape_profile_key(prefix: str, b: int, heads: int, q_len: int, k_len: int, dim_head: int, chunk: int | None) -> str:
    chunk_part = "eager" if chunk is None else f"chunk{chunk}"
    return f"{prefix}.B{b}.H{heads}.Q{q_len}.K{k_len}.D{dim_head}.{chunk_part}"


def _attention_trace(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    heads: int,
    mask: torch.Tensor | None,
    *,
    fallback_reason: str | None = None,
) -> None:
    if not _env_truthy(_ASCEND_ATTENTION_TRACE_ENV):
        return
    try:
        dim_head = q.shape[-1] // heads if heads else 0
        logger.info(
            "attention trace backend=%s b=%s heads=%s q_len=%s k_len=%s dim_head=%s dtype=%s mask=%s fallback=%s",
            backend,
            q.shape[0] if q.ndim >= 1 else "?",
            heads,
            q.shape[1] if q.ndim >= 2 else "?",
            k.shape[1] if k.ndim >= 2 else "?",
            dim_head,
            q.dtype,
            mask is not None,
            fallback_reason or "",
        )
    except Exception:  # pragma: no cover - tracing must never affect inference.
        logger.debug("failed to emit attention trace", exc_info=True)


def _torch_default_sdpa_priority() -> list[SDPBackend]:
    """Fetch torch's current default SDPA priority order at runtime.
    Used as the default for ``PytorchAttention`` so the wrapper-always
    code path matches torch's native dispatch order without hard-coding it
    (which would drift if torch updates the default).
    ``torch._C._get_sdp_priority_order`` is a private API; we accept that
    risk because the project pins ``torch`` in the lockfile, so any
    rename/removal surfaces on a controlled torch bump rather than silently.
    """
    return [SDPBackend(p) for p in torch._C._get_sdp_priority_order()]


memory_efficient_attention = None
flash_attn_interface = None
flash_attn_4_func = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None
try:
    from flash_attn.cute import flash_attn_func as flash_attn_4_func
except ImportError:
    flash_attn_4_func = None


class AttentionCallable(Protocol):
    """Unmasked attention. Backends without a mask kernel (FA3/FA4) implement only
    this protocol; backends that support masks too (Pytorch/SDPA, xFormers) are
    structurally usable here and as :class:`MaskedAttentionCallable`."""

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int) -> torch.Tensor: ...


class MaskedAttentionCallable(Protocol):
    """Masked attention. Mask is required (not optional) -- the caller has already
    decided this is the masked path and chosen a backend that can serve it. Used
    by :class:`Attention` when its forward receives a non-None ``mask``."""

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __init__(self, priority: list[SDPBackend] | None = None) -> None:
        # priority=None -> snapshot torch's default SDPA priority at construction.
        # Always passed through ``sdpa_kernel(..., set_priority=True)`` so the
        # call site is uniform regardless of how the priority was chosen.
        self._priority = priority if priority is not None else _torch_default_sdpa_priority()

    @property
    def label(self) -> str:
        """Human-readable identifier (used in the AUTOMATIC selection log).
        Encodes the SDPA priority list so a single-backend pin reads differently
        from the full-priority dispatcher walk."""
        return f"SDPA[{'>'.join(b.name for b in self._priority)}]"

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        with sdpa_kernel(self._priority, set_priority=True):
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False
            )
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class AscendEagerAttention(AttentionCallable):
    label = "AscendEager"

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
        scale = dim_head**-0.5
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            scores = scores + mask.to(device=scores.device, dtype=scores.dtype)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        out = torch.matmul(probs, v)
        return out.transpose(1, 2).reshape(b, -1, heads * dim_head)


class AscendFusedAttention(AttentionCallable):
    """Fused FlashAttention via ``torch_npu.npu_fusion_attention``.

    Uses the hardware-tiled kernel that fuses Q×K^T scaling, masking, softmax,
    and ×V into a single call without materializing the full (B,H,S,S) score
    matrix.  Falls back to :class:`AscendChunkedAttention` if the kernel is
    unavailable or raises at runtime (e.g. unsupported head_dim).

    Env knobs:
      LTX2_ASCEND_ATTENTION=fused   — select this backend explicitly
      LTX2_ASCEND_FA_INNER_PRECISE  — inner_precise flag (default 0)
    """

    label = "AscendFused"

    def __init__(self) -> None:
        self._inner_precise = int(os.getenv("LTX2_ASCEND_FA_INNER_PRECISE", "0"))
        self._fallback: AscendChunkedAttention | None = None

    def _get_fallback(self) -> "AscendChunkedAttention":
        if self._fallback is None:
            self._fallback = AscendChunkedAttention()
        return self._fallback

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        import torch_npu  # noqa: F401 — registers npu_fusion_attention

        b, q_len, total_dim = q.shape
        dim_head = total_dim // heads
        # npu_fusion_attention expects BNSD: (B, heads, S, dim_head)
        q4d = q.view(b, q_len, heads, dim_head).transpose(1, 2).contiguous()
        k4d = k.view(b, -1, heads, dim_head).transpose(1, 2).contiguous()
        v4d = v.view(b, -1, heads, dim_head).transpose(1, 2).contiguous()

        scale = dim_head**-0.5

        # Convert additive float mask to boolean mask for the fused kernel.
        # npu_fusion_attention convention: 1 = masked (position ignored), 0 = attend.
        # Our additive mask uses large negative values (e.g. -inf or -1e9) for masked positions.
        atten_mask = None
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # Convert: positions with large negative additive bias → True (masked)
            atten_mask = (mask < -1.0).to(torch.bool)

        try:
            out, *_ = torch_npu.npu_fusion_attention(
                q4d,
                k4d,
                v4d,
                heads,
                "BNSD",
                pse=None,
                padding_mask=None,
                atten_mask=atten_mask,
                scale=scale,
                keep_prob=1.0,
                pre_tockens=2147483647,
                next_tockens=2147483647,
                inner_precise=self._inner_precise,
            )
        except RuntimeError:
            # Fallback for unsupported shapes/dtypes
            return self._get_fallback()(q, k, v, heads, mask)

        # out is (B, heads, S, dim_head) in BNSD layout
        return out.transpose(1, 2).reshape(b, q_len, heads * dim_head)


class AscendStreamingAttention(AttentionCallable):
    """Optional custom Ascend streaming attention with exact chunked fallback.

    The native AscendC kernel is intentionally optional.  This wrapper is safe to
    select before the custom op is built: every unsupported shape, missing symbol,
    or runtime failure falls back to :class:`AscendChunkedAttention` unless strict
    mode is requested for validation.

    Env knobs:
      LTX2_ASCEND_ATTENTION=streaming       — select this backend explicitly
      LTX2_ASCEND_STREAMING_ATTN_MIN_T      — minimum q/k length for dispatch
      LTX2_ASCEND_STREAMING_ATTN_BLOCK_M/N  — tile hints passed to the native op
      LTX2_ASCEND_STREAMING_ATTN_STRICT=1   — raise instead of falling back
      LTX2_ASCEND_STREAMING_ATTN_LOG_FALLBACK=1 — log fallback reasons once
    """

    label = "AscendStreaming"

    def __init__(self) -> None:
        self._fallback: AscendChunkedAttention | None = None
        self._logged_fallbacks: set[str] = set()

    def _get_fallback(self) -> "AscendChunkedAttention":
        if self._fallback is None:
            self._fallback = AscendChunkedAttention()
        return self._fallback

    @property
    def _strict(self) -> bool:
        return _env_truthy(_ASCEND_STREAMING_ATTN_STRICT_ENV)

    def _fallback_call(
        self,
        reason: str,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None,
    ) -> torch.Tensor:
        _attention_trace(self.label, q, k, v, heads, mask, fallback_reason=reason)
        if self._strict:
            raise RuntimeError(f"Ascend streaming attention unsupported: {reason}")
        if _env_truthy(_ASCEND_STREAMING_ATTN_LOG_FALLBACK_ENV) and reason not in self._logged_fallbacks:
            logger.info("Ascend streaming attention falling back to chunked attention: %s", reason)
            self._logged_fallbacks.add(reason)
        return self._get_fallback()(q, k, v, heads, mask)

    def _unsupported_reason(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None,
    ) -> str | None:
        if mask is not None:
            return "mask-present"
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            return "expected-3d-qkv"
        if heads <= 0:
            return "invalid-head-count"
        if q.device.type != "npu" or k.device.type != "npu" or v.device.type != "npu":
            return "non-npu-tensor"
        if q.device != k.device or q.device != v.device:
            return "mixed-devices"
        if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
            return "non-fp16-qkv"
        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            return "batch-mismatch"
        if k.shape[1] != v.shape[1]:
            return "kv-length-mismatch"
        if q.shape[1] != k.shape[1]:
            return "non-square-attention"
        if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
            return "inner-dim-mismatch"
        if q.shape[-1] % heads != 0:
            return "inner-dim-not-divisible-by-heads"
        dim_head = q.shape[-1] // heads
        if dim_head not in _STREAMING_SUPPORTED_HEAD_DIMS:
            return f"unsupported-head-dim-{dim_head}"
        min_t = _env_int(_ASCEND_STREAMING_ATTN_MIN_T_ENV, _STREAMING_DEFAULT_MIN_T)
        if q.shape[1] < min_t:
            return f"below-min-t-{min_t}"
        if q.shape[1] % 16 != 0:
            return "seq-len-not-multiple-of-16"
        return None

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        reason = self._unsupported_reason(q, k, v, heads, mask)
        if reason is not None:
            return self._fallback_call(reason, q, k, v, heads, mask)

        block_m = _env_int(_ASCEND_STREAMING_ATTN_BLOCK_M_ENV, 0)
        block_n = _env_int(_ASCEND_STREAMING_ATTN_BLOCK_N_ENV, 0)
        b, q_len, total_dim = q.shape
        dim_head = total_dim // heads
        scale = dim_head**-0.5

        try:
            from ltx2_ascend_ops.streaming_attention import streaming_attention
        except Exception as exc:  # pragma: no cover - depends on optional package.
            return self._fallback_call(f"native-op-unavailable:{type(exc).__name__}", q, k, v, heads, mask)

        key = _shape_profile_key("streaming", b, heads, q_len, q_len, dim_head, None)
        try:
            with _profile_detail(f"{key}.layout", q.device):
                q4d = q.view(b, q_len, heads, dim_head).transpose(1, 2).contiguous()
                k4d = k.view(b, q_len, heads, dim_head).transpose(1, 2).contiguous()
                v4d = v.view(b, q_len, heads, dim_head).transpose(1, 2).contiguous()
            with _profile_detail(f"{key}.op", q.device):
                out = streaming_attention(q4d, k4d, v4d, scale=scale, block_m=block_m, block_n=block_n)
            _attention_trace(self.label, q, k, v, heads, mask)
        except Exception as exc:  # pragma: no cover - depends on optional native op.
            message = str(exc).splitlines()[0][:240]
            detail = f":{message}" if message else ""
            return self._fallback_call(f"native-op-runtime:{type(exc).__name__}{detail}", q, k, v, heads, mask)

        if out.shape != q4d.shape:
            return self._fallback_call("native-op-output-shape", q, k, v, heads, mask)
        if out.dtype != v.dtype or out.device != v.device:
            return self._fallback_call("native-op-output-dtype-device", q, k, v, heads, mask)
        # torch_npu can mishandle reshape directly on the non-contiguous BNSD->BT(H*D)
        # transpose view after custom-op output on larger shapes.  Materialize the
        # layout conversion on NPU before flattening; this keeps compute on-device and
        # matches the benchmark helper's exact BNSD unpack path.
        with _profile_detail(f"{key}.out_layout", q.device):
            return out.transpose(1, 2).contiguous().view(b, q_len, heads * dim_head)


def _ascend_softmax_fp16() -> bool:
    """When True, keep softmax in the input dtype (fp16) instead of upcasting to fp32.

    On Ascend 910A the fp32 upcast is the dominant attention cost (~14× slower)
    because the score matrix must be copied to/from fp32 through HBM.  The
    precision difference is negligible (max ~2.4e-4, mean ~1e-5) and the
    transformer already runs in fp16 with precision-sensitive paths protected
    elsewhere (embeddings fp32, guidance combine fp32, VAE decoder fp32).

    Enable via ``LTX2_ASCEND_SOFTMAX_FP16=1``.
    """
    return os.getenv("LTX2_ASCEND_SOFTMAX_FP16", "").lower() in {"1", "true", "yes", "on"}


def _ascend_scaled_masked_softmax_disabled() -> bool:
    """Return True unless the experimental scaled-masked-softmax path is opted in.

    Focused TP-HQ sweeps on 910A showed ``torch_npu.npu_scaled_masked_softmax`` is
    not a safe default for the real rank-local shapes: K=6240 produced large output
    corruption (max abs ~12) and K=128 was non-bit-identical while also slower.
    Keep the mathematically stable ``torch.softmax(scores.float())`` path as the
    default; allow this primitive only via the explicit opt-in
    ``LTX2_ASCEND_SCALED_MASKED_SOFTMAX=1`` for future experiments.
    """
    return os.getenv(_ASCEND_SCALED_MASKED_SOFTMAX_ENV, "").strip().lower() not in _TRUTHY_ENV_VALUES


def _ascend_scaled_masked_softmax_min_t() -> int:
    # Experimental path only; default-off via _ascend_scaled_masked_softmax_disabled.
    return max(1, _env_int(_ASCEND_SCALED_MASKED_SOFTMAX_MIN_T_ENV, 2048))


def _ascend_longk_softmax_enabled() -> bool:
    # Experimental native vector-only softmax path. Keep default-off until it is
    # numerically validated and faster in the 4-card TP-HQ good path.
    return os.getenv(_ASCEND_LONGK_SOFTMAX_NATIVE_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES


def _ascend_longk_softmax_min_k() -> int:
    return max(1, _env_int(_ASCEND_LONGK_SOFTMAX_MIN_K_ENV, 8192))


def _tp_hq_shape_policy_enabled() -> bool:
    # Default-off in the backend: the CLI enables it only for the validated TP-HQ
    # default path and leaves explicit user chunk overrides authoritative.
    return os.getenv(_ASCEND_TP_HQ_SHAPE_POLICY_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES


def _tp_hq_chunk_for_shape(
    b: int,
    heads: int,
    q_len: int,
    k_len: int,
    dim_head: int,
    requested_chunk: int,
) -> int:
    """Shape-aware TP-HQ chunk policy for validated rank-local self-attention.

    The generic TP-HQ CLI default remains chunk1536 for non-fp16-softmax paths
    because stage-1 self-attention (Q=K=6240,D=128,H=8) regresses with the old
    fp32-upcast softmax at larger chunks.  With the scoped TP-HQ fp16-softmax
    default, the same stage-1 shape is bit-identical across chunks and faster at
    chunk4096/eager.  The dominant stage-2 self shape (Q=K=24960,D=128,H=8)
    remains consistently bit-identical and fastest at chunk2048.  Apply this
    only when the caller is using the scoped TP-HQ default chunk1536; any explicit
    ``LTX2_ASCEND_ATTENTION_CHUNK`` override remains authoritative.
    """
    if requested_chunk != 1536:
        return requested_chunk
    is_tp_hq_self = b == 1 and heads == 8 and dim_head == 128 and q_len == k_len
    if is_tp_hq_self and q_len >= 20000:
        return 2048
    if is_tp_hq_self and q_len == 6240 and _ascend_softmax_fp16():
        return 4096
    return requested_chunk


class AscendChunkedAttention(AttentionCallable):
    label = "AscendChunked"

    def __init__(self, query_chunk_size: int | None = None, eager_max_mb: int | None = None) -> None:
        # 4096 is a measured, bit-exact speedup for representative long 910A attention
        # (T8192 D128: chunk512 ~27.8ms -> chunk4096 ~22.0ms, identical output to
        # smaller chunks).  Keep LTX2_ASCEND_ATTENTION_CHUNK as the explicit
        # fallback/tuning switch.
        self.query_chunk_size = query_chunk_size or int(os.getenv("LTX2_ASCEND_ATTENTION_CHUNK", "4096"))
        if eager_max_mb is None:
            eager_max_mb = int(os.getenv(_ASCEND_ATTENTION_EAGER_MAX_MB_ENV, "1024"))
        self.eager_max_bytes = eager_max_mb * 1024 * 1024
        # The per-chunk fp32 score working set is capped independently from the full
        # eager threshold.  This lets TP-HQ keep full eager conservative while raising
        # only the chunk-cap budget so T24960/D128 can use the validated chunk1536.
        chunk_max_mb = int(os.getenv(_ASCEND_ATTENTION_CHUNK_MAX_MB_ENV, str(eager_max_mb)))
        self.chunk_max_bytes = chunk_max_mb * 1024 * 1024
        # Using cat instead of indexed output writes is bit-exact and modestly faster
        # on long chunked attention. Cap the default to small outputs to avoid
        # retaining large per-chunk tensors in memory; env remains the fallback knob.
        self.cat_output_max_bytes = int(os.getenv("LTX2_ASCEND_ATTENTION_CAT_MAX_MB", "32")) * 1024 * 1024
        self._scaled_masked_softmax_disabled = _ascend_scaled_masked_softmax_disabled()
        self._scaled_masked_softmax_min_t = _ascend_scaled_masked_softmax_min_t()
        self._scaled_masked_softmax_op = None
        self._scaled_masked_softmax_unavailable = False
        self._scaled_masked_softmax_masks: dict[tuple[str, tuple[int, ...]], torch.Tensor] = {}
        # Optional native vector-only long-K softmax op (default-off). Resolved
        # lazily on first eligible call so a missing/broken op never affects the
        # default fp32-softmax path.
        self._longk_softmax_enabled = _ascend_longk_softmax_enabled()
        self._longk_softmax_min_k = _ascend_longk_softmax_min_k()
        self._longk_softmax_op = None
        self._longk_softmax_unavailable = False

    def _should_use_scaled_masked_softmax(self, scores: torch.Tensor) -> bool:
        if self._scaled_masked_softmax_disabled or self._scaled_masked_softmax_unavailable:
            return False
        if scores.device.type != "npu" or scores.ndim != 4:
            return False
        if scores.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            return False
        key_len = scores.shape[-1]
        # CANN 9.0.0 ScaledMaskedSoftmax ND compile contract on 910A requires
        # the last dimension to be in [32, 8192] and divisible by 32.  Violations
        # fail asynchronously at synchronize time, so avoid dispatch entirely.
        if key_len < 32 or key_len > 8192 or key_len % 32 != 0:
            return False
        min_t = self._scaled_masked_softmax_min_t
        return scores.shape[-2] >= min_t and key_len >= min_t

    def _get_scaled_masked_softmax_mask(self, scores: torch.Tensor) -> torch.Tensor:
        mask_shape = (1, 1, scores.shape[-2], scores.shape[-1])
        key = (str(scores.device), mask_shape)
        mask = self._scaled_masked_softmax_masks.get(key)
        if mask is None or mask.device != scores.device:
            # Keep batch/head broadcast dimensions so the mask does not scale with
            # the number of attention heads.  910A requires the query dimension to
            # be present for long-K shapes; a scalar/key-only mask is not equivalent
            # on all probed shapes.
            if len(self._scaled_masked_softmax_masks) >= 4:
                self._scaled_masked_softmax_masks.clear()
            mask = torch.zeros(mask_shape, device=scores.device, dtype=torch.bool)
            self._scaled_masked_softmax_masks[key] = mask
        return mask

    def _should_use_longk_softmax(self, scores: torch.Tensor, out_dtype: torch.dtype) -> bool:
        if not self._longk_softmax_enabled or self._longk_softmax_unavailable:
            return False
        if scores.device.type != "npu" or scores.ndim != 4:
            return False
        if scores.dtype != torch.float16 or out_dtype != torch.float16:
            return False
        key_len = scores.shape[-1]
        # v1 is a vector-only no-mask softmax over a contiguous fp16 B,H,Q,K score
        # tensor. It is aimed at TP-HQ long-K self-attention chunks where K=24960.
        if key_len < self._longk_softmax_min_k or key_len % 16 != 0:
            return False
        return scores.is_contiguous()

    def _longk_softmax(self, scores: torch.Tensor) -> torch.Tensor:
        if self._longk_softmax_op is None:
            from ltx2_ascend_ops.long_k_softmax import long_k_softmax  # noqa: PLC0415

            self._longk_softmax_op = long_k_softmax
        return self._longk_softmax_op(scores)

    def _softmax(
        self,
        scores: torch.Tensor,
        out_dtype: torch.dtype,
        *,
        allow_scaled_masked_softmax: bool = True,
    ) -> torch.Tensor:
        """Softmax with optional fp32 upcast (default) or native-dtype fast path."""
        if _ascend_softmax_fp16():
            return torch.softmax(scores, dim=-1)
        if allow_scaled_masked_softmax and self._should_use_scaled_masked_softmax(scores):
            try:
                if self._scaled_masked_softmax_op is None:
                    import torch_npu  # noqa: PLC0415

                    self._scaled_masked_softmax_op = torch_npu.npu_scaled_masked_softmax
                x = scores.float()
                mask = self._get_scaled_masked_softmax_mask(x)
                return self._scaled_masked_softmax_op(x, mask, 1.0, False).to(dtype=out_dtype)
            except (ImportError, AttributeError, RuntimeError):
                self._scaled_masked_softmax_unavailable = True
                logger.debug("falling back from npu_scaled_masked_softmax", exc_info=True)
        if allow_scaled_masked_softmax and self._should_use_longk_softmax(scores, out_dtype):
            try:
                return self._longk_softmax(scores)
            except (ImportError, AttributeError, RuntimeError):
                self._longk_softmax_unavailable = True
                logger.debug("falling back from native long-K softmax", exc_info=True)
        return torch.softmax(scores.float(), dim=-1).to(dtype=out_dtype)

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, q_len, total_dim = q.shape
        dim_head = total_dim // heads
        with _profile_detail(_shape_profile_key("chunked.layout", b, heads, q_len, k.shape[1], dim_head, 0), q.device):
            q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
        k_len = k.shape[-2]
        scale = dim_head**-0.5

        if mask is not None:
            with _profile_detail(_shape_profile_key("chunked.mask", b, heads, q_len, k_len, dim_head, 0), q.device):
                if mask.ndim == 2:
                    mask = mask.unsqueeze(0)
                if mask.ndim == 3:
                    mask = mask.unsqueeze(1)
                mask = mask.to(device=q.device, dtype=q.dtype)

        # Small attention fits comfortably and is faster as one eager matmul.
        if b * heads * q_len * k_len * torch.finfo(torch.float32).bits // 8 <= self.eager_max_bytes:
            key = _shape_profile_key("chunked.eager", b, heads, q_len, k_len, dim_head, None)
            with _profile_detail(f"{key}.qk", q.device):
                scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if mask is not None:
                with _profile_detail(f"{key}.mask_add", q.device):
                    scores = scores + mask
            with _profile_detail(f"{key}.softmax", q.device):
                probs = self._softmax(scores, v.dtype, allow_scaled_masked_softmax=mask is None)
            with _profile_detail(f"{key}.pv", q.device):
                out = torch.matmul(probs, v)
            with _profile_detail(f"{key}.out_layout", q.device):
                return out.transpose(1, 2).reshape(b, q_len, heads * dim_head)

        with _profile_detail(_shape_profile_key("chunked.kt", b, heads, q_len, k_len, dim_head, 0), q.device):
            kt = k.transpose(-2, -1)
        chunk = max(1, self.query_chunk_size)
        if _tp_hq_shape_policy_enabled():
            chunk = _tp_hq_chunk_for_shape(b, heads, q_len, k_len, dim_head, chunk)
        # Each chunk's score is upcast to fp32 inside _softmax (b*heads*chunk*k_len*4
        # bytes). Cap the effective chunk against chunk_max_bytes so very long key
        # lengths shrink the chunk instead of materializing a multi-GB fp32 transient
        # (TP memory safety).  Keep this separate from the full eager threshold: the
        # latter decides algorithm shape; this one only caps chunked working-set size.
        score_row_bytes = b * heads * k_len * (torch.finfo(torch.float32).bits // 8)
        if score_row_bytes > 0:
            chunk = min(chunk, max(1, self.chunk_max_bytes // score_row_bytes))
        out_bytes = b * heads * q_len * dim_head * torch.finfo(v.dtype).bits // 8
        key = _shape_profile_key("chunked.loop", b, heads, q_len, k_len, dim_head, chunk)
        if out_bytes <= self.cat_output_max_bytes:
            chunks = []
            for start in range(0, q_len, chunk):
                end = min(start + chunk, q_len)
                with _profile_detail(f"{key}.qk", q.device):
                    scores = torch.matmul(q[:, :, start:end], kt) * scale
                if mask is not None:
                    with _profile_detail(f"{key}.mask_add", q.device):
                        mask_chunk = mask if mask.shape[-2] == 1 else mask[..., start:end, :]
                        scores = scores + mask_chunk
                with _profile_detail(f"{key}.softmax", q.device):
                    probs = self._softmax(scores, v.dtype, allow_scaled_masked_softmax=mask is None)
                with _profile_detail(f"{key}.pv", q.device):
                    chunks.append(torch.matmul(probs, v))
            with _profile_detail(f"{key}.cat", q.device):
                out = torch.cat(chunks, dim=2)
        else:
            with _profile_detail(f"{key}.alloc_out", v.device):
                out = torch.empty((b, heads, q_len, dim_head), device=v.device, dtype=v.dtype)
            for start in range(0, q_len, chunk):
                end = min(start + chunk, q_len)
                with _profile_detail(f"{key}.qk", q.device):
                    scores = torch.matmul(q[:, :, start:end], kt) * scale
                if mask is not None:
                    with _profile_detail(f"{key}.mask_add", q.device):
                        mask_chunk = mask if mask.shape[-2] == 1 else mask[..., start:end, :]
                        scores = scores + mask_chunk
                with _profile_detail(f"{key}.softmax", q.device):
                    probs = self._softmax(scores, v.dtype, allow_scaled_masked_softmax=mask is None)
                with _profile_detail(f"{key}.pv", q.device):
                    out[:, :, start:end] = torch.matmul(probs, v)
        with _profile_detail(f"{key}.out_layout", q.device):
            return out.transpose(1, 2).reshape(b, q_len, heads * dim_head)


class XFormersAttention(AttentionCallable):
    label = "xFormers"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    label = "FlashAttention3"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention4(AttentionCallable):
    label = "FlashAttention4"

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
    ) -> torch.Tensor:
        if flash_attn_4_func is None:
            raise RuntimeError("FlashAttention4 was selected but `flash-attn-4` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        out, _ = flash_attn_4_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out


# --- Automatic selection -----------------------------------------------------
# AUTOMATIC inspects installed extras and the GPU arch and returns the fastest
# usable callable for each path. The selection runs once per process (cached)
# and logs the resulting label once. The unmasked and masked picks are
# independent: each calls its own helper and may end up on different backends
# (e.g. FA3 unmasked + xFormers masked on H100).


def _sdpa_can_use(backend: SDPBackend, *, with_mask: bool) -> bool:
    """Ask torch whether *backend* can run with the given mask shape.
    ``MATH`` is the universal SDPA fallback (pure PyTorch ops, no kernel
    requirements) so it returns True everywhere, CPU included. The other
    backends use ``torch.backends.cuda.can_use_*`` capability checks (no GPU
    compute, no synchronization) and are False without CUDA. The probe shapes
    are small but realistic enough to surface constraints (head dim, dtype)
    that the per-backend rules care about.
    """
    if backend is SDPBackend.MATH:
        return True
    if not torch.cuda.is_available():
        return False
    q = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.float16)
    k = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.float16)
    v = torch.empty(1, 4, 128, 64, device="cuda", dtype=torch.float16)
    mask = torch.zeros(1, 4, 128, 128, device="cuda", dtype=torch.float16) if with_mask else None
    params = torch.backends.cuda.SDPAParams(q, k, v, mask, 0.0, False, False)
    if backend is SDPBackend.CUDNN_ATTENTION:
        return torch.backends.cuda.can_use_cudnn_attention(params, debug=False)
    if backend is SDPBackend.FLASH_ATTENTION:
        return torch.backends.cuda.can_use_flash_attention(params, debug=False)
    if backend is SDPBackend.EFFICIENT_ATTENTION:
        return torch.backends.cuda.can_use_efficient_attention(params, debug=False)
    return False


_SDPA_FULL_PRIORITY: tuple[SDPBackend, ...] = (
    SDPBackend.CUDNN_ATTENTION,
    SDPBackend.FLASH_ATTENTION,
    SDPBackend.EFFICIENT_ATTENTION,
    SDPBackend.MATH,
)


def _sdpa_full_priority() -> PytorchAttention:
    """Hand SDPA the full backend priority order; let torch's dispatcher pick at call time.
    ``sdpa_kernel(_SDPA_FULL_PRIORITY, set_priority=True)`` enables all four
    backends and orders them; torch then walks the order at call time and picks
    the first backend whose ``can_use_*`` check passes for the actual
    shapes/dtype/mask. FLASH is rejected automatically when a mask is present;
    CUDNN may be rejected under deterministic mode; MATH is the universal
    fallback. Probing per-backend usability up front from generic probe shapes
    cannot anticipate the variety of real call sites (e.g. broadcast key-only
    masks, large head dim), so we defer the choice to the dispatcher.
    """
    return PytorchAttention(priority=list(_SDPA_FULL_PRIORITY))


def _ascend_attention_mode() -> str:
    return os.getenv(_ASCEND_ATTENTION_ENV, "").strip().lower()


def ascend_unmasked_attention_backend() -> AttentionCallable:
    """Resolve the Ascend unmasked attention backend from ``LTX2_ASCEND_ATTENTION``.

    This helper is shared with tensor-parallel attention so TP does not bypass the
    selected Ascend backend by hard-coding chunked attention.
    """
    ascend_attn = _ascend_attention_mode()
    if ascend_attn == "fused":
        return AscendFusedAttention()
    if ascend_attn in _ASCEND_STREAMING_ATTENTION_VALUES:
        return AscendStreamingAttention()
    return AscendChunkedAttention()


def ascend_masked_attention_backend() -> MaskedAttentionCallable:
    """Resolve the Ascend masked attention backend.

    Streaming attention v1 deliberately does not handle masks because LTX masks
    are additive and can include fractional/log-space weights.  Masked Ascend
    calls therefore stay on the exact chunked fallback unless ``fused`` is
    explicitly requested.
    """
    ascend_attn = _ascend_attention_mode()
    if ascend_attn == "fused":
        return AscendFusedAttention()
    return AscendChunkedAttention()


def _select_primary_attention() -> AttentionCallable:
    """Pick the fastest unmasked attention based on installed extras and GPU arch.
    Priority by arch:
    - Hopper (sm_90, H100): FA3 / xFormers (mutually exclusive at import) > FA4 > SDPA.
    - Datacenter Blackwell (sm_100, B200): FA4 > SDPA. FA4 is intentionally *not*
      picked on consumer Blackwell (sm_120) -- known regressions in newer
      FA4 betas; users who want it on sm_120 must opt in explicitly.
    - Everywhere else (Ada, Ampere, CPU): SDPA with the full backend priority
      list -- torch's runtime dispatcher picks the best fit at call time.
    """
    ascend_attn = _ascend_attention_mode()
    if (
        ascend_attn == "fused"
        or ascend_attn in _ASCEND_CHUNKED_ATTENTION_VALUES
        or ascend_attn in _ASCEND_STREAMING_ATTENTION_VALUES
    ):
        return ascend_unmasked_attention_backend()
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major == 9:
            if flash_attn_interface is not None:
                return FlashAttention3()
            if memory_efficient_attention is not None:
                return XFormersAttention()
            if flash_attn_4_func is not None:
                return FlashAttention4()
        if major == 10 and flash_attn_4_func is not None:
            return FlashAttention4()
    return _sdpa_full_priority()


def _select_masked_attention() -> MaskedAttentionCallable:
    """Pick a mask-aware attention. Prefers xFormers when installed; else SDPA with
    the full priority list (the dispatcher rejects FLASH automatically when a
    mask is present and walks past it)."""
    ascend_attn = _ascend_attention_mode()
    if (
        ascend_attn == "fused"
        or ascend_attn in _ASCEND_CHUNKED_ATTENTION_VALUES
        or ascend_attn in _ASCEND_STREAMING_ATTENTION_VALUES
    ):
        return ascend_masked_attention_backend()
    if torch.cuda.is_available() and memory_efficient_attention is not None:
        return XFormersAttention()
    return _sdpa_full_priority()


@functools.cache
def automatic_attention() -> AttentionCallable:
    """Cached AUTOMATIC pick for the unmasked path. Logs the chosen label once
    per process."""
    fn = _select_primary_attention()
    logger.info("Automatic attention selected: %s", fn.label)
    return fn


@functools.cache
def automatic_masked_attention() -> MaskedAttentionCallable:
    """Cached AUTOMATIC pick for the masked path. Logs the chosen label once
    per process."""
    fn = _select_masked_attention()
    logger.info("Automatic masked attention selected: %s", fn.label)
    return fn


def _resolve_sdpa_variant(backend: SDPBackend, name: str, *, with_mask: bool) -> PytorchAttention:
    """Build a single-backend ``PytorchAttention`` pin, raising if the backend
    can't actually serve the call on this machine. Used by both
    :meth:`AttentionFunction.to_callable` and :meth:`MaskedAttentionFunction.to_callable`;
    ``with_mask`` differs between the two so the capability check considers
    the protocol the caller intends to use. Not used for ``MATH`` -- MATH is
    the universal fallback and would falsely fail the CUDA-only probe on CPU.
    """
    if not _sdpa_can_use(backend, with_mask=with_mask):
        raise RuntimeError(
            f"{name} selected but the SDPA {backend.name} backend is not usable on this machine "
            "(either no CUDA, the backend rejected the probe shapes, or "
            "torch.use_deterministic_algorithms(True) excluded it)."
        )
    return PytorchAttention(priority=[backend])


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLASH_ATTENTION_4 = "flash_attention_4"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_FLASH = "sdpa_flash"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
    # Pick the fastest unmasked backend for the current GPU/extras combo; see
    # :func:`automatic_attention`. Default for :class:`AttentionOps`.
    AUTOMATIC = "automatic"

    def to_callable(self) -> AttentionCallable:  # noqa: PLR0911
        """Resolve to a concrete callable. Use this at module init time so that
        torch.compile can trace through the attention call without graph breaks.
        Every non-AUTOMATIC variant raises :class:`RuntimeError` when the backend
        isn't usable on this machine -- missing package or SDPA backend rejected
        on this hardware (e.g. cuDNN under ``torch.use_deterministic_algorithms``).
        Opting in means "this kernel or fail loudly". ``AUTOMATIC`` returns the
        cached :func:`automatic_attention` instance so the once-per-process log
        fires only on the first resolution.
        """
        match self:
            case AttentionFunction.AUTOMATIC:
                return automatic_attention()
            case AttentionFunction.PYTORCH:
                return PytorchAttention()
            case AttentionFunction.XFORMERS:
                if memory_efficient_attention is None:
                    raise RuntimeError("AttentionFunction.XFORMERS selected but `xformers` is not installed.")
                return XFormersAttention()
            case AttentionFunction.FLASH_ATTENTION_3:
                if flash_attn_interface is None:
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_3 selected but `flash-attn-3` is not installed."
                    )
                return FlashAttention3()
            case AttentionFunction.FLASH_ATTENTION_4:
                if flash_attn_4_func is None:
                    raise RuntimeError(
                        "AttentionFunction.FLASH_ATTENTION_4 selected but `flash-attn-4` is not installed."
                    )
                return FlashAttention4()
            case AttentionFunction.SDPA_MATH:
                return PytorchAttention(priority=[SDPBackend.MATH])
            case AttentionFunction.SDPA_CUDNN:
                return _resolve_sdpa_variant(
                    SDPBackend.CUDNN_ATTENTION, "AttentionFunction.SDPA_CUDNN", with_mask=False
                )
            case AttentionFunction.SDPA_FLASH:
                return _resolve_sdpa_variant(
                    SDPBackend.FLASH_ATTENTION, "AttentionFunction.SDPA_FLASH", with_mask=False
                )
            case AttentionFunction.SDPA_EFFICIENT:
                return _resolve_sdpa_variant(
                    SDPBackend.EFFICIENT_ATTENTION, "AttentionFunction.SDPA_EFFICIENT", with_mask=False
                )


class MaskedAttentionFunction(Enum):
    """Backends usable on the masked path. Mirrors :class:`AttentionFunction` minus
    the variants the torch SDPA dispatcher (or the wrapped kernel) rejects with a
    mask: ``SDPA_FLASH`` -- FLASH kernel cannot serve an additive ``attn_mask``;
    ``FLASH_ATTENTION_3``/``FLASH_ATTENTION_4`` -- neither has a mask kernel at all.
    Keeping them out makes "this backend cannot mask" a type error, not a runtime one."""

    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    SDPA_CUDNN = "sdpa_cudnn"
    SDPA_EFFICIENT = "sdpa_efficient"
    SDPA_MATH = "sdpa_math"
    # Pick the fastest mask-capable backend for the current extras combo; see
    # :func:`automatic_masked_attention`. Default for the masked slot of
    # :class:`AttentionOps`.
    AUTOMATIC = "automatic"

    def to_callable(self) -> MaskedAttentionCallable:
        """Resolve to a concrete masked callable. Same backend classes as
        :meth:`AttentionFunction.to_callable`; the protocol returned just exposes
        the masked call signature.
        Non-AUTOMATIC variants raise :class:`RuntimeError` when the backend isn't
        usable for the masked path on this machine. SDPA probes run with
        ``with_mask=True`` so the capability check considers the protocol the
        caller will actually use."""
        match self:
            case MaskedAttentionFunction.AUTOMATIC:
                return automatic_masked_attention()
            case MaskedAttentionFunction.PYTORCH:
                return PytorchAttention()
            case MaskedAttentionFunction.XFORMERS:
                if memory_efficient_attention is None:
                    raise RuntimeError("MaskedAttentionFunction.XFORMERS selected but `xformers` is not installed.")
                return XFormersAttention()
            case MaskedAttentionFunction.SDPA_MATH:
                return PytorchAttention(priority=[SDPBackend.MATH])
            case MaskedAttentionFunction.SDPA_CUDNN:
                return _resolve_sdpa_variant(
                    SDPBackend.CUDNN_ATTENTION, "MaskedAttentionFunction.SDPA_CUDNN", with_mask=True
                )
            case MaskedAttentionFunction.SDPA_EFFICIENT:
                return _resolve_sdpa_variant(
                    SDPBackend.EFFICIENT_ATTENTION, "MaskedAttentionFunction.SDPA_EFFICIENT", with_mask=True
                )


@dataclass(frozen=True)
class AttentionOps:
    """Pluggable callables consumed by :class:`Attention`."""

    attention_function: AttentionCallable = field(default_factory=lambda: AttentionFunction.AUTOMATIC.to_callable())
    masked_attention_function: MaskedAttentionCallable = field(
        default_factory=lambda: MaskedAttentionFunction.AUTOMATIC.to_callable()
    )
    preattention_function: PreAttentionCallable = field(default_factory=PytorchPreAttention)
    gated_attention_function: GatedAttentionCallable = field(default_factory=PytorchGatedAttention)


class Attention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.SPLIT,
        ops: AttentionOps | None = None,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        if ops is None:
            ops = AttentionOps()
        self.rope_type = rope_type
        self.attention_function = ops.attention_function
        self.masked_attention_function = ops.masked_attention_function
        self.preattention_function = ops.preattention_function
        self.gated_attention_function = ops.gated_attention_function

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.
        When ``perturbation_mask`` is all zeros, the expensive query/key path
        (linear projections, RMSNorm, RoPE) is skipped entirely and only the
        value projection is used as a pass-through.
        Args:
            x: Query input tensor of shape ``(B, T, query_dim)``.
            context: Key/value context tensor of shape ``(B, S, context_dim)``.
                Falls back to ``x`` (self-attention) when *None*.
            mask: Optional attention mask. Interpretation depends on the attention
                backend (additive bias for xformers/PyTorch SDPA). A non-None
                ``mask`` routes to ``masked_attention_function``; ``None`` keeps
                the unmasked path.
            pe: Rotary positional embeddings applied to both ``q`` and ``k``.
            k_pe: Separate rotary positional embeddings for ``k`` only. When
                *None*, ``pe`` is reused for keys.
            perturbation_mask: Optional mask in ``[0, 1]`` that
                blends the attention output with the raw value projection:
                ``out = attn_out * mask + v * (1 - mask)``.
                **1** keeps the full attention output, **0** bypasses attention
                and passes the value projection through unchanged.
                *None* or all-ones means standard attention; all-zeros skips
                the query/key path entirely for efficiency.
            all_perturbed: Whether all perturbations are active for this block.
        Returns:
            Output tensor of shape ``(B, T, query_dim)``.
        """
        context = x if context is None else context
        use_attention = not all_perturbed

        v = self.to_v(context)

        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)
            q, k = self.preattention_function(q, k, self, mask, pe, k_pe)
            if mask is None:
                out = self.attention_function(q, k, v, self.heads)  # (B, T, H*D)
            else:
                out = self.masked_attention_function(q, k, v, self.heads, mask)

            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        # Apply per-head gating if enabled
        if self.to_gate_logits is not None:
            out = self.gated_attention_function(x, out, self)

        return self.to_out(out)

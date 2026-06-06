from __future__ import annotations

import atexit
import gc
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch
import torch.distributed as dist

from ltx_core.accelerator import synchronize
from ltx_core.distributed.hccl import is_rank0
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.transformer.attention import (
    Attention,
    ascend_masked_attention_backend,
    ascend_unmasked_attention_backend,
)
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.gelu_approx import GELUApprox
from ltx_core.model.transformer.rope_npu import apply_rotary_emb_pair_backend


logger = logging.getLogger(__name__)
_PROFILE_DETAIL_ENV = "LTX2_ASCEND_PROFILE_DETAIL"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_PROFILE_STATS: dict[str, list[float]] = {}
_PROFILE_REGISTERED = False


def _detail_profile_enabled() -> bool:
    return os.environ.get(_PROFILE_DETAIL_ENV, "").strip().lower() in _TRUTHY_ENV_VALUES


def _record_profile(name: str, elapsed: float) -> None:
    stats = _PROFILE_STATS.setdefault(name, [0.0, 0.0])
    stats[0] += 1.0
    stats[1] += elapsed


def _emit_profile_summary() -> None:
    if not _PROFILE_STATS:
        return
    try:
        rank0 = is_rank0() if dist.is_available() and dist.is_initialized() else os.environ.get("RANK", "0") == "0"
    except Exception:
        rank0 = os.environ.get("RANK", "0") == "0"
    if not rank0:
        return
    for name, (count, total) in sorted(_PROFILE_STATS.items(), key=lambda item: (-item[1][1], item[0])):
        avg = total / count if count else 0.0
        logger.info("[profile-detail] %s count=%d total=%.3fs avg=%.6fs", name, int(count), total, avg)


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


def _shard_range(size: int, rank: int, world_size: int) -> slice:
    if size % world_size != 0:
        raise ValueError(f"cannot shard dimension {size} across {world_size} ranks")
    local = size // world_size
    return slice(rank * local, (rank + 1) * local)


def _to_device_if_needed(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move ``value`` only when it is not already on ``device``.

    ``Tensor.to(device)`` is normally a no-op when the tensor is already on the
    target device, but keeping the hot tensor-parallel path explicit avoids
    backend-specific no-op dispatch overhead and makes the intended fast path
    clear.
    """
    if value.device == device:
        return value
    return value.to(device=device, non_blocking=True)


def _to_device_dtype_if_needed(value: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if value.device == device and value.dtype == dtype:
        return value
    return value.to(device=device, dtype=dtype, non_blocking=True)


def _to_device_safe(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    dtype = torch.float16 if device.type == "npu" else value.dtype
    if value.dtype != dtype:
        value = value.to(dtype=dtype)
    if not value.is_contiguous():
        value = value.contiguous()
    return _to_device_if_needed(value, device)


def _copy_linear_rows(source: torch.nn.Linear, rows: slice, device: torch.device) -> torch.nn.Linear:
    out_features = rows.stop - rows.start
    dtype = torch.float16 if device.type == "npu" else source.weight.dtype
    target = torch.nn.Linear(source.in_features, out_features, bias=source.bias is not None, device="meta", dtype=dtype)
    target.weight = torch.nn.Parameter(_to_device_safe(source.weight[rows], device))
    if source.bias is not None:
        target.bias = torch.nn.Parameter(_to_device_safe(source.bias[rows], device))
    return target


def _copy_linear_columns_without_bias(source: torch.nn.Linear, columns: slice, device: torch.device) -> torch.nn.Linear:
    in_features = columns.stop - columns.start
    dtype = torch.float16 if device.type == "npu" else source.weight.dtype
    target = torch.nn.Linear(in_features, source.out_features, bias=False, device="meta", dtype=dtype)
    target.weight = torch.nn.Parameter(_to_device_safe(source.weight[:, columns], device))
    return target


def _parameter_slice(value: torch.Tensor, tensor_slice: slice, device: torch.device) -> torch.nn.Parameter:
    return torch.nn.Parameter(_to_device_safe(value[tensor_slice], device))


def _concat_linear_rows(modules: tuple[torch.nn.Linear, ...]) -> torch.nn.Linear | None:
    if not modules:
        return None
    first = modules[0]
    if any(module.in_features != first.in_features for module in modules):
        return None
    has_bias = first.bias is not None
    if any((module.bias is not None) != has_bias for module in modules):
        return None
    out_features = sum(module.out_features for module in modules)
    target = torch.nn.Linear(
        first.in_features,
        out_features,
        bias=has_bias,
        device="meta",
        dtype=first.weight.dtype,
    )
    weight = torch.cat([module.weight for module in modules], dim=0).contiguous()
    target.weight = torch.nn.Parameter(weight)
    if has_bias:
        bias = torch.cat([module.bias for module in modules], dim=0).contiguous()
        target.bias = torch.nn.Parameter(bias)
    return target


def _linear_slice(module: torch.nn.Linear, x: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    bias = None if module.bias is None else module.bias[start:stop]
    return torch.nn.functional.linear(x, module.weight[start:stop], bias)


class HCCLTensorParallelAttention(torch.nn.Module):
    def __init__(
        self,
        source: Attention,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.process_group = process_group
        self.heads = source.heads
        self.dim_head = source.dim_head
        self.local_heads = self.heads // world_size
        if self.local_heads * world_size != self.heads:
            raise ValueError(f"cannot shard {self.heads} attention heads across {world_size} ranks")
        self.local_inner_dim = self.local_heads * self.dim_head
        self.inner_dim = self.heads * self.dim_head
        self.head_slice = _shard_range(self.heads, rank, world_size)
        self.inner_slice = _shard_range(self.inner_dim, rank, world_size)
        self.rope_type = source.rope_type
        self.norm_eps = source.q_norm.eps
        self.attention_function = source.attention_function
        self.masked_attention_function = source.masked_attention_function
        if device.type == "npu":
            self.npu_attention_function = ascend_unmasked_attention_backend()
            self.npu_masked_attention_function = ascend_masked_attention_backend()
        else:
            self.npu_attention_function = None
            self.npu_masked_attention_function = None

        self.q_norm_weight = _parameter_slice(source.q_norm.weight, self.inner_slice, device)
        self.k_norm_weight = _parameter_slice(source.k_norm.weight, self.inner_slice, device)
        q_proj = _copy_linear_rows(source.to_q, self.inner_slice, device)
        k_proj = _copy_linear_rows(source.to_k, self.inner_slice, device)
        v_proj = _copy_linear_rows(source.to_v, self.inner_slice, device)
        self.to_qkv = _concat_linear_rows((q_proj, k_proj, v_proj))
        if self.to_qkv is not None:
            self.to_q = None
            self.to_k = None
            self.to_v = None
            self.to_kv = None
        else:
            self.to_q = q_proj
            self.to_kv = _concat_linear_rows((k_proj, v_proj))
            if self.to_kv is not None:
                self.to_k = None
                self.to_v = None
            else:
                self.to_k = k_proj
                self.to_v = v_proj
        self.to_gate_logits = (
            _copy_linear_rows(source.to_gate_logits, self.head_slice, device)
            if source.to_gate_logits is not None
            else None
        )
        self.to_out = _copy_linear_columns_without_bias(source.to_out[0], self.inner_slice, device)
        self.out_bias = (
            torch.nn.Parameter(_to_device_safe(source.to_out[0].bias, device))
            if source.to_out[0].bias is not None
            else None
        )

    def _global_rms_norm(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        square_sum = value.float().pow(2).sum(dim=-1, keepdim=True)
        with _profile_detail("tp_attn.rms_norm.all_reduce", value.device):
            dist.all_reduce(square_sum, op=dist.ReduceOp.SUM, group=self.process_group)
        scale = torch.rsqrt(square_sum / self.inner_dim + self.norm_eps).to(dtype=value.dtype)
        return value * scale * weight

    def _global_rms_norm_pair(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Normalize q/k with one HCCL reduction when their prefix shapes match.

        Tensor-parallel q/k RMSNorm needs a global square-sum over the sharded
        hidden dimension.  q and k usually have identical ``(B,T)`` prefixes in
        self-attention, so their two scalar square-sum tensors can be concatenated
        along the last dimension.  Cross-attention has different q/k sequence
        lengths; pack its flattened square-sum tensors instead, still using one
        exact ``all_reduce`` launch.
        """
        q_square_sum = q.float().pow(2).sum(dim=-1, keepdim=True)
        k_square_sum = k.float().pow(2).sum(dim=-1, keepdim=True)

        if q.shape[:-1] != k.shape[:-1]:
            # Cross-attention has different q/k sequence lengths, so the cheap
            # same-shape concat path below cannot be used directly.  The two
            # reductions are still mathematically independent elementwise sums,
            # so flatten and pack them into one HCCL all_reduce launch, then
            # split back to the original prefix shapes.  This preserves the exact
            # RMSNorm formula while removing one collective launch per cross-attn
            # q/k normalization.
            q_numel = q_square_sum.numel()
            q_shape = q_square_sum.shape
            k_shape = k_square_sum.shape
            square_sums = torch.cat((q_square_sum.reshape(-1), k_square_sum.reshape(-1)), dim=0)
            with _profile_detail("tp_attn.rms_norm_pair_packed.all_reduce", q.device):
                dist.all_reduce(square_sums, op=dist.ReduceOp.SUM, group=self.process_group)
            q_square_sum = square_sums[:q_numel].view(q_shape)
            k_square_sum = square_sums[q_numel:].view(k_shape)
            q_scale = torch.rsqrt(q_square_sum / self.inner_dim + self.norm_eps).to(dtype=q.dtype)
            k_scale = torch.rsqrt(k_square_sum / self.inner_dim + self.norm_eps).to(dtype=k.dtype)
            return q * q_scale * self.q_norm_weight, k * k_scale * self.k_norm_weight

        square_sums = torch.cat((q_square_sum, k_square_sum), dim=-1)
        with _profile_detail("tp_attn.rms_norm_pair.all_reduce", q.device):
            dist.all_reduce(square_sums, op=dist.ReduceOp.SUM, group=self.process_group)
        q_scale = torch.rsqrt(square_sums[..., :1] / self.inner_dim + self.norm_eps).to(dtype=q.dtype)
        k_scale = torch.rsqrt(square_sums[..., 1:] / self.inner_dim + self.norm_eps).to(dtype=k.dtype)
        return q * q_scale * self.q_norm_weight, k * k_scale * self.k_norm_weight

    def _slice_rope(
        self,
        pe: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if pe is None:
            return None
        cos, sin = pe
        if cos.ndim == 4 and cos.shape[1] == self.heads:
            return (
                _to_device_dtype_if_needed(cos[:, self.head_slice], self.device, dtype),
                _to_device_dtype_if_needed(sin[:, self.head_slice], self.device, dtype),
            )
        if cos.shape[-1] == self.inner_dim:
            return (
                _to_device_dtype_if_needed(cos[..., self.inner_slice], self.device, dtype),
                _to_device_dtype_if_needed(sin[..., self.inner_slice], self.device, dtype),
            )
        return (
            _to_device_dtype_if_needed(cos, self.device, dtype),
            _to_device_dtype_if_needed(sin, self.device, dtype),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        k_pe: tuple[torch.Tensor, torch.Tensor] | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
    ) -> torch.Tensor:
        self_attention = context is None or context is x
        context = x if context is None else context
        x = _to_device_if_needed(x, self.device)
        context = x if self_attention else _to_device_if_needed(context, self.device)
        mask = None if mask is None else _to_device_if_needed(mask, self.device)
        perturbation_mask = None if perturbation_mask is None else _to_device_if_needed(perturbation_mask, self.device)
        use_attention = not all_perturbed

        attn_kind = "self" if self_attention else "cross"
        if not use_attention:
            with _profile_detail(f"tp_attn.{attn_kind}.value_proj_only", self.device):
                if self.to_qkv is not None:
                    out = _linear_slice(self.to_qkv, context, 2 * self.local_inner_dim, 3 * self.local_inner_dim)
                elif self.to_kv is not None:
                    out = _linear_slice(self.to_kv, context, self.local_inner_dim, 2 * self.local_inner_dim)
                else:
                    out = self.to_v(context)
        else:
            with _profile_detail(f"tp_attn.{attn_kind}.qkv_proj", self.device):
                if self_attention and self.to_qkv is not None:
                    q, k, v = self.to_qkv(x).split(self.local_inner_dim, dim=-1)
                elif self.to_qkv is not None:
                    q = _linear_slice(self.to_qkv, x, 0, self.local_inner_dim)
                    k, v = _linear_slice(self.to_qkv, context, self.local_inner_dim, 3 * self.local_inner_dim).split(
                        self.local_inner_dim,
                        dim=-1,
                    )
                elif self.to_kv is not None:
                    q = self.to_q(x)
                    k, v = self.to_kv(context).split(self.local_inner_dim, dim=-1)
                else:
                    q = self.to_q(x)
                    k = self.to_k(context)
                    v = self.to_v(context)
            with _profile_detail(f"tp_attn.{attn_kind}.qk_rms_norm", self.device):
                q, k = self._global_rms_norm_pair(q, k)
            with _profile_detail(f"tp_attn.{attn_kind}.rope", self.device):
                pe = self._slice_rope(pe, dtype=q.dtype)
                k_pe = self._slice_rope(k_pe, dtype=k.dtype)
                if pe is not None:
                    q, k = apply_rotary_emb_pair_backend(q, k, pe, k_pe, self.rope_type)
            with _profile_detail(f"tp_attn.{attn_kind}.mask_prep", self.device):
                if mask is not None and mask.dtype is not torch.bool:
                    mask = mask.to(dtype=q.dtype)
                    if q.device.type == "npu":
                        mask = torch.where(mask < -1.0, torch.full_like(mask, -1.0e4), mask)
            with _profile_detail(f"tp_attn.{attn_kind}.attention_backend", self.device):
                if mask is not None and self.npu_masked_attention_function is not None:
                    out = self.npu_masked_attention_function(q, k, v, self.local_heads, mask)
                elif self.npu_attention_function is not None:
                    out = self.npu_attention_function(q, k, v, self.local_heads)
                elif mask is None:
                    out = self.attention_function(q, k, v, self.local_heads)
                else:
                    out = self.masked_attention_function(q, k, v, self.local_heads, mask)
            if perturbation_mask is not None:
                with _profile_detail(f"tp_attn.{attn_kind}.perturbation_blend", self.device):
                    out = out * perturbation_mask + v * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            with _profile_detail(f"tp_attn.{attn_kind}.gate", self.device):
                gate_logits = self.to_gate_logits(x)
                b, t, _ = out.shape
                out = out.view(b, t, self.local_heads, self.dim_head)
                gates = 2.0 * torch.sigmoid(gate_logits)
                if not torch.is_grad_enabled():
                    out.mul_(gates.unsqueeze(-1))
                else:
                    out = out * gates.unsqueeze(-1)
                out = out.view(b, t, self.local_inner_dim)

        with _profile_detail(f"tp_attn.{attn_kind}.out_proj", self.device):
            result = self.to_out(out)
        with _profile_detail(f"tp_attn.{attn_kind}.out_all_reduce", self.device):
            dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.process_group)
        if self.out_bias is not None:
            with _profile_detail(f"tp_attn.{attn_kind}.out_bias", self.device):
                result.add_(self.out_bias)
        return result


class HCCLTensorParallelFeedForward(torch.nn.Module):
    def __init__(
        self,
        source: FeedForward,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        super().__init__()
        self.process_group = process_group
        project_in = source.net[0]
        if not isinstance(project_in, GELUApprox):
            raise TypeError(f"unsupported feed-forward input module {type(project_in)!r}")
        project_out = source.net[2]
        if not isinstance(project_out, torch.nn.Linear):
            raise TypeError(f"unsupported feed-forward output module {type(project_out)!r}")
        inner_slice = _shard_range(project_in.proj.out_features, rank, world_size)
        self.project_in = _copy_linear_rows(project_in.proj, inner_slice, device)
        self.project_out = _copy_linear_columns_without_bias(project_out, inner_slice, device)
        self.out_bias = (
            torch.nn.Parameter(_to_device_safe(project_out.bias, device))
            if project_out.bias is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = self.project_in.weight.device
        x = _to_device_if_needed(x, device)
        with _profile_detail("tp_ff.project_in_gelu", device):
            hidden = torch.nn.functional.gelu(self.project_in(x), approximate="tanh")
        with _profile_detail("tp_ff.project_out", device):
            result = self.project_out(hidden)
        with _profile_detail("tp_ff.out_all_reduce", device):
            dist.all_reduce(result, op=dist.ReduceOp.SUM, group=self.process_group)
        if self.out_bias is not None:
            with _profile_detail("tp_ff.out_bias", device):
                result.add_(self.out_bias)
        return result


def _replace_attention(
    block: torch.nn.Module,
    name: str,
    rank: int,
    world_size: int,
    device: torch.device,
    process_group: dist.ProcessGroup | None,
) -> None:
    if hasattr(block, name):
        setattr(
            block,
            name,
            HCCLTensorParallelAttention(
                getattr(block, name),
                rank=rank,
                world_size=world_size,
                device=device,
                process_group=process_group,
            ),
        )


def _replace_feed_forward(
    block: torch.nn.Module,
    name: str,
    rank: int,
    world_size: int,
    device: torch.device,
    process_group: dist.ProcessGroup | None,
) -> None:
    if hasattr(block, name):
        setattr(
            block,
            name,
            HCCLTensorParallelFeedForward(
                getattr(block, name),
                rank=rank,
                world_size=world_size,
                device=device,
                process_group=process_group,
            ),
        )


def apply_hccl_tensor_parallel(
    model: torch.nn.Module,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | None = None,
    process_group: dist.ProcessGroup | None = None,
) -> torch.nn.Module:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("HCCL tensor parallelism requires torch.distributed.init_process_group('hccl')")
    rank = dist.get_rank(group=process_group) if rank is None else rank
    world_size = dist.get_world_size(group=process_group) if world_size is None else world_size
    if device is None:
        local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
        device = torch.device("npu", local_rank)
    if world_size <= 1:
        return model.to(device)

    blocks = getattr(model, "transformer_blocks", None)
    if blocks is None:
        return model.to(device)

    for block in blocks:
        _replace_attention(block, "attn1", rank, world_size, device, process_group)
        _replace_attention(block, "attn2", rank, world_size, device, process_group)
        _replace_attention(block, "audio_attn1", rank, world_size, device, process_group)
        _replace_attention(block, "audio_attn2", rank, world_size, device, process_group)
        _replace_attention(block, "audio_to_video_attn", rank, world_size, device, process_group)
        _replace_attention(block, "video_to_audio_attn", rank, world_size, device, process_group)
        _replace_feed_forward(block, "ff", rank, world_size, device, process_group)
        _replace_feed_forward(block, "audio_ff", rank, world_size, device, process_group)
        block.to(device)
        gc.collect()

    model.to(device)
    model.tensor_parallel = True
    model.tensor_parallel_rank = rank
    model.tensor_parallel_world_size = world_size
    model.tensor_parallel_device = device
    model.tensor_parallel_process_group = process_group
    return model


def build_hccl_tensor_parallel_op(
    *,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | None = None,
    process_group: dist.ProcessGroup | None = None,
    label: str | None = None,
) -> ModuleOps:
    label = label or f"rank{rank if rank is not None else 'env'}_world{world_size if world_size is not None else 'env'}"

    def matcher(model: torch.nn.Module) -> bool:
        return hasattr(model, "transformer_blocks")

    def mutator(model: torch.nn.Module) -> torch.nn.Module:
        return apply_hccl_tensor_parallel(
            model,
            rank=rank,
            world_size=world_size,
            device=device,
            process_group=process_group,
        )

    mutator._ltx2_post_load = True
    return ModuleOps(name=f"ascend_hccl_tensor_parallel_{label}", matcher=matcher, mutator=mutator)

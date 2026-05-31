from __future__ import annotations

import gc
import os

import torch
import torch.distributed as dist

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.transformer.attention import Attention
from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.gelu_approx import GELUApprox
from ltx_core.model.transformer.rope import apply_rotary_emb


def _shard_range(size: int, rank: int, world_size: int) -> slice:
    if size % world_size != 0:
        raise ValueError(f"cannot shard dimension {size} across {world_size} ranks")
    local = size // world_size
    return slice(rank * local, (rank + 1) * local)


def _to_device_safe(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    dtype = torch.float16 if device.type == "npu" else value.dtype
    if value.dtype != dtype:
        value = value.to(dtype=dtype)
    if not value.is_contiguous():
        value = value.contiguous()
    return value.to(device=device, non_blocking=True)


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


class HCCLTensorParallelAttention(torch.nn.Module):
    def __init__(
        self,
        source: Attention,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.device = device
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

        self.q_norm_weight = _parameter_slice(source.q_norm.weight, self.inner_slice, device)
        self.k_norm_weight = _parameter_slice(source.k_norm.weight, self.inner_slice, device)
        self.to_q = _copy_linear_rows(source.to_q, self.inner_slice, device)
        self.to_k = _copy_linear_rows(source.to_k, self.inner_slice, device)
        self.to_v = _copy_linear_rows(source.to_v, self.inner_slice, device)
        self.to_out = _copy_linear_columns_without_bias(source.to_out[0], self.inner_slice, device)
        self.out_bias = (
            torch.nn.Parameter(_to_device_safe(source.to_out[0].bias, device))
            if source.to_out[0].bias is not None
            else None
        )
        self.to_gate_logits = (
            _copy_linear_rows(source.to_gate_logits, self.head_slice, device)
            if source.to_gate_logits is not None
            else None
        )

    def _global_rms_norm(self, value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        square_sum = value.float().pow(2).sum(dim=-1, keepdim=True)
        dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
        scale = torch.rsqrt(square_sum / self.inner_dim + self.norm_eps).to(dtype=value.dtype)
        return value * scale * weight

    def _slice_rope(self, pe: tuple[torch.Tensor, torch.Tensor] | None) -> tuple[torch.Tensor, torch.Tensor] | None:
        if pe is None:
            return None
        cos, sin = pe
        if cos.ndim == 4 and cos.shape[1] == self.heads:
            return cos[:, self.head_slice].to(self.device), sin[:, self.head_slice].to(self.device)
        if cos.shape[-1] == self.inner_dim:
            return cos[..., self.inner_slice].to(self.device), sin[..., self.inner_slice].to(self.device)
        return cos.to(self.device), sin.to(self.device)

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
        context = x if context is None else context
        x = x.to(self.device)
        context = context.to(self.device)
        mask = None if mask is None else mask.to(self.device)
        perturbation_mask = None if perturbation_mask is None else perturbation_mask.to(self.device)
        use_attention = not all_perturbed

        v = self.to_v(context)
        if not use_attention:
            out = v
        else:
            q = self.to_q(x)
            k = self.to_k(context)
            q = self._global_rms_norm(q, self.q_norm_weight)
            k = self._global_rms_norm(k, self.k_norm_weight)
            pe = self._slice_rope(pe)
            k_pe = self._slice_rope(k_pe)
            if pe is not None:
                q = apply_rotary_emb(q, pe, self.rope_type)
                k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
            if mask is None:
                out = self.attention_function(q, k, v, self.local_heads)
            else:
                out = self.masked_attention_function(q, k, v, self.local_heads, mask)
            if perturbation_mask is not None:
                out = out * perturbation_mask + v * (1 - perturbation_mask)

        if self.to_gate_logits is not None:
            gate_logits = self.to_gate_logits(x)
            b, t, _ = out.shape
            out = out.view(b, t, self.local_heads, self.dim_head)
            gates = 2.0 * torch.sigmoid(gate_logits)
            out = out * gates.unsqueeze(-1)
            out = out.view(b, t, self.local_inner_dim)

        result = self.to_out(out)
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        if self.out_bias is not None:
            result = result + self.out_bias
        return result


class HCCLTensorParallelFeedForward(torch.nn.Module):
    def __init__(
        self,
        source: FeedForward,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
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
        x = x.to(self.project_in.weight.device)
        hidden = torch.nn.functional.gelu(self.project_in(x), approximate="tanh")
        result = self.project_out(hidden)
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        if self.out_bias is not None:
            result = result + self.out_bias
        return result


def _replace_attention(block: torch.nn.Module, name: str, rank: int, world_size: int, device: torch.device) -> None:
    if hasattr(block, name):
        setattr(block, name, HCCLTensorParallelAttention(getattr(block, name), rank=rank, world_size=world_size, device=device))


def _replace_feed_forward(block: torch.nn.Module, name: str, rank: int, world_size: int, device: torch.device) -> None:
    if hasattr(block, name):
        setattr(block, name, HCCLTensorParallelFeedForward(getattr(block, name), rank=rank, world_size=world_size, device=device))


def apply_hccl_tensor_parallel(
    model: torch.nn.Module,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | None = None,
) -> torch.nn.Module:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("HCCL tensor parallelism requires torch.distributed.init_process_group('hccl')")
    rank = dist.get_rank() if rank is None else rank
    world_size = dist.get_world_size() if world_size is None else world_size
    if device is None:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device("npu", local_rank)
    if world_size <= 1:
        return model.to(device)

    blocks = getattr(model, "transformer_blocks", None)
    if blocks is None:
        return model.to(device)

    for block in blocks:
        _replace_attention(block, "attn1", rank, world_size, device)
        _replace_attention(block, "attn2", rank, world_size, device)
        _replace_attention(block, "audio_attn1", rank, world_size, device)
        _replace_attention(block, "audio_attn2", rank, world_size, device)
        _replace_attention(block, "audio_to_video_attn", rank, world_size, device)
        _replace_attention(block, "video_to_audio_attn", rank, world_size, device)
        _replace_feed_forward(block, "ff", rank, world_size, device)
        _replace_feed_forward(block, "audio_ff", rank, world_size, device)
        block.to(device)
        gc.collect()

    model.to(device)
    model.tensor_parallel = True
    model.tensor_parallel_rank = rank
    model.tensor_parallel_world_size = world_size
    model.tensor_parallel_device = device
    return model


def build_hccl_tensor_parallel_op(
    *,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | None = None,
) -> ModuleOps:
    label = f"rank{rank if rank is not None else 'env'}_world{world_size if world_size is not None else 'env'}"

    def matcher(model: torch.nn.Module) -> bool:
        return hasattr(model, "transformer_blocks")

    def mutator(model: torch.nn.Module) -> torch.nn.Module:
        return apply_hccl_tensor_parallel(model, rank=rank, world_size=world_size, device=device)

    mutator._ltx2_post_load = True
    return ModuleOps(name=f"ascend_hccl_tensor_parallel_{label}", matcher=matcher, mutator=mutator)

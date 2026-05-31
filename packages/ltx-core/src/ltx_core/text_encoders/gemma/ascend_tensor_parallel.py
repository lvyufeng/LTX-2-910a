from __future__ import annotations

import copy
import gc
import os

import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from transformers.models.gemma3.modeling_gemma3 import Gemma3Attention, Gemma3MLP, apply_rotary_pos_emb, repeat_kv

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.transformer.ascend_tensor_parallel import (
    HCCLTensorParallelAttention,
    HCCLTensorParallelFeedForward,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder


def _shard_range(size: int, rank: int, world_size: int) -> slice:
    if size % world_size != 0:
        raise ValueError(f"cannot shard dimension {size} across {world_size} ranks")
    local = size // world_size
    return slice(rank * local, (rank + 1) * local)


def _copy_linear_rows(source: nn.Linear, rows: slice, device: torch.device) -> nn.Linear:
    out_features = rows.stop - rows.start
    target = nn.Linear(
        source.in_features,
        out_features,
        bias=source.bias is not None,
        device=device,
        dtype=source.weight.dtype,
    )
    with torch.no_grad():
        target.weight.copy_(source.weight[rows].to(device=device, dtype=source.weight.dtype))
        if source.bias is not None:
            target.bias.copy_(source.bias[rows].to(device=device, dtype=source.bias.dtype))
    return target


def _copy_linear_columns_without_bias(source: nn.Linear, columns: slice, device: torch.device) -> nn.Linear:
    in_features = columns.stop - columns.start
    target = nn.Linear(in_features, source.out_features, bias=False, device=device, dtype=source.weight.dtype)
    with torch.no_grad():
        target.weight.copy_(source.weight[:, columns].to(device=device, dtype=source.weight.dtype))
    return target


class HCCLVocabParallelEmbedding(nn.Module):
    def __init__(self, source: nn.Embedding, *, rank: int, world_size: int, device: torch.device) -> None:
        super().__init__()
        rows = _shard_range(source.num_embeddings, rank, world_size)
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.vocab_start = rows.start
        self.vocab_end = rows.stop
        self.num_embeddings = source.num_embeddings
        self.local_num_embeddings = rows.stop - rows.start
        self.embedding_dim = source.embedding_dim
        self.padding_idx = source.padding_idx
        self.weight = nn.Parameter(source.weight[rows].to(device=device, dtype=source.weight.dtype).clone())
        embed_scale = getattr(source, "embed_scale", torch.tensor(1.0, dtype=source.weight.dtype))
        self.register_buffer("embed_scale", embed_scale.to(device=device, dtype=source.weight.dtype), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(self.device)
        local_mask = (input_ids >= self.vocab_start) & (input_ids < self.vocab_end)
        local_ids = (input_ids - self.vocab_start).masked_fill(~local_mask, 0)
        output = F.embedding(local_ids, self.weight)
        output = output * local_mask.unsqueeze(-1).to(output.dtype)
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        return output * self.embed_scale.to(dtype=output.dtype)


class HCCLVocabParallelLinear(nn.Module):
    def __init__(self, source: nn.Linear, *, rank: int, world_size: int, device: torch.device) -> None:
        super().__init__()
        rows = _shard_range(source.out_features, rank, world_size)
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.local_out_features = rows.stop - rows.start
        self.weight = nn.Parameter(source.weight[rows].to(device=device, dtype=source.weight.dtype).clone())
        self.bias = (
            nn.Parameter(source.bias[rows].to(device=device, dtype=source.bias.dtype).clone())
            if source.bias is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = F.linear(x.to(self.device), self.weight, self.bias)
        gathered = [torch.empty_like(local) for _ in range(self.world_size)]
        dist.all_gather(gathered, local)
        return torch.cat(gathered, dim=-1)


class HCCLColumnParallelLinear(nn.Module):
    def __init__(self, source: nn.Linear, *, rank: int, world_size: int, device: torch.device) -> None:
        super().__init__()
        columns = _shard_range(source.in_features, rank, world_size)
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.input_slice = columns
        self.local_in_features = columns.stop - columns.start
        self.weight = nn.Parameter(source.weight[:, columns].to(device=device, dtype=source.weight.dtype).clone())
        self.bias = (
            nn.Parameter(source.bias.to(device=device, dtype=source.bias.dtype).clone())
            if source.bias is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_local = x[..., self.input_slice].to(self.device)
        result = F.linear(x_local, self.weight, None)
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        if self.bias is not None:
            result = result + self.bias
        return result


class HCCLTensorParallelGemmaAttention(nn.Module):
    def __init__(
        self,
        source: Gemma3Attention,
        *,
        rank: int,
        world_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.config = source.config
        self.layer_idx = source.layer_idx
        self.is_sliding = source.is_sliding
        self.head_dim = source.head_dim
        self.num_attention_heads = source.config.num_attention_heads
        self.num_key_value_heads = source.config.num_key_value_heads
        self.local_attention_heads = self.num_attention_heads // world_size
        self.local_key_value_heads = self.num_key_value_heads // world_size
        if self.local_attention_heads * world_size != self.num_attention_heads:
            raise ValueError(f"cannot shard {self.num_attention_heads} Gemma query heads across {world_size} ranks")
        if self.local_key_value_heads * world_size != self.num_key_value_heads:
            raise ValueError(f"cannot shard {self.num_key_value_heads} Gemma KV heads across {world_size} ranks")
        self.num_key_value_groups = self.local_attention_heads // self.local_key_value_heads
        self.scaling = source.scaling
        self.attention_dropout = source.attention_dropout
        self.is_causal = source.is_causal
        self.attn_logit_softcapping = source.attn_logit_softcapping
        self.sliding_window = source.sliding_window

        q_slice = _shard_range(self.num_attention_heads * self.head_dim, rank, world_size)
        kv_slice = _shard_range(self.num_key_value_heads * self.head_dim, rank, world_size)
        self.q_proj = _copy_linear_rows(source.q_proj, q_slice, device)
        self.k_proj = _copy_linear_rows(source.k_proj, kv_slice, device)
        self.v_proj = _copy_linear_rows(source.v_proj, kv_slice, device)
        self.o_proj = _copy_linear_columns_without_bias(source.o_proj, q_slice, device)
        self.out_bias = (
            nn.Parameter(source.o_proj.bias.to(device=device, dtype=source.o_proj.bias.dtype).clone())
            if source.o_proj.bias is not None
            else None
        )
        self.q_norm = copy.deepcopy(source.q_norm).to(device)
        self.k_norm = copy.deepcopy(source.k_norm).to(device)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None,
        past_key_values: object | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        hidden_states = hidden_states.to(self.device)
        input_shape = hidden_states.shape[:-1]
        q_shape = (*input_shape, self.local_attention_heads, self.head_dim)
        kv_shape = (*input_shape, self.local_key_value_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(q_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(kv_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(kv_shape).transpose(1, 2)

        query_states = self.q_norm(query_states)
        key_states = self.k_norm(key_states)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos.to(self.device), sin.to(self.device))

        if past_key_values is not None:
            cache_kwargs = {"sin": sin.to(self.device), "cos": cos.to(self.device), "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if self.attn_logit_softcapping is not None:
            attn_weights = torch.tanh(attn_weights / self.attn_logit_softcapping) * self.attn_logit_softcapping
        if attention_mask is not None:
            causal_mask = attention_mask.to(self.device)[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(*input_shape, -1)

        result = self.o_proj(attn_output)
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        if self.out_bias is not None:
            result = result + self.out_bias
        return result, attn_weights if kwargs.get("output_attentions", False) else None


class HCCLTensorParallelGemmaMLP(nn.Module):
    def __init__(self, source: Gemma3MLP, *, rank: int, world_size: int, device: torch.device) -> None:
        super().__init__()
        rows = _shard_range(source.intermediate_size, rank, world_size)
        self.gate_proj = _copy_linear_rows(source.gate_proj, rows, device)
        self.up_proj = _copy_linear_rows(source.up_proj, rows, device)
        self.down_proj = _copy_linear_columns_without_bias(source.down_proj, rows, device)
        self.act_fn = source.act_fn
        self.hidden_size = source.hidden_size
        self.intermediate_size = source.intermediate_size
        self.config = source.config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.gate_proj.weight.device)
        hidden = self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        result = self.down_proj(hidden)
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result


def _language_model(module: GemmaTextEncoder) -> nn.Module | None:
    conditional = getattr(module, "model", None)
    inner = getattr(conditional, "model", None)
    return getattr(inner, "language_model", None)


def apply_hccl_gemma_tensor_parallel(
    model: GemmaTextEncoder,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | None = None,
) -> GemmaTextEncoder:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Gemma HCCL tensor parallelism requires torch.distributed.init_process_group('hccl')")
    rank = dist.get_rank() if rank is None else rank
    world_size = dist.get_world_size() if world_size is None else world_size
    if device is None:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device("npu", local_rank)
    if world_size <= 1:
        return model.to(device)

    language_model = _language_model(model)
    if language_model is None:
        return model.to(device)

    language_model.embed_tokens = HCCLVocabParallelEmbedding(
        language_model.embed_tokens,
        rank=rank,
        world_size=world_size,
        device=device,
    )
    conditional = model.model
    if hasattr(conditional, "lm_head"):
        conditional.lm_head = HCCLVocabParallelLinear(conditional.lm_head, rank=rank, world_size=world_size, device=device)

    for layer in language_model.layers:
        layer.self_attn = HCCLTensorParallelGemmaAttention(layer.self_attn, rank=rank, world_size=world_size, device=device)
        layer.mlp = HCCLTensorParallelGemmaMLP(layer.mlp, rank=rank, world_size=world_size, device=device)
        layer.to(device)
        gc.collect()

    model.to(device)
    model.gemma_tensor_parallel = True
    model.gemma_tensor_parallel_rank = rank
    model.gemma_tensor_parallel_world_size = world_size
    model.gemma_tensor_parallel_device = device
    return model


def build_hccl_gemma_tensor_parallel_op(
    *,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | None = None,
) -> ModuleOps:
    label = f"rank{rank if rank is not None else 'env'}_world{world_size if world_size is not None else 'env'}"

    def matcher(model: torch.nn.Module) -> bool:
        return isinstance(model, GemmaTextEncoder) and _language_model(model) is not None

    def mutator(model: GemmaTextEncoder) -> GemmaTextEncoder:
        return apply_hccl_gemma_tensor_parallel(model, rank=rank, world_size=world_size, device=device)

    mutator._ltx2_post_load = True
    return ModuleOps(name=f"ascend_hccl_gemma_tensor_parallel_{label}", matcher=matcher, mutator=mutator)


def _replace_connector_blocks(connector: nn.Module | None, rank: int, world_size: int, device: torch.device) -> None:
    if connector is None:
        return
    blocks = getattr(connector, "transformer_1d_blocks", None)
    if blocks is None:
        connector.to(device)
        return
    for block in blocks:
        if hasattr(block, "attn1"):
            block.attn1 = HCCLTensorParallelAttention(block.attn1, rank=rank, world_size=world_size, device=device)
        if hasattr(block, "ff"):
            block.ff = HCCLTensorParallelFeedForward(block.ff, rank=rank, world_size=world_size, device=device)
        block.to(device)
        gc.collect()
    connector.to(device)


def apply_hccl_embeddings_processor_tensor_parallel(
    model: EmbeddingsProcessor,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | None = None,
) -> EmbeddingsProcessor:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Embeddings processor HCCL tensor parallelism requires torch.distributed.init_process_group('hccl')")
    rank = dist.get_rank() if rank is None else rank
    world_size = dist.get_world_size() if world_size is None else world_size
    if device is None:
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device("npu", local_rank)
    if world_size <= 1:
        return model.to(device)

    feature_extractor = model.feature_extractor
    if hasattr(feature_extractor, "aggregate_embed"):
        feature_extractor.aggregate_embed = HCCLColumnParallelLinear(
            feature_extractor.aggregate_embed,
            rank=rank,
            world_size=world_size,
            device=device,
        )
    if hasattr(feature_extractor, "video_aggregate_embed"):
        feature_extractor.video_aggregate_embed = HCCLColumnParallelLinear(
            feature_extractor.video_aggregate_embed,
            rank=rank,
            world_size=world_size,
            device=device,
        )
    if hasattr(feature_extractor, "audio_aggregate_embed") and feature_extractor.audio_aggregate_embed is not None:
        feature_extractor.audio_aggregate_embed = HCCLColumnParallelLinear(
            feature_extractor.audio_aggregate_embed,
            rank=rank,
            world_size=world_size,
            device=device,
        )
    feature_extractor.to(device)

    _replace_connector_blocks(model.video_connector, rank, world_size, device)
    _replace_connector_blocks(model.audio_connector, rank, world_size, device)
    model.to(device)
    model.tensor_parallel = True
    model.tensor_parallel_rank = rank
    model.tensor_parallel_world_size = world_size
    model.tensor_parallel_device = device
    return model


def build_hccl_embeddings_processor_tensor_parallel_op(
    *,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | None = None,
) -> ModuleOps:
    label = f"rank{rank if rank is not None else 'env'}_world{world_size if world_size is not None else 'env'}"

    def matcher(model: torch.nn.Module) -> bool:
        return isinstance(model, EmbeddingsProcessor)

    def mutator(model: EmbeddingsProcessor) -> EmbeddingsProcessor:
        return apply_hccl_embeddings_processor_tensor_parallel(model, rank=rank, world_size=world_size, device=device)

    mutator._ltx2_post_load = True
    return ModuleOps(name=f"ascend_hccl_embeddings_processor_tensor_parallel_{label}", matcher=matcher, mutator=mutator)

from __future__ import annotations

import argparse
import copy
import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
CORE_SRC = REPO_ROOT / "packages" / "ltx-core" / "src"
if str(CORE_SRC) not in sys.path:
    sys.path.insert(0, str(CORE_SRC))

from ltx_core.model.transformer.ascend_tensor_parallel import (  # noqa: E402
    _copy_linear_columns_without_bias,
    _copy_linear_rows,
    _shard_range,
)

# fp16 == the dtype NPU inference actually runs in; bf16 == the dtype the CPU
# reference run uses. fp32 (unsharded) is the ground-truth baseline both are
# measured against. A module whose fp16 error trips these thresholds should be
# upcast to fp32 on the NPU.
REL_WARN = 0.05
ABS_WARN = 0.05

FLAGGED: list[str] = []


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _err(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual = actual.float()
    expected = expected.float()
    diff = (actual - expected).abs()
    max_abs = diff.max().item()
    max_rel = (diff / expected.abs().clamp_min(1e-3)).max().item()
    return max_abs, max_rel


def _report(name: str, ref_fp32: torch.Tensor, tp_fp16: torch.Tensor, tp_bf16: torch.Tensor) -> None:
    abs16, rel16 = _err(tp_fp16, ref_fp32)
    absbf, relbf = _err(tp_bf16, ref_fp32)
    flag16 = abs16 > ABS_WARN and rel16 > REL_WARN
    marker = "  <== fp16 LARGE ERROR -> use fp32 on NPU" if flag16 else ""
    print(
        f"{name:24s} fp16[abs={abs16:.5f} rel={rel16:.5f}]  "
        f"bf16[abs={absbf:.5f} rel={relbf:.5f}]{marker}"
    )
    if flag16:
        FLAGGED.append(name)


def _dtype_copies(module: torch.nn.Module) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    m32 = copy.deepcopy(module).float()
    m16 = copy.deepcopy(m32).to(torch.float16)
    mbf = copy.deepcopy(m32).to(torch.bfloat16)
    return m32, m16, mbf


def _column_parallel(linear: torch.nn.Linear, x: torch.Tensor, world_size: int) -> torch.Tensor:
    result = None
    for rank in range(world_size):
        columns = _shard_range(linear.in_features, rank, world_size)
        shard = _copy_linear_columns_without_bias(linear, columns, torch.device("cpu"))(x[..., columns])
        result = shard if result is None else result + shard
    if linear.bias is not None:
        result = result + linear.bias
    return result


def _row_parallel(linear: torch.nn.Linear, x: torch.Tensor, world_size: int) -> torch.Tensor:
    parts = []
    for rank in range(world_size):
        rows = _shard_range(linear.out_features, rank, world_size)
        parts.append(_copy_linear_rows(linear, rows, torch.device("cpu"))(x))
    return torch.cat(parts, dim=-1)


def _feed_forward_parallel(ff: torch.nn.Module, x: torch.Tensor, world_size: int) -> torch.Tensor:
    project_in = ff.net[0].proj
    project_out = ff.net[2]
    result = None
    for rank in range(world_size):
        inner = _shard_range(project_in.out_features, rank, world_size)
        local_in = _copy_linear_rows(project_in, inner, torch.device("cpu"))
        local_out = _copy_linear_columns_without_bias(project_out, inner, torch.device("cpu"))
        hidden = torch.nn.functional.gelu(local_in(x), approximate="tanh")
        shard = local_out(hidden)
        result = shard if result is None else result + shard
    if project_out.bias is not None:
        result = result + project_out.bias
    return result


def _global_rms_norm_parts(parts: list[torch.Tensor], weights: list[torch.Tensor], eps: float) -> list[torch.Tensor]:
    full = torch.cat(parts, dim=-1)
    scale = torch.rsqrt(full.float().pow(2).mean(dim=-1, keepdim=True) + eps).to(dtype=full.dtype)
    return [part * scale * weight for part, weight in zip(parts, weights, strict=True)]


def _attention_parallel(attn: torch.nn.Module, x: torch.Tensor, context: torch.Tensor, world_size: int) -> torch.Tensor:
    inner_dim = attn.heads * attn.dim_head
    q_parts = []
    k_parts = []
    v_parts = []
    for rank in range(world_size):
        inner = _shard_range(inner_dim, rank, world_size)
        q_parts.append(_copy_linear_rows(attn.to_q, inner, torch.device("cpu"))(x))
        k_parts.append(_copy_linear_rows(attn.to_k, inner, torch.device("cpu"))(context))
        v_parts.append(_copy_linear_rows(attn.to_v, inner, torch.device("cpu"))(context))

    q_parts = _global_rms_norm_parts(
        q_parts,
        [attn.q_norm.weight[_shard_range(inner_dim, rank, world_size)] for rank in range(world_size)],
        attn.q_norm.eps,
    )
    k_parts = _global_rms_norm_parts(
        k_parts,
        [attn.k_norm.weight[_shard_range(inner_dim, rank, world_size)] for rank in range(world_size)],
        attn.k_norm.eps,
    )

    result = None
    local_heads = attn.heads // world_size
    for rank in range(world_size):
        inner = _shard_range(inner_dim, rank, world_size)
        local = attn.attention_function(q_parts[rank], k_parts[rank], v_parts[rank], local_heads)
        shard = _copy_linear_columns_without_bias(attn.to_out[0], inner, torch.device("cpu"))(local)
        result = shard if result is None else result + shard
    if attn.to_out[0].bias is not None:
        result = result + attn.to_out[0].bias
    return result


class EagerAttention:
    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, _, inner_dim = q.shape
        dim_head = inner_dim // heads
        q = q.view(batch, -1, heads, dim_head).transpose(1, 2)
        k = k.view(batch, -1, heads, dim_head).transpose(1, 2)
        v = v.view(batch, -1, heads, dim_head).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (dim_head**-0.5)
        if mask is not None:
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            scores = scores + mask.to(device=scores.device, dtype=scores.dtype)
        probs = torch.softmax(scores.float(), dim=-1).to(dtype=v.dtype)
        out = torch.matmul(probs, v)
        return out.transpose(1, 2).reshape(batch, -1, inner_dim)


# ----------------------------------------------------------------------------
# Gemma text-encoder TP simulations (mirror the HCCL classes' sharding math on
# CPU without requiring torch.distributed).
# ----------------------------------------------------------------------------


def _gemma_attention_parallel(attn: torch.nn.Module, hidden: torch.Tensor, position_embeddings, world_size: int) -> torch.Tensor:
    from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb, repeat_kv

    head_dim = attn.head_dim
    num_q = attn.config.num_attention_heads
    num_kv = attn.config.num_key_value_heads
    local_q = num_q // world_size
    local_kv = num_kv // world_size
    groups = local_q // local_kv
    cos, sin = position_embeddings
    input_shape = hidden.shape[:-1]

    result = None
    for rank in range(world_size):
        q_slice = _shard_range(num_q * head_dim, rank, world_size)
        kv_slice = _shard_range(num_kv * head_dim, rank, world_size)
        q_proj = _copy_linear_rows(attn.q_proj, q_slice, torch.device("cpu"))
        k_proj = _copy_linear_rows(attn.k_proj, kv_slice, torch.device("cpu"))
        v_proj = _copy_linear_rows(attn.v_proj, kv_slice, torch.device("cpu"))
        o_proj = _copy_linear_columns_without_bias(attn.o_proj, q_slice, torch.device("cpu"))

        q = q_proj(hidden).view(*input_shape, local_q, head_dim).transpose(1, 2)
        k = k_proj(hidden).view(*input_shape, local_kv, head_dim).transpose(1, 2)
        v = v_proj(hidden).view(*input_shape, local_kv, head_dim).transpose(1, 2)
        q = attn.q_norm(q)
        k = attn.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, groups)
        v = repeat_kv(v, groups)
        scores = torch.matmul(q, k.transpose(2, 3)) * attn.scaling
        if attn.attn_logit_softcapping is not None:
            scores = torch.tanh(scores / attn.attn_logit_softcapping) * attn.attn_logit_softcapping
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(probs, v).transpose(1, 2).contiguous().reshape(*input_shape, -1)
        shard = o_proj(out)
        result = shard if result is None else result + shard
    if attn.o_proj.bias is not None:
        result = result + attn.o_proj.bias
    return result


def _gemma_mlp_parallel(mlp: torch.nn.Module, x: torch.Tensor, world_size: int) -> torch.Tensor:
    result = None
    for rank in range(world_size):
        rows = _shard_range(mlp.intermediate_size, rank, world_size)
        gate = _copy_linear_rows(mlp.gate_proj, rows, torch.device("cpu"))
        up = _copy_linear_rows(mlp.up_proj, rows, torch.device("cpu"))
        down = _copy_linear_columns_without_bias(mlp.down_proj, rows, torch.device("cpu"))
        hidden = mlp.act_fn(gate(x)) * up(x)
        shard = down(hidden)
        result = shard if result is None else result + shard
    if mlp.down_proj.bias is not None:
        result = result + mlp.down_proj.bias
    return result


def _check_diffusion(upstream_core: Path, world_size: int) -> None:
    upstream_attention = _load_module(
        "upstream_ltx_attention",
        upstream_core / "ltx_core" / "model" / "transformer" / "attention.py",
    )
    upstream_feed_forward = _load_module(
        "upstream_ltx_feed_forward",
        upstream_core / "ltx_core" / "model" / "transformer" / "feed_forward.py",
    )

    print("== diffusion transformer TP ==")

    row_linear = torch.nn.Linear(16, 32)
    x = torch.randn(2, 5, 16)
    m32, m16, mbf = _dtype_copies(row_linear)
    _report(
        "row_parallel_linear",
        m32(x),
        _row_parallel(m16, x.to(torch.float16), world_size),
        _row_parallel(mbf, x.to(torch.bfloat16), world_size),
    )

    column_linear = torch.nn.Linear(32, 16)
    xc = torch.randn(2, 5, 32)
    m32, m16, mbf = _dtype_copies(column_linear)
    _report(
        "column_parallel_linear",
        m32(xc),
        _column_parallel(m16, xc.to(torch.float16), world_size),
        _column_parallel(mbf, xc.to(torch.bfloat16), world_size),
    )

    ff = upstream_feed_forward.FeedForward(dim=16, dim_out=16, mult=4)
    m32, m16, mbf = _dtype_copies(ff)
    _report(
        "feed_forward",
        m32(x),
        _feed_forward_parallel(m16, x.to(torch.float16), world_size),
        _feed_forward_parallel(mbf, x.to(torch.bfloat16), world_size),
    )

    ops = upstream_attention.AttentionOps(
        attention_function=EagerAttention(),
        masked_attention_function=EagerAttention(),
    )
    attn = upstream_attention.Attention(query_dim=16, context_dim=16, heads=4, dim_head=4, ops=ops)
    context = torch.randn(2, 5, 16)
    m32, m16, mbf = _dtype_copies(attn)
    _report(
        "attention",
        m32(x, context=context),
        _attention_parallel(m16, x.to(torch.float16), context.to(torch.float16), world_size),
        _attention_parallel(mbf, x.to(torch.bfloat16), context.to(torch.bfloat16), world_size),
    )


def _check_gemma(world_size: int) -> None:
    try:
        from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
        from transformers.models.gemma3.modeling_gemma3 import (
            Gemma3Attention,
            Gemma3MLP,
            Gemma3RotaryEmbedding,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"== gemma text encoder TP == skipped ({exc})")
        return

    print("== gemma text encoder TP ==")
    hidden_size = 64
    config = Gemma3TextConfig(
        vocab_size=64,
        hidden_size=hidden_size,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=32,
        attn_implementation="eager",
    )
    config._attn_implementation = "eager"
    seq = 6
    hidden = torch.randn(1, seq, hidden_size)

    attn = Gemma3Attention(config, layer_idx=0).eval()
    rotary = Gemma3RotaryEmbedding(config)
    position_ids = torch.arange(seq).unsqueeze(0)
    cos, sin = rotary(hidden, position_ids)

    m32, m16, mbf = _dtype_copies(attn)
    cos32, sin32 = cos.float(), sin.float()
    with torch.no_grad():
        ref, _ = m32(m32_hidden := hidden.float(), position_embeddings=(cos32, sin32), attention_mask=None)
        tp16 = _gemma_attention_parallel(
            m16, hidden.to(torch.float16), (cos.to(torch.float16), sin.to(torch.float16)), world_size
        )
        tpbf = _gemma_attention_parallel(
            mbf, hidden.to(torch.bfloat16), (cos.to(torch.bfloat16), sin.to(torch.bfloat16)), world_size
        )
    _report("gemma_attention", ref, tp16, tpbf)

    mlp = Gemma3MLP(config).eval()
    m32, m16, mbf = _dtype_copies(mlp)
    with torch.no_grad():
        ref = m32(hidden.float())
        tp16 = _gemma_mlp_parallel(m16, hidden.to(torch.float16), world_size)
        tpbf = _gemma_mlp_parallel(mbf, hidden.to(torch.bfloat16), world_size)
    _report("gemma_mlp", ref, tp16, tpbf)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CPU precision check for Ascend TP slicing: fp16 (NPU path) vs bf16 (CPU path) vs fp32 baseline"
    )
    parser.add_argument("--upstream-root", default="/tmp/ltx2-upstream", help="path to a clean upstream LTX-2 checkout")
    parser.add_argument("--world-size", type=int, default=4)
    args = parser.parse_args()

    upstream_core = Path(args.upstream_root) / "packages" / "ltx-core" / "src"
    if not upstream_core.exists():
        raise SystemExit(f"missing upstream core path: {upstream_core}")
    if str(upstream_core) not in sys.path:
        sys.path.insert(0, str(upstream_core))

    torch.manual_seed(0)
    _check_diffusion(upstream_core, args.world_size)
    _check_gemma(args.world_size)

    print()
    if FLAGGED:
        print(f"fp16 modules with large error (recommend fp32 on NPU): {', '.join(FLAGGED)}")
    else:
        print("no module exceeded the fp16 error thresholds; fp16 on NPU is acceptable")


if __name__ == "__main__":
    main()

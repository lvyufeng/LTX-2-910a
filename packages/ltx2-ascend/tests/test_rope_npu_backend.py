"""NPU-gated equivalence test for the RoPE NPU backend.

Phase 1 of the Ascend custom-operator work established that CANN's
``torch_npu.npu_rotary_mul`` (half mode) reproduces LTX
:func:`apply_split_rotary_emb` bit-for-bit in fp16.  Phase 4 HQ A/B showed a
small positive loop-speed delta, so the backend is enabled by default on NPU.
This test pins that property so a future torch_npu/CANN update that regresses it
is caught: the default backend and ``LTX2_ASCEND_ROPE=npu`` must equal the eager
PyTorch path exactly on the real LTX shapes, and INTERLEAVED rope must fall back
(never raise).

Skipped entirely when torch_npu / an NPU device is unavailable, so it is inert
on CPU/GPU CI hosts.
"""
from __future__ import annotations

import pytest
import torch

from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except Exception:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


pytestmark = pytest.mark.skipif(not _npu_available(), reason="requires an Ascend NPU + torch_npu")


# (name, batch, heads, seq, head_dim) over real LTX attention shapes.
_CASES = [
    ("video-small", 1, 32, 256, 128),
    ("audio", 1, 32, 256, 64),
    ("video-tp-shard", 1, 8, 512, 128),
]


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    """The backend resolver is process-cached; reset it around each test."""
    from ltx_core.model.transformer import rope_npu

    rope_npu._rope_backend.cache_clear()
    yield
    rope_npu._rope_backend.cache_clear()


@pytest.mark.parametrize("name,b,heads,t,d", _CASES)
def test_npu_backend_matches_eager_split_3d(monkeypatch, name, b, heads, t, d):
    from ltx_core.accelerator import configure_npu_runtime
    from ltx_core.model.transformer.rope_npu import apply_rotary_emb_backend

    configure_npu_runtime(0)
    dev = torch.device("npu", 0)
    monkeypatch.setenv("LTX2_ASCEND_ROPE", "npu")

    half = d // 2
    torch.manual_seed(0)
    x = torch.randn((b, t, heads * d), dtype=torch.float16, device=dev)
    cos = torch.randn((b, heads, t, half), dtype=torch.float16, device=dev)
    sin = torch.randn((b, heads, t, half), dtype=torch.float16, device=dev)

    eager = apply_rotary_emb(x.clone(), (cos, sin), LTXRopeType.SPLIT)
    backend = apply_rotary_emb_backend(x.clone(), (cos, sin), LTXRopeType.SPLIT)
    torch.npu.synchronize()

    assert backend.shape == eager.shape
    assert backend.dtype == eager.dtype
    # Phase 1 measured exactly 0.0 abs diff; require bit-equality.
    assert torch.equal(backend, eager), (
        f"{name}: npu_rotary_mul rope diverged from eager SPLIT "
        f"(max abs {(backend.float() - eager.float()).abs().max().item():.3e})"
    )


def test_interleaved_falls_back(monkeypatch):
    from ltx_core.accelerator import configure_npu_runtime
    from ltx_core.model.transformer.rope_npu import apply_rotary_emb_backend

    configure_npu_runtime(0)
    dev = torch.device("npu", 0)
    monkeypatch.setenv("LTX2_ASCEND_ROPE", "npu")

    b, heads, t, d = 1, 8, 128, 64
    torch.manual_seed(1)
    x = torch.randn((b, t, heads * d), dtype=torch.float16, device=dev)
    # INTERLEAVED uses full-width cos/sin.
    cos = torch.randn((b, t, heads * d), dtype=torch.float16, device=dev)
    sin = torch.randn((b, t, heads * d), dtype=torch.float16, device=dev)

    eager = apply_rotary_emb(x.clone(), (cos, sin), LTXRopeType.INTERLEAVED)
    backend = apply_rotary_emb_backend(x.clone(), (cos, sin), LTXRopeType.INTERLEAVED)
    torch.npu.synchronize()

    assert torch.equal(backend, eager)


def test_backend_enabled_by_default(monkeypatch):
    """With the gate unset the backend selects npu_rotary_mul by default."""
    from ltx_core.accelerator import configure_npu_runtime
    from ltx_core.model.transformer import rope_npu
    from ltx_core.model.transformer.rope_npu import apply_rotary_emb_backend

    configure_npu_runtime(0)
    dev = torch.device("npu", 0)
    monkeypatch.delenv("LTX2_ASCEND_ROPE", raising=False)

    b, heads, t, d = 1, 8, 128, 128
    half = d // 2
    torch.manual_seed(2)
    x = torch.randn((b, t, heads * d), dtype=torch.float16, device=dev)
    cos = torch.randn((b, heads, t, half), dtype=torch.float16, device=dev)
    sin = torch.randn((b, heads, t, half), dtype=torch.float16, device=dev)

    eager = apply_rotary_emb(x.clone(), (cos, sin), LTXRopeType.SPLIT)
    backend = apply_rotary_emb_backend(x.clone(), (cos, sin), LTXRopeType.SPLIT)
    torch.npu.synchronize()

    assert rope_npu._rope_backend() == "npu"
    assert torch.equal(backend, eager)


def test_backend_can_be_disabled(monkeypatch):
    """LTX2_ASCEND_ROPE=eager forces the exact PyTorch fallback."""
    from ltx_core.accelerator import configure_npu_runtime
    from ltx_core.model.transformer import rope_npu
    from ltx_core.model.transformer.rope_npu import apply_rotary_emb_backend

    configure_npu_runtime(0)
    dev = torch.device("npu", 0)
    monkeypatch.setenv("LTX2_ASCEND_ROPE", "eager")

    b, heads, t, d = 1, 8, 128, 128
    half = d // 2
    torch.manual_seed(3)
    x = torch.randn((b, t, heads * d), dtype=torch.float16, device=dev)
    cos = torch.randn((b, heads, t, half), dtype=torch.float16, device=dev)
    sin = torch.randn((b, heads, t, half), dtype=torch.float16, device=dev)

    eager = apply_rotary_emb(x.clone(), (cos, sin), LTXRopeType.SPLIT)
    backend = apply_rotary_emb_backend(x.clone(), (cos, sin), LTXRopeType.SPLIT)
    torch.npu.synchronize()

    assert rope_npu._rope_backend() == "eager"
    assert torch.equal(backend, eager)


def test_pair_backend_matches_individual_eager(monkeypatch):
    """q/k pair helper must stay bit-exact while reusing duplicated freqs."""
    from ltx_core.accelerator import configure_npu_runtime
    from ltx_core.model.transformer.rope_npu import apply_rotary_emb_pair_backend

    configure_npu_runtime(0)
    dev = torch.device("npu", 0)
    monkeypatch.setenv("LTX2_ASCEND_ROPE", "npu")

    b, heads, t, d = 1, 8, 512, 128
    half = d // 2
    torch.manual_seed(4)
    q = torch.randn((b, t, heads * d), dtype=torch.float16, device=dev)
    k = torch.randn((b, t, heads * d), dtype=torch.float16, device=dev)
    cos = torch.randn((b, heads, t, half), dtype=torch.float16, device=dev)
    sin = torch.randn((b, heads, t, half), dtype=torch.float16, device=dev)
    pe = (cos, sin)

    eager_q = apply_rotary_emb(q.clone(), pe, LTXRopeType.SPLIT)
    eager_k = apply_rotary_emb(k.clone(), pe, LTXRopeType.SPLIT)
    backend_q, backend_k = apply_rotary_emb_pair_backend(q.clone(), k.clone(), pe, None, LTXRopeType.SPLIT)
    torch.npu.synchronize()

    assert torch.equal(backend_q, eager_q)
    assert torch.equal(backend_k, eager_k)

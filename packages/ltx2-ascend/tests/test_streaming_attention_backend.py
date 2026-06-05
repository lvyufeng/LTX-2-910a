from __future__ import annotations

import importlib

import pytest
import torch

from ltx_core.model.transformer.attention import (
    AscendChunkedAttention,
    AscendStreamingAttention,
    ascend_masked_attention_backend,
    ascend_unmasked_attention_backend,
)
streaming_attention_ops = importlib.import_module("ltx2_ascend_ops.streaming_attention")


def test_streaming_backend_selected_by_env(monkeypatch):
    monkeypatch.setenv("LTX2_ASCEND_ATTENTION", "streaming")

    assert isinstance(ascend_unmasked_attention_backend(), AscendStreamingAttention)
    assert isinstance(ascend_masked_attention_backend(), AscendChunkedAttention)


def test_streaming_backend_alias_custom(monkeypatch):
    monkeypatch.setenv("LTX2_ASCEND_ATTENTION", "custom")

    assert isinstance(ascend_unmasked_attention_backend(), AscendStreamingAttention)


@pytest.mark.parametrize("mode", ["eager", "math", ""])
def test_non_streaming_ascend_modes_use_chunked(monkeypatch, mode):
    if mode:
        monkeypatch.setenv("LTX2_ASCEND_ATTENTION", mode)
    else:
        monkeypatch.delenv("LTX2_ASCEND_ATTENTION", raising=False)

    assert isinstance(ascend_unmasked_attention_backend(), AscendChunkedAttention)
    assert isinstance(ascend_masked_attention_backend(), AscendChunkedAttention)


def test_chunked_attention_uses_optimized_defaults(monkeypatch):
    monkeypatch.delenv("LTX2_ASCEND_ATTENTION_CHUNK", raising=False)
    monkeypatch.delenv("LTX2_ASCEND_ATTENTION_CAT_MAX_MB", raising=False)

    attention = AscendChunkedAttention()

    assert attention.query_chunk_size == 4096
    assert attention.cat_output_max_bytes == 32 * 1024 * 1024


def test_chunked_attention_env_fallback_knobs(monkeypatch):
    monkeypatch.setenv("LTX2_ASCEND_ATTENTION_CHUNK", "512")
    monkeypatch.setenv("LTX2_ASCEND_ATTENTION_CAT_MAX_MB", "0")

    attention = AscendChunkedAttention()

    assert attention.query_chunk_size == 512
    assert attention.cat_output_max_bytes == 0


def test_streaming_falls_back_on_cpu(monkeypatch):
    monkeypatch.setenv("LTX2_ASCEND_STREAMING_ATTN_STRICT", "0")
    q = torch.randn(1, 8, 128, dtype=torch.float32)
    k = torch.randn(1, 8, 128, dtype=torch.float32)
    v = torch.randn(1, 8, 128, dtype=torch.float32)

    expected = AscendChunkedAttention()(q, k, v, heads=2)
    actual = AscendStreamingAttention()(q, k, v, heads=2)

    assert torch.allclose(actual, expected)


def test_streaming_strict_reports_unsupported_cpu(monkeypatch):
    monkeypatch.setenv("LTX2_ASCEND_STREAMING_ATTN_STRICT", "1")
    q = torch.randn(1, 8, 128, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="non-npu-tensor"):
        AscendStreamingAttention()(q, q, q, heads=2)



def test_streaming_strict_reports_masked_unsupported(monkeypatch):
    monkeypatch.setenv("LTX2_ASCEND_STREAMING_ATTN_STRICT", "1")
    q = torch.randn(1, 8, 128, dtype=torch.float32)
    mask = torch.zeros(1, 1, 8, 8)

    with pytest.raises(RuntimeError, match="mask-present"):
        AscendStreamingAttention()(q, q, q, heads=2, mask=mask)


def test_native_dispatch_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LTX2_ASCEND_STREAMING_ATTN_ENABLE_NATIVE", raising=False)
    streaming_attention_ops._resolve_native_op.cache_clear()

    assert not streaming_attention_ops.is_available()
    assert "disabled" in streaming_attention_ops.availability_report()

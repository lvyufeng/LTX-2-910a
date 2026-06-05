import os

import torch

from ltx_core.model.transformer.attention import (
    AscendChunkedAttention,
    _ascend_longk_softmax_enabled,
    _ascend_longk_softmax_min_k,
    _ascend_scaled_masked_softmax_disabled,
    _ascend_softmax_fp16,
    _tp_hq_chunk_for_shape,
    _tp_hq_shape_policy_enabled,
)


def test_scaled_masked_softmax_is_default_off(monkeypatch):
    monkeypatch.delenv("LTX2_ASCEND_SCALED_MASKED_SOFTMAX", raising=False)

    assert _ascend_scaled_masked_softmax_disabled()


def test_scaled_masked_softmax_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("LTX2_ASCEND_SCALED_MASKED_SOFTMAX", "1")

    assert not _ascend_scaled_masked_softmax_disabled()


def test_tp_hq_shape_policy_is_backend_default_off(monkeypatch):
    monkeypatch.delenv("LTX2_ASCEND_TP_HQ_SHAPE_POLICY", raising=False)

    assert not _tp_hq_shape_policy_enabled()


def test_tp_hq_shape_policy_stage2_long_k_uses_validated_chunk2048():
    assert _tp_hq_chunk_for_shape(1, 8, 24960, 24960, 128, 1536) == 2048


def test_softmax_fp16_requires_explicit_or_cli_scoped_opt_in(monkeypatch):
    monkeypatch.delenv("LTX2_ASCEND_SOFTMAX_FP16", raising=False)
    assert not _ascend_softmax_fp16()

    monkeypatch.setenv("LTX2_ASCEND_SOFTMAX_FP16", "1")
    assert _ascend_softmax_fp16()


def test_tp_hq_shape_policy_keeps_stage1_chunk1536_without_fp16_softmax(monkeypatch):
    monkeypatch.delenv("LTX2_ASCEND_SOFTMAX_FP16", raising=False)
    assert _tp_hq_chunk_for_shape(1, 8, 6240, 6240, 128, 1536) == 1536


def test_tp_hq_shape_policy_stage1_fp16_softmax_uses_validated_chunk4096(monkeypatch):
    monkeypatch.setenv("LTX2_ASCEND_SOFTMAX_FP16", "1")
    assert _tp_hq_chunk_for_shape(1, 8, 6240, 6240, 128, 1536) == 4096


def test_tp_hq_shape_policy_keeps_explicit_chunk_override_authoritative():
    assert _tp_hq_chunk_for_shape(1, 8, 24960, 24960, 128, 1024) == 1024
    assert _tp_hq_chunk_for_shape(1, 8, 24960, 24960, 128, 4096) == 4096


def test_tp_hq_shape_policy_ignores_cross_attention():
    assert _tp_hq_chunk_for_shape(1, 8, 24960, 128, 128, 1536) == 1536
    assert _tp_hq_chunk_for_shape(1, 8, 126, 24960, 64, 1536) == 1536


def test_longk_softmax_is_default_off(monkeypatch):
    monkeypatch.delenv("LTX2_ASCEND_LONGK_SOFTMAX_NATIVE", raising=False)

    assert not _ascend_longk_softmax_enabled()
    assert not AscendChunkedAttention()._longk_softmax_enabled


def test_longk_softmax_requires_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("LTX2_ASCEND_LONGK_SOFTMAX_NATIVE", "1")

    assert _ascend_longk_softmax_enabled()
    assert AscendChunkedAttention()._longk_softmax_enabled


def test_longk_softmax_min_k_default_and_override(monkeypatch):
    monkeypatch.delenv("LTX2_ASCEND_LONGK_SOFTMAX_MIN_K", raising=False)
    assert _ascend_longk_softmax_min_k() == 8192

    monkeypatch.setenv("LTX2_ASCEND_LONGK_SOFTMAX_MIN_K", "4096")
    assert _ascend_longk_softmax_min_k() == 4096


def test_longk_softmax_disabled_skips_dispatch_decision(monkeypatch):
    monkeypatch.delenv("LTX2_ASCEND_LONGK_SOFTMAX_NATIVE", raising=False)
    attention = AscendChunkedAttention()

    class _FakeScores:
        # Even with a long-K NPU-looking shape, a disabled native op must not be
        # considered for dispatch.
        device = type("Dev", (), {"type": "npu"})()
        ndim = 4
        dtype = torch.float16
        shape = (1, 8, 24960, 24960)

        def is_contiguous(self):
            return True

    assert not attention._should_use_longk_softmax(_FakeScores(), torch.float16)

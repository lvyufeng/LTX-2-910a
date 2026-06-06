import torch

from ltx_core.model.transformer import ascend_tensor_parallel as tp


def _local_rms_norm(value: torch.Tensor, weight: torch.Tensor, inner_dim: int, eps: float) -> torch.Tensor:
    square_sum = value.float().pow(2).sum(dim=-1, keepdim=True)
    scale = torch.rsqrt(square_sum / inner_dim + eps).to(dtype=value.dtype)
    return value * scale * weight


def test_cross_attention_rms_norm_packs_mismatched_shapes_into_one_reduce(monkeypatch):
    calls: list[torch.Tensor] = []

    def fake_all_reduce(tensor, op=None, group=None):
        calls.append(tensor)

    monkeypatch.setattr(tp.dist, "all_reduce", fake_all_reduce)

    module = tp.HCCLTensorParallelAttention.__new__(tp.HCCLTensorParallelAttention)
    module.inner_dim = 4
    module.norm_eps = 1e-6
    module.q_norm_weight = torch.tensor([1.0, 1.1, 0.9, 1.2], dtype=torch.float32)
    module.k_norm_weight = torch.tensor([0.8, 1.3, 1.0, 0.7], dtype=torch.float32)
    module.process_group = object()

    q = torch.randn(2, 3, 4, dtype=torch.float16)
    k = torch.randn(2, 5, 4, dtype=torch.float16)

    actual_q, actual_k = module._global_rms_norm_pair(q, k)

    assert len(calls) == 1
    assert calls[0].shape == (q.shape[0] * q.shape[1] + k.shape[0] * k.shape[1],)
    torch.testing.assert_close(actual_q, _local_rms_norm(q, module.q_norm_weight, module.inner_dim, module.norm_eps))
    torch.testing.assert_close(actual_k, _local_rms_norm(k, module.k_norm_weight, module.inner_dim, module.norm_eps))


def test_self_attention_rms_norm_pair_same_shape_uses_one_reduce(monkeypatch):
    calls: list[torch.Tensor] = []

    def fake_all_reduce(tensor, op=None, group=None):
        calls.append(tensor)

    monkeypatch.setattr(tp.dist, "all_reduce", fake_all_reduce)

    module = tp.HCCLTensorParallelAttention.__new__(tp.HCCLTensorParallelAttention)
    module.inner_dim = 4
    module.norm_eps = 1e-6
    module.q_norm_weight = torch.tensor([1.0, 1.1, 0.9, 1.2], dtype=torch.float32)
    module.k_norm_weight = torch.tensor([0.8, 1.3, 1.0, 0.7], dtype=torch.float32)
    module.process_group = object()

    q = torch.randn(2, 3, 4, dtype=torch.float16)
    k = torch.randn(2, 3, 4, dtype=torch.float16)

    actual_q, actual_k = module._global_rms_norm_pair(q, k)

    assert len(calls) == 1
    assert calls[0].shape == (q.shape[0], q.shape[1], 2)
    torch.testing.assert_close(actual_q, _local_rms_norm(q, module.q_norm_weight, module.inner_dim, module.norm_eps))
    torch.testing.assert_close(actual_k, _local_rms_norm(k, module.k_norm_weight, module.inner_dim, module.norm_eps))

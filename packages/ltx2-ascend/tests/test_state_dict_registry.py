import torch

from ltx_core.loader.helpers import load_state_dict
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.registry import DummyRegistry, StateDictRegistry
from ltx_core.loader.sd_ops import SDOps


class _FakeLoader:
    def __init__(self):
        self.calls = 0

    def metadata(self, path: str) -> dict:
        return {}

    def load(self, path, sd_ops=None, device=None):
        self.calls += 1
        tensor = torch.tensor([self.calls], device=device or torch.device("cpu"))
        return StateDict(sd={"weight": tensor}, device=tensor.device, size=tensor.numel() * tensor.element_size(), dtype={tensor.dtype})


def test_state_dict_registry_reuses_identical_loads():
    loader = _FakeLoader()
    registry = StateDictRegistry()
    sd_ops = SDOps("fake")

    first = load_state_dict("/tmp/model.safetensors", loader, registry, torch.device("cpu"), sd_ops)
    second = load_state_dict("/tmp/model.safetensors", loader, registry, torch.device("cpu"), sd_ops)

    assert first is second
    assert loader.calls == 1


def test_state_dict_registry_keys_include_sd_ops():
    loader = _FakeLoader()
    registry = StateDictRegistry()

    first = load_state_dict("/tmp/model.safetensors", loader, registry, torch.device("cpu"), SDOps("a"))
    second = load_state_dict("/tmp/model.safetensors", loader, registry, torch.device("cpu"), SDOps("b"))

    assert first is not second
    assert loader.calls == 2


def test_dummy_registry_does_not_cache_loads():
    loader = _FakeLoader()
    registry = DummyRegistry()
    sd_ops = SDOps("fake")

    first = load_state_dict("/tmp/model.safetensors", loader, registry, torch.device("cpu"), sd_ops)
    second = load_state_dict("/tmp/model.safetensors", loader, registry, torch.device("cpu"), sd_ops)

    assert first is not second
    assert loader.calls == 2

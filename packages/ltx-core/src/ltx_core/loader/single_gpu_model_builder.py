import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Generic, Iterator

import torch
from torch import nn

from ltx_core.accelerator import get_default_device
from ltx_core.loader.fuse_loras import FuseRule, apply_loras, bf16_fuse_rule
from ltx_core.loader.helpers import create_meta_model, load_state_dict, read_model_config
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import (
    LoRAAdaptableProtocol,
    LoraPathStrengthAndSDOps,
    LoraStateDictWithStrength,
    ModelBuilderProtocol,
    StateDict,
    StateDictLoader,
)
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
from ltx_core.model.model_protocol import ModelConfigurator, ModelType

logger: logging.Logger = logging.getLogger(__name__)


def _profile_detail_enabled() -> bool:
    return os.environ.get("LTX2_ASCEND_PROFILE_DETAIL", "").lower() in {"1", "true", "yes", "on"}


@contextmanager
def _profile_detail_section(name: str) -> Iterator[None]:
    if not _profile_detail_enabled():
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("[profile-detail] model_builder.%s %.3fs", name, time.perf_counter() - start)


def _check_uninitialized(model: nn.Module) -> list[str]:
    """Return names of any parameters/buffers still on meta device."""
    names = []
    for name, param in model.named_parameters():
        if str(param.device) == "meta":
            names.append(name)
    for name, buf in model.named_buffers():
        if str(buf.device) == "meta":
            names.append(name)
    return names


def _load_model_weights(
    meta_model: nn.Module,
    model_path: str | tuple[str, ...],
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    loader: StateDictLoader,
    registry: Registry,
    device: torch.device,
    dtype: torch.dtype | None,
    model_sd_ops: SDOps | None = None,
    lora_load_device: torch.device | None = None,
    fuse_rule: FuseRule = bf16_fuse_rule,
) -> None:
    """Load base weights and fuse LoRAs into *meta_model* in-place."""
    if lora_load_device is None:
        lora_load_device = device

    with _profile_detail_section("load_base"):
        model_sd = load_state_dict(model_path, loader, registry, device, model_sd_ops)

    lora_strengths = [lora.strength for lora in loras]
    if not lora_strengths or (min(lora_strengths) == 0 and max(lora_strengths) == 0):
        sd = model_sd.sd
        if dtype is not None:
            with _profile_detail_section("cast_base"):
                sd = {key: value.to(dtype=dtype) for key, value in sd.items()}
        with _profile_detail_section("assign_base"):
            meta_model.load_state_dict(sd, strict=False, assign=True)
        return

    with _profile_detail_section("load_loras"):
        lora_state_dicts = [
            load_state_dict([lora.path], loader, registry, lora_load_device, lora.sd_ops) for lora in loras
        ]
    lora_sd_and_strengths = [
        LoraStateDictWithStrength(sd, strength) for sd, strength in zip(lora_state_dicts, lora_strengths, strict=True)
    ]
    with _profile_detail_section("apply_loras"):
        final_sd = apply_loras(
            model_sd=model_sd,
            lora_sd_and_strengths=lora_sd_and_strengths,
            fuse_rule=fuse_rule,
            destination_sd=model_sd if isinstance(registry, DummyRegistry) else None,
        )
    fused_sd = final_sd.sd
    if dtype is not None:
        with _profile_detail_section("cast_fused"):
            fused_sd = {key: value.to(dtype=dtype) for key, value in fused_sd.items()}
    with _profile_detail_section("assign_fused"):
        meta_model.load_state_dict(fused_sd, strict=False, assign=True)


@dataclass(frozen=True)
class SingleGPUModelBuilder(Generic[ModelType], ModelBuilderProtocol[ModelType], LoRAAdaptableProtocol):
    """
    Builder for PyTorch models residing on a single GPU.
    Attributes:
        model_class_configurator: Class responsible for constructing the model from a config dict.
        model_path: Path (or tuple of shard paths) to the model's `.safetensors` checkpoint(s).
        model_sd_ops: Optional state-dict operations applied when loading the model weights.
        module_ops: Sequence of module-level mutations applied to the meta model before weight loading.
        loras: Sequence of LoRA adapters (path, strength, optional sd_ops) to fuse into the model.
        model_loader: Strategy for loading state dicts from disk. Defaults to
            :class:`SafetensorsModelStateDictLoader`.
        registry: Cache for already-loaded state dicts. Defaults to :class:`DummyRegistry` (no caching).
        lora_load_device: Device used when loading LoRA weight tensors from disk. Defaults to
            ``torch.device("cpu")``, which keeps LoRA weights in CPU memory and transfers them to
            the target GPU sequentially during fusion, reducing peak GPU memory usage compared to
            loading all LoRA weights directly onto the GPU at once.
        fuse_rule: Per-policy LoRA merge rule. Defaults to ``bf16_fuse_rule``;
    """

    model_class_configurator: type[ModelConfigurator[ModelType]]
    model_path: str | tuple[str, ...]
    model_sd_ops: SDOps | None = None
    module_ops: tuple[ModuleOps, ...] = field(default_factory=tuple)
    loras: tuple[LoraPathStrengthAndSDOps, ...] = field(default_factory=tuple)
    model_loader: StateDictLoader = field(default_factory=SafetensorsModelStateDictLoader)
    registry: Registry = field(default_factory=DummyRegistry)
    lora_load_device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    fuse_rule: FuseRule = bf16_fuse_rule

    def lora(self, lora_path: str, strength: float, sd_ops: SDOps) -> "SingleGPUModelBuilder":
        return replace(self, loras=(*self.loras, LoraPathStrengthAndSDOps(lora_path, strength, sd_ops)))

    def with_sd_ops(self, sd_ops: SDOps | None) -> "SingleGPUModelBuilder":
        return replace(self, model_sd_ops=sd_ops)

    def with_module_ops(self, module_ops: tuple[ModuleOps, ...]) -> "SingleGPUModelBuilder":
        return replace(self, module_ops=module_ops)

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> "SingleGPUModelBuilder":
        return replace(self, loras=loras)

    def with_registry(self, registry: Registry) -> "SingleGPUModelBuilder":
        return replace(self, registry=registry)

    def with_lora_load_device(self, device: torch.device) -> "SingleGPUModelBuilder":
        return replace(self, lora_load_device=device)

    def with_fuse_rule(self, fuse_rule: FuseRule) -> "SingleGPUModelBuilder":
        return replace(self, fuse_rule=fuse_rule)

    def model_config(self) -> dict:
        return read_model_config(self.model_path, self.model_loader)

    def meta_model(self, config: dict, module_ops: tuple[ModuleOps, ...]) -> ModelType:
        return create_meta_model(self.model_class_configurator, config, module_ops)

    def load_sd(
        self, paths: list[str], registry: Registry, device: torch.device | None, sd_ops: SDOps | None = None
    ) -> StateDict:
        return load_state_dict(paths, self.model_loader, registry, device, sd_ops)

    def _return_model(self, meta_model: ModelType, device: torch.device) -> ModelType:
        uninitialized = _check_uninitialized(meta_model)
        if uninitialized:
            logger.warning(f"Uninitialized parameters or buffers: {uninitialized}")
            return meta_model
        return meta_model.to(device)

    def build(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs: object,  # noqa: ARG002
    ) -> ModelType:
        device = get_default_device() if device is None else device
        post_load_ops = tuple(op for op in self.module_ops if getattr(op.mutator, "_ltx2_post_load", False))
        load_device = torch.device("cpu") if post_load_ops or device.type == "npu" else device
        with _profile_detail_section("model_config"):
            config = self.model_config()
        with _profile_detail_section("meta_model"):
            meta_model = self.meta_model(config, self.module_ops)

        load_dtype = dtype
        if device.type == "npu" and load_dtype is None:
            load_dtype = torch.float16
        with _profile_detail_section("load_weights"):
            _load_model_weights(
                meta_model=meta_model,
                model_path=self.model_path,
                loras=self.loras,
                loader=self.model_loader,
                registry=self.registry,
                device=load_device,
                dtype=load_dtype,
                model_sd_ops=self.model_sd_ops,
                lora_load_device=self.lora_load_device,
                fuse_rule=self.fuse_rule,
            )
        with _profile_detail_section(f"return_model.{device.type}"):
            model = self._return_model(meta_model, load_device if post_load_ops else device)
        for op in post_load_ops:
            if op.matcher(model):
                with _profile_detail_section(f"post_load_op.{op.name}"):
                    model = op.mutator(model)
        return model

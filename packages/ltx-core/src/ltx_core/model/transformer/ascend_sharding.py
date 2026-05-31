from __future__ import annotations

import os

import torch

from ltx_core.accelerator import balanced_block_device_map, parse_device_list
from ltx_core.loader.module_ops import ModuleOps


def apply_layerwise_device_map(model: torch.nn.Module, devices: list[torch.device] | None = None) -> torch.nn.Module:
    blocks = getattr(model, "transformer_blocks", None)
    if blocks is None:
        return model
    if devices is None:
        devices = parse_device_list(os.getenv("LTX2_ASCEND_DEVICES"), default_count=4)
    if len(devices) <= 1:
        return model
    primary = devices[0]
    device_map = balanced_block_device_map(len(blocks), devices)
    model.block_device_map = device_map
    block_container_id = id(blocks)
    block_ids = {id(block) for block in blocks}
    for child in model.children():
        if id(child) != block_container_id and id(child) not in block_ids:
            child.to(primary)
    for idx, block in enumerate(blocks):
        block.to(device_map[idx])
    return model


def build_layerwise_device_map_op(devices: list[torch.device] | None = None) -> ModuleOps:
    label = ",".join(str(d) for d in devices) if devices is not None else "env"

    def matcher(model: torch.nn.Module) -> bool:
        return hasattr(model, "transformer_blocks")

    def mutator(model: torch.nn.Module) -> torch.nn.Module:
        return apply_layerwise_device_map(model, devices)

    mutator._ltx2_post_load = True
    return ModuleOps(name=f"ascend_layerwise_device_map_{label}", matcher=matcher, mutator=mutator)


def apply_gemma_layerwise_device_map(model: torch.nn.Module, devices: list[torch.device] | None = None) -> torch.nn.Module:
    if devices is None:
        devices = parse_device_list(os.getenv("LTX2_ASCEND_DEVICES"), default_count=4)
    if len(devices) <= 1:
        return model
    gemma_model = getattr(model, "model", None)
    inner = getattr(gemma_model, "model", None)
    language_model = getattr(inner, "language_model", None)
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return model

    primary = devices[0]
    device_map = balanced_block_device_map(len(layers), devices)
    model.gemma_layer_device_map = device_map
    model.to(primary)
    for idx, layer in enumerate(layers):
        layer.to(device_map[idx])
    return model


def build_gemma_layerwise_device_map_op(devices: list[torch.device] | None = None) -> ModuleOps:
    label = ",".join(str(d) for d in devices) if devices is not None else "env"

    def matcher(model: torch.nn.Module) -> bool:
        gemma_model = getattr(model, "model", None)
        inner = getattr(gemma_model, "model", None)
        language_model = getattr(inner, "language_model", None)
        return hasattr(language_model, "layers")

    def mutator(model: torch.nn.Module) -> torch.nn.Module:
        return apply_gemma_layerwise_device_map(model, devices)

    mutator._ltx2_post_load = True
    return ModuleOps(name=f"ascend_gemma_layerwise_device_map_{label}", matcher=matcher, mutator=mutator)

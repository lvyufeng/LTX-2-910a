"""Shared model-construction helpers used by both SingleGPUModelBuilder and StreamingModelBuilder."""

from __future__ import annotations

import logging
import os
import time
from typing import TypeVar

import torch
from torch import nn

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.registry import Registry
from ltx_core.loader.sd_ops import SDOps
from ltx_core.model.model_protocol import ModelConfigurator

_M = TypeVar("_M", bound=nn.Module)
logger = logging.getLogger(__name__)


def _profile_detail_enabled() -> bool:
    return os.environ.get("LTX2_ASCEND_PROFILE_DETAIL", "").lower() in {"1", "true", "yes", "on"}


def _sd_ops_name(sd_ops: SDOps | None) -> str:
    return sd_ops.name if sd_ops is not None else "<none>"


def load_state_dict(
    paths: str | tuple[str, ...] | list[str],
    loader: StateDictLoader,
    registry: Registry,
    device: torch.device | None,
    sd_ops: SDOps | None = None,
) -> StateDict:
    """Load a state dict from disk, using registry caching."""
    if isinstance(paths, str):
        path_list = [paths]
    elif isinstance(paths, tuple):
        path_list = list(paths)
    else:
        path_list = paths
    detail = _profile_detail_enabled()
    sd_ops_label = _sd_ops_name(sd_ops)
    cached = registry.get(path_list, sd_ops)
    if cached is not None:
        if detail:
            logger.info("[profile-detail] state_dict.cache_hit paths=%d sd_ops=%s", len(path_list), sd_ops_label)
        return cached
    if detail:
        logger.info("[profile-detail] state_dict.cache_miss paths=%d sd_ops=%s", len(path_list), sd_ops_label)
    start = time.perf_counter() if detail else 0.0
    result = loader.load(path_list, sd_ops=sd_ops, device=device)
    if detail:
        logger.info(
            "[profile-detail] state_dict.load paths=%d sd_ops=%s %.3fs",
            len(path_list),
            sd_ops_label,
            time.perf_counter() - start,
        )
    registry.add(path_list, sd_ops=sd_ops, state_dict=result)
    return result


def read_model_config(
    model_path: str | tuple[str, ...],
    loader: StateDictLoader,
) -> dict:
    """Read metadata from the first shard of a checkpoint."""
    first = model_path[0] if isinstance(model_path, tuple) else model_path
    return loader.metadata(first)


def create_meta_model(
    configurator: type[ModelConfigurator[_M]],
    config: dict,
    module_ops: tuple[ModuleOps, ...] = (),
) -> _M:
    """Create a model on the meta device and apply module operations."""
    with torch.device("meta"):
        model = configurator.from_config(config)
    for op in module_ops:
        if getattr(op.mutator, "_ltx2_post_load", False):
            continue
        if op.matcher(model):
            model = op.mutator(model)
    return model

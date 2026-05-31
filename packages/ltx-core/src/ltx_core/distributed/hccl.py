from __future__ import annotations

import os
from typing import TypeVar

import torch
import torch.distributed as dist

from ltx_core.accelerator import configure_npu_runtime

T = TypeVar("T")


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", rank()))


def is_rank0() -> bool:
    return rank() == 0


def init_hccl_if_requested(enable: bool, device_index: int | None = None) -> torch.device | None:
    if not enable:
        return None
    local = local_rank()
    selected = local if device_index is None else device_index
    configure_npu_runtime(selected)
    device = torch.device("npu", selected)
    torch.npu.set_device(device)
    if not is_distributed():
        dist.init_process_group("hccl")
    return device


def destroy_process_group() -> None:
    if is_distributed():
        dist.destroy_process_group()


def broadcast_object(value: T | None, src: int = 0) -> T:
    payload = [value]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]


def broadcast_tensor(value: torch.Tensor | None, *, device: torch.device, src: int = 0) -> torch.Tensor:
    meta = None
    if rank() == src:
        if value is None:
            meta = None
        else:
            meta = (tuple(value.shape), value.dtype)
    meta = broadcast_object(meta, src=src)
    if meta is None:
        return None
    shape, dtype = meta
    return broadcast_tensor_like(value, shape=shape, dtype=dtype, device=device, src=src)


def broadcast_tensor_like(
    value: torch.Tensor | None,
    *,
    shape: tuple[int, ...] | torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    src: int = 0,
) -> torch.Tensor:
    if rank() == src:
        if value is None:
            raise ValueError("source rank must provide a tensor")
        tensor = value.to(device=device, dtype=dtype).contiguous()
    else:
        tensor = torch.empty(tuple(shape), device=device, dtype=dtype)
    dist.broadcast(tensor, src=src)
    return tensor


def broadcast_tensor_tuple(
    value: tuple[torch.Tensor, ...] | None,
    *,
    device: torch.device,
    src: int = 0,
) -> tuple[torch.Tensor, ...] | None:
    length = len(value) if rank() == src and value is not None else None
    length = broadcast_object(length, src=src)
    if length is None:
        return None
    return tuple(broadcast_tensor(value[idx] if rank() == src else None, device=device, src=src) for idx in range(length))


def all_gather_tensor(value: torch.Tensor, *, device: torch.device) -> list[torch.Tensor]:
    value = value.to(device=device).contiguous()
    shape = tuple(value.shape)
    dtype = value.dtype
    shapes = [None for _ in range(world_size())]
    dist.all_gather_object(shapes, (shape, dtype))
    max_shape = tuple(max(s[i] for s, _ in shapes) for i in range(len(shape)))
    padded = torch.zeros(max_shape, device=device, dtype=dtype)
    slices = tuple(slice(0, dim) for dim in shape)
    padded[slices] = value
    gathered = [torch.empty(max_shape, device=device, dtype=dtype) for _ in range(world_size())]
    dist.all_gather(gathered, padded)
    return [tensor[tuple(slice(0, dim) for dim in s)].contiguous() for tensor, (s, _) in zip(gathered, shapes, strict=True)]

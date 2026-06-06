from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import torch
import torch.distributed as dist

from ltx_core.accelerator import configure_npu_runtime

T = TypeVar("T")


@dataclass(frozen=True)
class HCCLGroup:
    """Description of a HCCL process subgroup.

    ``process_group=None`` means the default/global process group.  ``group_rank``
    is subgroup-local; ``leader_global_rank`` is the global rank used as source
    for PyTorch collectives.
    """

    name: str
    ranks: tuple[int, ...]
    process_group: dist.ProcessGroup | None
    global_rank: int
    group_rank: int
    group_world_size: int
    leader_global_rank: int

    @property
    def is_member(self) -> bool:
        return self.group_rank >= 0

    @property
    def is_leader(self) -> bool:
        return self.is_member and self.global_rank == self.leader_global_rank


ProcessGroupLike = HCCLGroup | dist.ProcessGroup | None


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _global_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def _process_group(group: ProcessGroupLike) -> dist.ProcessGroup | None:
    if isinstance(group, HCCLGroup):
        return group.process_group
    return group


def _skip_group_collective(group: ProcessGroupLike) -> bool:
    return isinstance(group, HCCLGroup) and not group.is_member


def _set_current_device(device: torch.device) -> None:
    """Keep HCCL object/tensor collectives on the rank's intended NPU.

    Some model-loading paths may change torch_npu's current device.  PyTorch
    object collectives create small internal tensors on the current accelerator,
    so set it explicitly before every helper that has a target device.
    """

    if device.type == "npu":
        torch.npu.set_device(device)


def rank(group: ProcessGroupLike = None) -> int:
    if isinstance(group, HCCLGroup):
        return group.group_rank
    if group is not None and is_distributed():
        return dist.get_rank(group=group)
    return _global_rank()


def world_size(group: ProcessGroupLike = None) -> int:
    if isinstance(group, HCCLGroup):
        return group.group_world_size
    if group is not None and is_distributed():
        return dist.get_world_size(group=group)
    return dist.get_world_size() if is_distributed() else 1


def local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", _global_rank()))


def is_rank0(group: ProcessGroupLike = None) -> bool:
    if isinstance(group, HCCLGroup):
        return group.is_leader
    return rank(group) == 0


def default_hccl_group(name: str = "default") -> HCCLGroup:
    size = world_size()
    current_global_rank = _global_rank()
    return HCCLGroup(
        name=name,
        ranks=tuple(range(size)),
        process_group=None,
        global_rank=current_global_rank,
        group_rank=current_global_rank,
        group_world_size=size,
        leader_global_rank=0,
    )


def create_hccl_group(name: str, ranks: Sequence[int]) -> HCCLGroup:
    if not ranks:
        raise ValueError("HCCL group ranks must not be empty")
    rank_tuple = tuple(int(r) for r in ranks)
    if len(set(rank_tuple)) != len(rank_tuple):
        raise ValueError(f"HCCL group {name!r} has duplicate ranks: {rank_tuple}")
    current_global_rank = _global_rank()
    group_rank = rank_tuple.index(current_global_rank) if current_global_rank in rank_tuple else -1
    process_group = None
    if is_distributed():
        process_group = dist.new_group(ranks=list(rank_tuple))
    elif rank_tuple != (0,):
        raise RuntimeError("cannot create a nontrivial HCCL group before distributed initialization")
    return HCCLGroup(
        name=name,
        ranks=rank_tuple,
        process_group=process_group,
        global_rank=current_global_rank,
        group_rank=group_rank,
        group_world_size=len(rank_tuple),
        leader_global_rank=rank_tuple[0],
    )


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


def broadcast_object(value: T | None, src: int = 0, group: ProcessGroupLike = None) -> T:
    if not is_distributed():
        return value
    if _skip_group_collective(group):
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=src, group=_process_group(group))
    return payload[0]


def broadcast_tensor(
    value: torch.Tensor | None,
    *,
    device: torch.device,
    src: int = 0,
    group: ProcessGroupLike = None,
) -> torch.Tensor:
    _set_current_device(device)
    if not is_distributed():
        return None if value is None else value.to(device=device).contiguous()
    if _skip_group_collective(group):
        return None
    meta = None
    if _global_rank() == src:
        if value is None:
            meta = None
        else:
            meta = (tuple(value.shape), value.dtype)
    meta = broadcast_object(meta, src=src, group=group)
    if meta is None:
        return None
    shape, dtype = meta
    return broadcast_tensor_like(value, shape=shape, dtype=dtype, device=device, src=src, group=group)


def broadcast_tensor_like(
    value: torch.Tensor | None,
    *,
    shape: tuple[int, ...] | torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    src: int = 0,
    group: ProcessGroupLike = None,
) -> torch.Tensor:
    _set_current_device(device)
    if not is_distributed():
        if value is None:
            return torch.empty(tuple(shape), device=device, dtype=dtype)
        return value.to(device=device, dtype=dtype).contiguous()
    if _skip_group_collective(group):
        return None
    if _global_rank() == src:
        if value is None:
            raise ValueError("source rank must provide a tensor")
        tensor = value.to(device=device, dtype=dtype).contiguous()
    else:
        tensor = torch.empty(tuple(shape), device=device, dtype=dtype)
    dist.broadcast(tensor, src=src, group=_process_group(group))
    return tensor


def broadcast_tensor_from_global_rank(
    value: torch.Tensor | None,
    *,
    source_global_rank: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Broadcast a tensor over the default/global group for cross-stage handoff.

    This is intentionally not a TP subgroup collective: all ranks in the default
    HCCL group must call it in the same order.
    """

    return broadcast_tensor(value, device=device, src=source_global_rank, group=None)


def broadcast_tensor_tuple(
    value: tuple[torch.Tensor, ...] | None,
    *,
    device: torch.device,
    src: int = 0,
    group: ProcessGroupLike = None,
) -> tuple[torch.Tensor, ...] | None:
    if _skip_group_collective(group):
        return None
    length = len(value) if _global_rank() == src and value is not None else None
    length = broadcast_object(length, src=src, group=group)
    if length is None:
        return None
    return tuple(
        broadcast_tensor(
            value[idx] if _global_rank() == src and value is not None else None,
            device=device,
            src=src,
            group=group,
        )
        for idx in range(length)
    )


def all_gather_tensor(value: torch.Tensor, *, device: torch.device, group: ProcessGroupLike = None) -> list[torch.Tensor]:
    _set_current_device(device)
    value = value.to(device=device).contiguous()
    if not is_distributed():
        return [value]
    if _skip_group_collective(group):
        return []
    pg = _process_group(group)
    group_size = world_size(group)
    shape = tuple(value.shape)
    dtype = value.dtype
    shapes = [None for _ in range(group_size)]
    dist.all_gather_object(shapes, (shape, dtype), group=pg)
    max_shape = tuple(max(s[i] for s, _ in shapes) for i in range(len(shape)))
    padded = torch.zeros(max_shape, device=device, dtype=dtype)
    slices = tuple(slice(0, dim) for dim in shape)
    padded[slices] = value
    gathered = [torch.empty(max_shape, device=device, dtype=dtype) for _ in range(group_size)]
    dist.all_gather(gathered, padded, group=pg)
    return [tensor[tuple(slice(0, dim) for dim in s)].contiguous() for tensor, (s, _) in zip(gathered, shapes, strict=True)]

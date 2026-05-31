from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.model.upsampler.model import LatentUpsampler
from ltx_core.model.upsampler.pixel_shuffle import PixelShuffleND
from ltx_core.model.upsampler.res_block import ResBlock
from ltx_core.model.upsampler.spatial_rational_resampler import SpatialRationalResampler


def _shard_range(size: int, rank: int, world_size: int) -> slice:
    if size % world_size != 0:
        raise ValueError(f"cannot shard dimension {size} across {world_size} ranks")
    local = size // world_size
    return slice(rank * local, (rank + 1) * local)


def _to_device_safe(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    dtype = torch.float16 if device.type == "npu" else value.dtype
    if value.dtype != dtype:
        value = value.to(dtype=dtype)
    if not value.is_contiguous():
        value = value.contiguous()
    return value.to(device=device, non_blocking=True)


def _all_gather_channels(value: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value.contiguous())
    return torch.cat(gathered, dim=1).contiguous()


def _copy_conv_rows(source: nn.Conv2d | nn.Conv3d, rows: slice, device: torch.device) -> nn.Conv2d | nn.Conv3d:
    conv_cls = nn.Conv2d if isinstance(source, nn.Conv2d) else nn.Conv3d
    out_channels = rows.stop - rows.start
    target = conv_cls(
        in_channels=source.in_channels,
        out_channels=out_channels,
        kernel_size=source.kernel_size,
        stride=source.stride,
        padding=source.padding,
        dilation=source.dilation,
        groups=source.groups,
        bias=source.bias is not None,
        padding_mode=source.padding_mode,
        device="meta",
        dtype=torch.float16 if device.type == "npu" else source.weight.dtype,
    )
    target.weight = nn.Parameter(_to_device_safe(source.weight[rows], device))
    if source.bias is not None:
        target.bias = nn.Parameter(_to_device_safe(source.bias[rows], device))
    return target


def _copy_group_norm_channels(source: nn.GroupNorm, channels: slice, device: torch.device) -> nn.GroupNorm:
    local_channels = channels.stop - channels.start
    group_size = source.num_channels // source.num_groups
    if local_channels % group_size != 0:
        raise ValueError(
            f"cannot shard GroupNorm({source.num_groups}, {source.num_channels}) into {local_channels} channels"
        )
    local_groups = local_channels // group_size
    target = nn.GroupNorm(local_groups, local_channels, eps=source.eps, affine=source.affine, device="meta")
    if source.affine:
        target.weight = nn.Parameter(_to_device_safe(source.weight[channels], device))
        target.bias = nn.Parameter(_to_device_safe(source.bias[channels], device))
    return target


class HCCLRowParallelConv(nn.Module):
    def __init__(self, source: nn.Conv2d | nn.Conv3d, rows: slice, *, device: torch.device, gather_input: bool) -> None:
        super().__init__()
        self.conv = _copy_conv_rows(source, rows, device)
        self.device = device
        self.gather_input = gather_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if self.gather_input:
            x = _all_gather_channels(x)
        return self.conv(x)


class HCCLChannelParallelResBlock(nn.Module):
    def __init__(self, source: ResBlock, channels: slice, *, device: torch.device) -> None:
        super().__init__()
        self.conv1 = HCCLRowParallelConv(source.conv1, channels, device=device, gather_input=True)
        self.norm1 = _copy_group_norm_channels(source.norm1, channels, device)
        self.conv2 = HCCLRowParallelConv(source.conv2, channels, device=device, gather_input=True)
        self.norm2 = _copy_group_norm_channels(source.norm2, channels, device)
        self.activation = nn.SiLU()
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x.to(self.device)
        x = self.conv1(residual)
        x = self.norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.activation(x + residual)


class HCCLChannelParallelSpatialRationalResampler(nn.Module):
    def __init__(self, source: SpatialRationalResampler, channels: slice, *, device: torch.device) -> None:
        super().__init__()
        conv_rows = slice(channels.start * source.num**2, channels.stop * source.num**2)
        self.conv = HCCLRowParallelConv(source.conv, conv_rows, device=device, gather_input=True)
        self.pixel_shuffle = source.pixel_shuffle
        self.blur_down = source.blur_down.to(device)
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from einops import rearrange

        b, _, f, _, _ = x.shape
        x = rearrange(x.to(self.device), "b c f h w -> (b f) c h w")
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.blur_down(x)
        return rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)


class HCCLChannelParallelLatentUpsampler(nn.Module):
    def __init__(self, source: LatentUpsampler, *, rank: int, world_size: int, device: torch.device) -> None:
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.in_channels = source.in_channels
        self.mid_channels = source.mid_channels
        self.num_blocks_per_stage = source.num_blocks_per_stage
        self.dims = source.dims
        self.spatial_upsample = source.spatial_upsample
        self.temporal_upsample = source.temporal_upsample
        self.spatial_scale = source.spatial_scale
        self.rational_resampler = source.rational_resampler

        self.mid_slice = _shard_range(source.mid_channels, rank, world_size)
        self.final_slice = _shard_range(source.in_channels, rank, world_size)
        self.initial_conv = HCCLRowParallelConv(source.initial_conv, self.mid_slice, device=device, gather_input=False)
        self.initial_norm = _copy_group_norm_channels(source.initial_norm, self.mid_slice, device)
        self.initial_activation = nn.SiLU()
        self.res_blocks = nn.ModuleList(
            [HCCLChannelParallelResBlock(block, self.mid_slice, device=device) for block in source.res_blocks]
        )
        self.upsampler = self._build_upsampler(source.upsampler)
        self.post_upsample_res_blocks = nn.ModuleList(
            [HCCLChannelParallelResBlock(block, self.mid_slice, device=device) for block in source.post_upsample_res_blocks]
        )
        self.final_conv = HCCLRowParallelConv(source.final_conv, self.final_slice, device=device, gather_input=True)

    def _build_upsampler(self, source: nn.Module) -> nn.Module:
        if isinstance(source, SpatialRationalResampler):
            return HCCLChannelParallelSpatialRationalResampler(source, self.mid_slice, device=self.device)
        if isinstance(source, nn.Sequential) and len(source) == 2 and isinstance(source[1], PixelShuffleND):
            conv = source[0]
            if not isinstance(conv, (nn.Conv2d, nn.Conv3d)):
                raise TypeError(f"unsupported upsampler conv {type(conv)!r}")
            factor = conv.out_channels // self.mid_channels
            rows = slice(self.mid_slice.start * factor, self.mid_slice.stop * factor)
            return nn.Sequential(
                HCCLRowParallelConv(conv, rows, device=self.device, gather_input=True),
                source[1],
            )
        raise TypeError(f"unsupported latent upsampler module {type(source)!r}")

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        from einops import rearrange

        b, _, f, _, _ = latent.shape
        x = self.initial_conv(latent.to(self.device))
        x = self.initial_norm(x)
        x = self.initial_activation(x)

        for block in self.res_blocks:
            x = block(x)

        if self.temporal_upsample:
            x = self.upsampler(x)
            x = x[:, :, 1:, :, :]
        elif isinstance(self.upsampler, HCCLChannelParallelSpatialRationalResampler):
            x = self.upsampler(x)
        else:
            x = rearrange(x, "b c f h w -> (b f) c h w")
            x = self.upsampler(x)
            x = rearrange(x, "(b f) c h w -> b c f h w", b=b, f=f)

        for block in self.post_upsample_res_blocks:
            x = block(x)

        x = self.final_conv(x)
        return _all_gather_channels(x)


def apply_hccl_upsampler_tensor_parallel(
    model: nn.Module,
    *,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | None = None,
) -> nn.Module:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("HCCL upsampler tensor parallelism requires torch.distributed.init_process_group('hccl')")
    rank = dist.get_rank() if rank is None else rank
    world_size = dist.get_world_size() if world_size is None else world_size
    if device is None:
        device = torch.device("npu", int(__import__("os").environ.get("LOCAL_RANK", rank)))
    if world_size <= 1:
        return model.to(device)
    if not isinstance(model, LatentUpsampler):
        return model.to(device)
    wrapped = HCCLChannelParallelLatentUpsampler(model, rank=rank, world_size=world_size, device=device)
    wrapped.tensor_parallel = True
    wrapped.tensor_parallel_rank = rank
    wrapped.tensor_parallel_world_size = world_size
    wrapped.tensor_parallel_device = device
    return wrapped.eval()


def build_hccl_upsampler_tensor_parallel_op(
    *,
    rank: int | None = None,
    world_size: int | None = None,
    device: torch.device | None = None,
) -> ModuleOps:
    label = f"rank{rank if rank is not None else 'env'}_world{world_size if world_size is not None else 'env'}"

    def matcher(model: nn.Module) -> bool:
        return isinstance(model, LatentUpsampler)

    def mutator(model: nn.Module) -> nn.Module:
        return apply_hccl_upsampler_tensor_parallel(model, rank=rank, world_size=world_size, device=device)

    mutator._ltx2_post_load = True
    return ModuleOps(name=f"ascend_hccl_upsampler_tensor_parallel_{label}", matcher=matcher, mutator=mutator)

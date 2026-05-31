from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RandomTensorGenerator:
    """Generator wrapper that can draw random tensors on a staging device.

    ``torch.Generator`` must match the device passed to ``torch.randn``.  For
    reference comparisons across devices, this wrapper lets callers draw on a
    deterministic staging device (for example CPU) and then copy/cast the tensor
    to the model device.
    """

    generator: torch.Generator
    draw_device: torch.device
    draw_dtype: torch.dtype | None = None

    @property
    def device(self) -> torch.device:
        return self.draw_device


GeneratorLike = torch.Generator | RandomTensorGenerator | None


def make_generator(
    seed: int,
    target_device: torch.device | str,
    *,
    draw_device: torch.device | str | None = None,
    draw_dtype: torch.dtype | None = None,
) -> torch.Generator | RandomTensorGenerator:
    target_device = torch.device(target_device)
    if draw_device is None and draw_dtype is None:
        return torch.Generator(device=target_device).manual_seed(seed)

    resolved_draw_device = torch.device(draw_device) if draw_device is not None else target_device
    return RandomTensorGenerator(
        generator=torch.Generator(device=resolved_draw_device).manual_seed(seed),
        draw_device=resolved_draw_device,
        draw_dtype=draw_dtype,
    )


def randn_tensor(
    shape: torch.Size | Sequence[int],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: GeneratorLike = None,
) -> torch.Tensor:
    target_device = torch.device(device)
    if isinstance(generator, RandomTensorGenerator):
        noise = torch.randn(
            shape,
            generator=generator.generator,
            dtype=generator.draw_dtype or dtype,
            device=generator.draw_device,
        )
    else:
        noise = torch.randn(shape, generator=generator, dtype=dtype, device=target_device)

    if noise.device != target_device or noise.dtype != dtype:
        noise = noise.to(device=target_device, dtype=dtype)
    return noise

import os

import torch

from ltx_core.debug import dump_tensor, should_dump_tensor
from ltx_core.model.transformer.gelu_approx import GELUApprox


_FP32_FF_ENV = "LTX2_TRANSFORMER_FF_FP32"


def fp32_feed_forward_enabled(x: torch.Tensor) -> bool:
    return x.device.type == "npu" and os.getenv(_FP32_FF_ENV, "").lower() in {"1", "true", "yes", "on"}


class FeedForward(torch.nn.Module):
    def __init__(self, dim: int, dim_out: int, mult: int = 4) -> None:
        super().__init__()
        inner_dim = int(dim * mult)
        project_in = GELUApprox(dim, inner_dim)

        self.net = torch.nn.Sequential(project_in, torch.nn.Identity(), torch.nn.Linear(inner_dim, dim_out))

    def _forward_fp32(self, x: torch.Tensor, *, debug_prefix: str, debug: bool) -> torch.Tensor:
        project_in = self.net[0]
        project_out = self.net[2]
        original_dtype = x.dtype

        hidden_linear = torch.nn.functional.linear(
            x.to(torch.float32),
            project_in.proj.weight.to(torch.float32),
            project_in.proj.bias.to(torch.float32) if project_in.proj.bias is not None else None,
        )
        if debug:
            dump_tensor(f"{debug_prefix}.project_in.linear_out", hidden_linear)
        hidden = torch.nn.functional.gelu(hidden_linear, approximate="tanh")
        if debug:
            dump_tensor(f"{debug_prefix}.project_in.gelu_out", hidden)
            dump_tensor(f"{debug_prefix}.project_in_out", hidden)
        out = torch.nn.functional.linear(
            hidden,
            project_out.weight.to(torch.float32),
            project_out.bias.to(torch.float32) if project_out.bias is not None else None,
        )
        if debug:
            dump_tensor(f"{debug_prefix}.project_out", out)
        return out.to(original_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        debug_prefix = getattr(self, "_debug_prefix", "")
        debug = bool(debug_prefix) and should_dump_tensor(debug_prefix)
        if fp32_feed_forward_enabled(x):
            return self._forward_fp32(x, debug_prefix=debug_prefix, debug=debug)

        project_in = self.net[0]
        if debug:
            setattr(project_in, "_debug_prefix", f"{debug_prefix}.project_in")
        try:
            hidden = project_in(x)
        finally:
            if debug and hasattr(project_in, "_debug_prefix"):
                delattr(project_in, "_debug_prefix")
        if debug:
            dump_tensor(f"{debug_prefix}.project_in_out", hidden)
        hidden = self.net[1](hidden)
        out = self.net[2](hidden)
        if debug:
            dump_tensor(f"{debug_prefix}.project_out", out)
        return out

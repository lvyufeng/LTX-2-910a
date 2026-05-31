import torch

from ltx_core.debug import dump_tensor, should_dump_tensor


class GELUApprox(torch.nn.Module):
    def __init__(self, dim_in: int, dim_out: int) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        debug_prefix = getattr(self, "_debug_prefix", "")
        debug = bool(debug_prefix) and should_dump_tensor(debug_prefix)
        projected = self.proj(x)
        if debug:
            dump_tensor(f"{debug_prefix}.linear_out", projected)
        out = torch.nn.functional.gelu(projected, approximate="tanh")
        if debug:
            dump_tensor(f"{debug_prefix}.gelu_out", out)
        return out

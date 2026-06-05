"""Optional custom AscendC operators for LTX-2 Ascend inference.

The package is intentionally optional.  Model code should import it lazily and
fall back to the pure PyTorch/torch_npu path when native operator symbols are not
installed.
"""

from . import long_k_softmax as long_k_softmax_ops
from .long_k_softmax import long_k_softmax
from .streaming_attention import availability_report, is_available, streaming_attention

__all__ = [
    "availability_report",
    "is_available",
    "long_k_softmax",
    "long_k_softmax_ops",
    "streaming_attention",
]

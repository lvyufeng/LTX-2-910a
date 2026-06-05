from __future__ import annotations

import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

try:
    import torch_npu  # type: ignore[import-untyped]
except Exception as exc:  # pragma: no cover - build-time failure path
    raise RuntimeError("torch_npu must be importable to build the streaming attention binding") from exc

ROOT = Path(__file__).resolve().parents[1]
TORCH_NPU_ROOT = Path(torch_npu.__file__).resolve().parent
TORCH_NPU_INCLUDE = TORCH_NPU_ROOT / "include"
TORCH_NPU_LIB = TORCH_NPU_ROOT / "lib"

CANN_ROOT = Path(os.getenv("ASCEND_HOME_PATH", "/usr/local/Ascend/cann-9.0.0"))
CANN_INCLUDE = CANN_ROOT / "include"
CANN_LIB = CANN_ROOT / "lib64"
CANN_AARCH64_LIB = CANN_ROOT / "aarch64-linux" / "lib64"
CUSTOM_OP_INCLUDE = ROOT / "ascendc" / "streaming_attention" / "generated" / "build_out" / "autogen"

include_dirs = [
    str(TORCH_NPU_INCLUDE),
    str(TORCH_NPU_INCLUDE / "third_party" / "acl" / "inc"),
    str(TORCH_NPU_INCLUDE / "third_party" / "op-plugin"),
    str(CANN_INCLUDE),
]
if CUSTOM_OP_INCLUDE.exists():
    include_dirs.append(str(CUSTOM_OP_INCLUDE))

setup(
    name="ltx2_ascend_ops_streaming_attention_binding",
    ext_modules=[
        CppExtension(
            name="ltx2_ascend_ops._streaming_attention_binding",
            sources=[str(ROOT / "csrc" / "streaming_attention_binding.cpp")],
            include_dirs=include_dirs,
            library_dirs=[str(TORCH_NPU_LIB), str(CANN_LIB), str(CANN_AARCH64_LIB)],
            libraries=["torch_npu", "ascendcl", "nnopbase", "dl"],
            runtime_library_dirs=[str(TORCH_NPU_LIB), str(CANN_LIB), str(CANN_AARCH64_LIB)],
            extra_compile_args={
                "cxx": [
                    "-O2",
                    "-std=c++17",
                    "-Wno-unused-parameter",
                    "-Wno-deprecated-declarations",
                ]
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension.with_options(no_python_abi_suffix=False)},
    package_dir={"": "src"},
    packages=["ltx2_ascend_ops"],
)

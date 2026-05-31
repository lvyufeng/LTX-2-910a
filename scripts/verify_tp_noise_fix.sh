#!/bin/bash
# Verify TP noise synchronization fix by running a smoke-sized two-stage pipeline.
# This exercises the initial_latent path (stage 2) that previously caused
# corrupted/flower-screen output due to rank-divergent noise.
#
# This is not a quality benchmark: the tiny resolution and 4 denoising steps are
# expected to look poor. Pass condition is no corrupted/flower-screen output.
#
# Usage: bash scripts/verify_tp_noise_fix.sh
# Requires: torchrun, 4x NPU, conda env ltx2-npu activated

set -euo pipefail

# Activate conda env for torchrun / torch_npu
eval "$(conda shell.bash hook)"
conda activate ltx2-npu

if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

export PYTHONPATH="/mnt/data/lvyufeng/LTX-2-910a/packages/ltx2-ascend/src:\
/mnt/data/lvyufeng/LTX-2-910a/packages/ltx-pipelines/src:\
/mnt/data/lvyufeng/LTX-2-910a/packages/ltx-core/src:\
${PYTHONPATH:-}"

CHECKPOINT="/mnt/data/LTX-2.3/ltx-2.3-22b-dev.safetensors"
GEMMA_ROOT="/mnt/data/gemma-3-12b-it-qat-q4_0-unquantized"
UPSCALER="/mnt/data/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DISTILLED_LORA="/mnt/data/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

OUTPUT_DIR="/mnt/data/lvyufeng/LTX-2-910a/outputs"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT="${OUTPUT_DIR}/tp_twostage_fix_${TIMESTAMP}.mp4"

echo "=== TP Two-Stage Noise Fix Smoke Verification ==="
echo "Output: ${OUTPUT}"
echo "This smoke run is not a quality benchmark; low aesthetic quality is expected."
echo ""

python - <<'PY'
import sys
try:
    import torch
    import torch_npu  # noqa: F401
    torch.npu.set_compile_mode(jit_compile=False)
    print(f"torch_npu available: {torch.npu.is_available()}")
except Exception as exc:
    print("Ascend runtime preflight failed:", exc, file=sys.stderr)
    print(
        "Source the matching CANN set_env.sh, preserve its PYTHONPATH entries, "
        "and keep NPU JIT disabled for this runtime.",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

/usr/bin/time -v torchrun \
    --standalone --nnodes=1 --nproc_per_node=4 \
    -m ltx2_ascend.cli \
    --quality-preset smoke \
    --pipeline two-stage \
    --tensor-parallel \
    --devices 0,1,2,3 \
    --checkpoint "${CHECKPOINT}" \
    --gemma-root "${GEMMA_ROOT}" \
    --spatial-upscaler "${UPSCALER}" \
    --distilled-lora "${DISTILLED_LORA}" \
    --output "${OUTPUT}" \
    --height 128 --width 192 --frames 9 \
    --steps 4 --seed 42 \
    --no-tiling \
    --prompt "A calm lake at sunrise with gentle ripples." \
    2>&1 | tee "${OUTPUT_DIR}/tp_twostage_fix_${TIMESTAMP}.log"

echo ""
echo "=== Done. Check ${OUTPUT} only for no corrupted/flower-screen output. ==="

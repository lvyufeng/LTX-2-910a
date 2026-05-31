#!/bin/bash
# Run the Ascend production wrapper with the same tiny parameters as
# scripts/run_cpu_reference_one_stage.sh. The 22B transformer does not fit on a
# single 32GB 910A, so this uses 4-card layerwise sharding.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ltx2-npu

REPO_ROOT="/mnt/data/lvyufeng/LTX-2-910a"
OUTPUT_DIR="${REPO_ROOT}/outputs/reference"
OUTPUT="${OUTPUT_DIR}/ascend_cli_layerwise_one_stage_seed10_32x32_f9_s1.mp4"

mkdir -p "${OUTPUT_DIR}"

if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

export PYTHONPATH="${REPO_ROOT}/packages/ltx-core/src:${REPO_ROOT}/packages/ltx-pipelines/src:${REPO_ROOT}/packages/ltx2-ascend/src:${PYTHONPATH:-}"
export LTX2_ASCEND_ATTENTION=eager
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export LTX2_DEBUG_DUMP_DIR="${LTX2_DEBUG_DUMP_DIR:-${OUTPUT_DIR}/dumps}"
export LTX2_DEBUG_DUMP_PREFIX="ascend"

python -m ltx2_ascend.cli \
    --quality-preset standard \
    --pipeline one-stage \
    --device 0 \
    --devices 0,1,2,3 \
    --layerwise \
    --checkpoint /mnt/data/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --gemma-root /mnt/data/gemma-3-12b-it-qat-q4_0-unquantized \
    --prompt "A cinematic shot of a calm mountain lake at sunrise." \
    --height 32 \
    --width 32 \
    --frames 9 \
    --steps 1 \
    --seed 10 \
    --random-draw-device cpu \
    --random-draw-dtype float16 \
    --video-cfg 1.0 \
    --video-stg 0.0 \
    --video-rescale 0.0 \
    --a2v 1.0 \
    --audio-cfg 1.0 \
    --audio-stg 0.0 \
    --audio-rescale 0.0 \
    --v2a 1.0 \
    --no-tiling \
    --output "${OUTPUT}"

echo "Ascend CLI layerwise reference written to ${OUTPUT}"

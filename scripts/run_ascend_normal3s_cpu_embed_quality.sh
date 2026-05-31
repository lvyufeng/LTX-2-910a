#!/bin/bash
# Full-quality NPU run with the CPU/fp32 embeddings-processor fix in place.
# Uses the standard preset's production guidance (video cfg~3.0 / audio cfg~7.0,
# STG, 30 steps) — NOT the stripped-down cfg=1.0 config used for CPU-reference
# matching. This is the "watch the result" run: does the fix produce a clean
# video on NPU under the real production settings?
#
# fp16 transformer stays on the 4 NPUs; only Gemma (bf16) + embeddings processor
# (fp32) run on CPU. Noise is CPU-staged for reproducibility. No debug dumps.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ltx2-npu

REPO_ROOT="/mnt/data/lvyufeng/LTX-2-910a"
OUTPUT_DIR="${REPO_ROOT}/outputs/reference"
OUTPUT="${OUTPUT_DIR}/normal3s_ascend_cpu_embed_quality_512x768_f73_s30_seed10.mp4"

mkdir -p "${OUTPUT_DIR}"

if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

export PYTHONPATH="${REPO_ROOT}/packages/ltx-core/src:${REPO_ROOT}/packages/ltx-pipelines/src:${REPO_ROOT}/packages/ltx2-ascend/src:${PYTHONPATH:-}"
export LTX2_ASCEND_ATTENTION=eager
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
# No LTX2_DEBUG_DUMP_DIR: skip tensor dumps for a clean, fast quality run.

# Guidance / step args are intentionally omitted so the standard preset supplies
# its production defaults.
python -m ltx2_ascend.cli \
    --quality-preset standard \
    --pipeline one-stage \
    --device 0 \
    --devices 0,1,2,3 \
    --layerwise \
    --checkpoint /mnt/data/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --gemma-root /mnt/data/gemma-3-12b-it-qat-q4_0-unquantized \
    --prompt "A cinematic shot of a calm mountain lake at sunrise." \
    --height 512 \
    --width 768 \
    --frames 73 \
    --seed 10 \
    --text-encoder-device cpu \
    --text-encoder-dtype bfloat16 \
    --embeddings-processor-device cpu \
    --embeddings-processor-dtype float32 \
    --random-draw-device cpu \
    --random-draw-dtype float16 \
    --output "${OUTPUT}"

echo "Ascend CPU-embeddings full-quality video written to ${OUTPUT}"

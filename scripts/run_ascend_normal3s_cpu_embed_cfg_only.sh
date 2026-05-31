#!/bin/bash
# NPU quality run with the CPU/fp32 embeddings fix and a safer guidance stack for 910A fp16.
# Keeps CFG + rescale, disables STG and cross-modal isolation guidance because those
# add extra fp16 transformer passes and amplify small pass-to-pass errors into visible noise.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ltx2-npu

REPO_ROOT="/mnt/data/lvyufeng/LTX-2-910a"
OUTPUT_DIR="${REPO_ROOT}/outputs/reference"
OUTPUT="${OUTPUT_DIR}/normal3s_ascend_cpu_embed_cfg_only_512x768_f73_s30_seed10.mp4"

mkdir -p "${OUTPUT_DIR}"

if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

export PYTHONPATH="${REPO_ROOT}/packages/ltx-core/src:${REPO_ROOT}/packages/ltx-pipelines/src:${REPO_ROOT}/packages/ltx2-ascend/src:${PYTHONPATH:-}"
export LTX2_ASCEND_ATTENTION=eager
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export LTX2_GUIDANCE_FP32=1

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
    --steps 30 \
    --seed 10 \
    --text-encoder-device cpu \
    --text-encoder-dtype bfloat16 \
    --embeddings-processor-device cpu \
    --embeddings-processor-dtype float32 \
    --random-draw-device cpu \
    --random-draw-dtype float16 \
    --video-cfg 3.0 \
    --audio-cfg 7.0 \
    --video-stg 0.0 \
    --audio-stg 0.0 \
    --video-rescale 0.7 \
    --audio-rescale 0.7 \
    --a2v 1.0 \
    --v2a 1.0 \
    --output "${OUTPUT}"

echo "Ascend CPU-embeddings CFG-only video written to ${OUTPUT}"

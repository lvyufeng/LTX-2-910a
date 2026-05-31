#!/bin/bash
# Full-quality NPU run with critical precision-sensitive parts kept on NPU but run in fp32:
# - transformer: NPU fp16, layerwise on 4 NPUs
# - embeddings processor: NPU fp32
# - video VAE decoder: NPU fp32
# - guidance combine: NPU fp32
# Gemma text encoder stays CPU/bf16 for now to avoid 12B text-encoder NPU memory risk.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ltx2-npu

REPO_ROOT="/mnt/data/lvyufeng/LTX-2-910a"
OUTPUT_DIR="${REPO_ROOT}/outputs/reference"
OUTPUT="${OUTPUT_DIR}/normal3s_ascend_npu_fp32_embed_decoder_512x768_f73_s30_seed10.mp4"

mkdir -p "${OUTPUT_DIR}"

if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

export PYTHONPATH="${REPO_ROOT}/packages/ltx-core/src:${REPO_ROOT}/packages/ltx-pipelines/src:${REPO_ROOT}/packages/ltx2-ascend/src:${PYTHONPATH:-}"
export LTX2_ASCEND_ATTENTION=eager
export LTX2_GUIDANCE_FP32=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
unset LTX2_DEBUG_DUMP_DIR
unset LTX2_DEBUG_DUMP_PREFIX
unset LTX2_DEBUG_DUMP_FILTER

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
    --embeddings-processor-device npu:0 \
    --embeddings-processor-dtype float32 \
    --video-decoder-device npu:0 \
    --video-decoder-dtype float32 \
    --random-draw-device cpu \
    --random-draw-dtype float16 \
    --output "${OUTPUT}"

echo "Ascend NPU-fp32 embeddings+decoder video written to ${OUTPUT}"

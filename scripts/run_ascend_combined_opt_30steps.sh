#!/bin/bash
# Full-quality 3s / 30-step NPU run with precision-preserving operator optimizations:
# - no skip-step, full 30 denoise steps
# - full CFG/STG/modality guidance passes
# - transformer fp16, embeddings processor NPU fp32, video decoder NPU fp32, guidance combine fp32
# - attention softmax stays fp16 on Ascend to avoid expensive fp32 score-matrix upcast
# - attention query chunk increased to 1024 for better 910A throughput

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ltx2-npu

REPO_ROOT="/mnt/data/lvyufeng/LTX-2-910a"
OUTPUT_DIR="${REPO_ROOT}/outputs/reference"
OUTPUT="${OUTPUT_DIR}/combined_opt_512x768_f73_s30_seed10.mp4"
mkdir -p "${OUTPUT_DIR}"

if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

export PYTHONPATH="${REPO_ROOT}/packages/ltx-core/src:${REPO_ROOT}/packages/ltx-pipelines/src:${REPO_ROOT}/packages/ltx2-ascend/src:${PYTHONPATH:-}"
export LTX2_ASCEND_ATTENTION=eager
export LTX2_ASCEND_SOFTMAX_FP16=1
export LTX2_ASCEND_ATTENTION_CHUNK=1024
export LTX2_GUIDANCE_FP32=1
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
unset LTX2_ASCEND_PROFILE
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

echo "Combined optimized 3s video written to ${OUTPUT}"

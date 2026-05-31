#!/bin/bash
# Normal-resolution (3s, 512x768, 73 frames, 8 steps) Ascend run with the
# embeddings processor forced to CPU/fp32 and Gemma on CPU/bf16, keeping the
# fp16 transformer on the 4 NPUs. Matches scripts that produced
# normal3s_cpu_512x768_f73_s8_seed10.mp4 so the two are directly comparable.
# This is the end-to-end confirmation that the CPU/fp32 prompt-context fix
# removes the 花屏 (garbled output) seen with the NPU/fp16 embeddings processor.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ltx2-npu

REPO_ROOT="/mnt/data/lvyufeng/LTX-2-910a"
OUTPUT_DIR="${REPO_ROOT}/outputs/reference"
OUTPUT="${OUTPUT_DIR}/normal3s_ascend_cpu_embed_512x768_f73_s8_seed10.mp4"

mkdir -p "${OUTPUT_DIR}"

if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

export PYTHONPATH="${REPO_ROOT}/packages/ltx-core/src:${REPO_ROOT}/packages/ltx-pipelines/src:${REPO_ROOT}/packages/ltx2-ascend/src:${PYTHONPATH:-}"
export LTX2_ASCEND_ATTENTION=eager
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export LTX2_DEBUG_DUMP_DIR="${LTX2_DEBUG_DUMP_DIR:-${OUTPUT_DIR}/normal3s_ascend_cpu_embed_dumps_s8}"
export LTX2_DEBUG_DUMP_PREFIX="${LTX2_DEBUG_DUMP_PREFIX:-ascend_normal3s_cpu_embed_s8}"
export LTX2_DEBUG_DUMP_FILTER="${LTX2_DEBUG_DUMP_FILTER:-prompt,diffusion_stage,denoise.step,video_decoder}"

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
    --steps 8 \
    --seed 10 \
    --text-encoder-device cpu \
    --text-encoder-dtype bfloat16 \
    --embeddings-processor-device cpu \
    --embeddings-processor-dtype float32 \
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
    --output "${OUTPUT}"

echo "Ascend CPU-embeddings normal-res video written to ${OUTPUT}"
echo "Dumps in ${LTX2_DEBUG_DUMP_DIR}"

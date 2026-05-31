#!/bin/bash
# A/B variant of run_ascend_reference_one_stage.sh that additionally moves the
# embeddings processor to CPU/fp32 (matching the CPU reference) while keeping the
# fp16 transformer on the NPUs. Used to confirm that the NPU/fp16 embeddings
# processor — not the FFN — is the dominant CPU/Ascend prompt-context divergence.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ltx2-npu

REPO_ROOT="/mnt/data/lvyufeng/LTX-2-910a"
OUTPUT_DIR="${REPO_ROOT}/outputs/reference"
OUTPUT="${OUTPUT_DIR}/ascend_cpu_embed_one_stage_seed10_32x32_f9_s1.mp4"

mkdir -p "${OUTPUT_DIR}"

if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

export PYTHONPATH="${REPO_ROOT}/packages/ltx-core/src:${REPO_ROOT}/packages/ltx-pipelines/src:${REPO_ROOT}/packages/ltx2-ascend/src:${PYTHONPATH:-}"
export LTX2_ASCEND_ATTENTION=eager
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export LTX2_DEBUG_DUMP_DIR="${LTX2_DEBUG_DUMP_DIR:-${OUTPUT_DIR}/dumps_ascend_cpu_embed}"
export LTX2_DEBUG_DUMP_PREFIX="${LTX2_DEBUG_DUMP_PREFIX:-ascend_cpu_embed}"
export LTX2_DEBUG_DUMP_FILTER="${LTX2_DEBUG_DUMP_FILTER:-prompt,transformer.block.0,diffusion_stage}"

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
    --no-tiling \
    --output "${OUTPUT}"

echo "Ascend CPU-embeddings reference written to ${OUTPUT}"
echo "Dumps in ${LTX2_DEBUG_DUMP_DIR}"

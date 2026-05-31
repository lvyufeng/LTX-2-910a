#!/bin/bash
# Run the original/shared LTX one-stage pipeline on CPU as a tiny correctness reference.
# This is not a quality benchmark: 32x32, 9 frames, and 1 denoising step are chosen
# to make a 22B CPU reference minimally feasible.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ltx2-npu

REPO_ROOT="/mnt/data/lvyufeng/LTX-2-910a"
OUTPUT_DIR="${REPO_ROOT}/outputs/reference"
OUTPUT="${OUTPUT_DIR}/cpu_one_stage_seed10_32x32_f9_s1.mp4"

mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${REPO_ROOT}/packages/ltx-core/src:${REPO_ROOT}/packages/ltx-pipelines/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM=false
export LTX2_DEBUG_DUMP_DIR="${LTX2_DEBUG_DUMP_DIR:-${OUTPUT_DIR}/dumps}"
export LTX2_DEBUG_DUMP_PREFIX="cpu"

python -m ltx_pipelines.ti2vid_one_stage \
    --device cpu \
    --dtype float32 \
    --text-encoder-dtype bfloat16 \
    --random-draw-device cpu \
    --random-draw-dtype float16 \
    --checkpoint-path /mnt/data/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --gemma-root /mnt/data/gemma-3-12b-it-qat-q4_0-unquantized \
    --prompt "A cinematic shot of a calm mountain lake at sunrise." \
    --height 32 \
    --width 32 \
    --num-frames 9 \
    --num-inference-steps 1 \
    --seed 10 \
    --video-cfg-guidance-scale 1.0 \
    --video-stg-guidance-scale 0.0 \
    --video-rescale-scale 0.0 \
    --a2v-guidance-scale 1.0 \
    --audio-cfg-guidance-scale 1.0 \
    --audio-stg-guidance-scale 0.0 \
    --audio-rescale-scale 0.0 \
    --v2a-guidance-scale 1.0 \
    --max-batch-size 1 \
    --output-path "${OUTPUT}"

echo "CPU reference written to ${OUTPUT}"

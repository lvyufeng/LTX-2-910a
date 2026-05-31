#!/bin/bash
# Tiny CPU full-guidance reference for diagnosing NPU residual noise under
# production guidance. Uses CPU/fp32 diffusion, CPU/bf16 Gemma, CPU/fp32
# embeddings processor, same CPU-staged fp16 noise, and dumps guidance pass
# outputs (cond/uncond/ptb/mod/pred) at step 0.

set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ltx2-npu

REPO_ROOT="/mnt/data/lvyufeng/LTX-2-910a"
OUTPUT_DIR="${REPO_ROOT}/outputs/reference"
OUTPUT="${OUTPUT_DIR}/guidance_diag_cpu_tiny_s1.mp4"
mkdir -p "${OUTPUT_DIR}"

export PYTHONPATH="${REPO_ROOT}/packages/ltx-core/src:${REPO_ROOT}/packages/ltx-pipelines/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export PYTHONHASHSEED=0
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export TOKENIZERS_PARALLELISM=false
export LTX2_DEBUG_DUMP_DIR="${OUTPUT_DIR}/guidance_diag_cpu_tiny"
export LTX2_DEBUG_DUMP_PREFIX="cpu_guided_tiny"
export LTX2_DEBUG_DUMP_FILTER="prompt,diffusion_stage,guidance.step.0"

python -m ltx_pipelines.ti2vid_one_stage \
    --device cpu \
    --dtype float32 \
    --text-encoder-dtype bfloat16 \
    --embeddings-processor-device cpu \
    --embeddings-processor-dtype float32 \
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
    --video-cfg-guidance-scale 3.0 \
    --video-stg-guidance-scale 1.0 \
    --video-rescale-scale 0.7 \
    --a2v-guidance-scale 3.0 \
    --audio-cfg-guidance-scale 7.0 \
    --audio-stg-guidance-scale 1.0 \
    --audio-rescale-scale 0.7 \
    --v2a-guidance-scale 3.0 \
    --max-batch-size 1 \
    --output-path "${OUTPUT}"

echo "CPU guidance diagnostic written to ${OUTPUT}"

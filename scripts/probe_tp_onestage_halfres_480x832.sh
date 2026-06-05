#!/bin/bash
# Diagnostic probe: run ONE-STAGE at the exact pixel size used by two-stage
# stage 1 for a 960x1664 target: height=480 width=832. This checks whether
# the artifact is intrinsic to the half-res latent grid under all-NPU TP, or
# specific to the two-stage wrapper.
set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate ltx2-npu

REPO_ROOT="/mnt/data/lvyufeng/LTX-2-910a"
OUTPUT_DIR="${REPO_ROOT}/outputs/reference"
mkdir -p "${OUTPUT_DIR}"

if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi

cd "${REPO_ROOT}"
export PYTHONPATH="packages/ltx-core/src:packages/ltx-pipelines/src:packages/ltx2-ascend/src:${PYTHONPATH:-}"
export PYTHONHASHSEED=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export LTX2_ASCEND_PROFILE=1
export LTX2_GUIDANCE_FP32=1
export LTX2_ASCEND_ATTENTION_CHUNK=1024
unset LTX2_ASCEND_SOFTMAX_FP16 || true
unset LTX2_TP_EMBEDDINGS_PROCESSOR || true
unset LTX2_DECODE_STAGE1 || true
unset LTX2_SKIP_STAGE2 || true

TS="$(date +%Y%m%d_%H%M%S)"
TAG="tp_onestage_probe_halfres_allnpu_480x832_f9_s2"
OUT="${OUTPUT_DIR}/${TAG}_${TS}.mp4"
LOG="${OUTPUT_DIR}/${TAG}_${TS}.log"

CMD=(torchrun --standalone --nnodes=1 --nproc_per_node=4 -m ltx2_ascend.cli
    --quality-preset standard
    --pipeline one-stage
    --tensor-parallel
    --device 0
    --devices 0,1,2,3
    --checkpoint /mnt/data/LTX-2.3/ltx-2.3-22b-dev.safetensors
    --gemma-root /mnt/data/gemma-3-12b-it-qat-q4_0-unquantized
    --prompt "A cinematic shot of a calm mountain lake at sunrise."
    --height 480 --width 832 --frames 9 --fps 24 --steps 2 --seed 10
    --embeddings-processor-dtype float32
    --video-decoder-dtype float32
    --output "${OUT}")

{
    echo "CMDLINE: LTX2_GUIDANCE_FP32=${LTX2_GUIDANCE_FP32} LTX2_ASCEND_ATTENTION_CHUNK=${LTX2_ASCEND_ATTENTION_CHUNK} LTX2_ASCEND_PROFILE=${LTX2_ASCEND_PROFILE} ${CMD[*]}"
    echo "OUT=${OUT}"
    echo "---"
} > "${LOG}"

set +e
/usr/bin/time -v "${CMD[@]}" 2>&1 | tee -a "${LOG}"
STATUS="${PIPESTATUS[0]}"
set -e

echo "STATUS=${STATUS}" | tee -a "${LOG}"
echo "LOG=${LOG}"
echo "OUT=${OUT}"
exit "${STATUS}"

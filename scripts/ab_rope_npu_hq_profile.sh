#!/bin/bash
# Phase-4 A/B for the opt-in NPU RoPE backend (LTX2_ASCEND_ROPE=npu).
#
# Runs the VERIFIED GOOD PATH unchanged -- two-stage-hq, hq preset, 960x1664,
# 121 frames, 24 fps, 4-card HCCL tensor parallel, fp32 embeddings + video +
# audio decoders (CLI defaults on NPU), LTX2_GUIDANCE_FP32=1, attention chunk
# 1536 (CLI default for the TP HQ path) -- with two differences only:
#   * --steps 5 (per-pass transformer cost is constant across steps, so 5 steps
#     characterize the RoPE timing delta at full HQ resolution without paying
#     for all 15), and
#   * LTX2_ASCEND_PROFILE=1 so the per-stage [profile] diffusion_stage.loop
#     lines are emitted.
#
# Pass A = control (LTX2_ASCEND_ROPE=eager -> PyTorch rope).
# Pass B = candidate/default (LTX2_ASCEND_ROPE=npu -> npu_rotary_mul, bit-exact).
#
# No known-bad flags. Everything stays on NPU. Distinct output files per pass.
# Compare diffusion_stage.loop timing + /usr/bin/time peak RSS between passes;
# bit-exactness is already proven by test_rope_npu_backend.py.
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

# Verified good-path runtime env (identical for both passes).
export LTX2_GUIDANCE_FP32=1
export LTX2_ASCEND_PROFILE=1
# Defensive: ensure no known-bad / experimental flags leak in from the shell.
unset LTX2_TP_PROMPT_RANK0_ONLY LTX2_DISABLE_TP_TEXT_ENCODER LTX2_ASCEND_EXPERIMENTAL_PRECISION \
      LTX2_ASCEND_VIDEO_DECODER_AUTOCAST LTX2_ASCEND_AUDIO_DECODER_LOWRES_AUTOCAST \
      LTX2_ASCEND_EMBEDDINGS_FEATURE_EXTRACTOR_AUTOCAST LTX2_TP_EMBEDDINGS_PROCESSOR \
      LTX2_ASCEND_SOFTMAX_FP16 LTX2_DECODE_STAGE1 LTX2_DEBUG_DUMP_DIR 2>/dev/null || true

CHECKPOINT="/mnt/data/LTX-2.3/ltx-2.3-22b-dev.safetensors"
GEMMA_ROOT="/mnt/data/gemma-3-12b-it-qat-q4_0-unquantized"
UPSCALER="/mnt/data/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
DISTILLED_LORA="/mnt/data/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"

PROMPT="A cinematic shot of a calm mountain lake at sunrise."
HEIGHT=960
WIDTH=1664
FRAMES=121
FPS=24
STEPS=5
SEED=10
TS="$(date +%Y%m%d_%H%M%S)"

run_pass() {
    local tag="$1" rope_mode="$2"
    local out="${OUTPUT_DIR}/rope_ab_${tag}_960x1664_f${FRAMES}_s${STEPS}_${TS}.mp4"
    local log="${OUTPUT_DIR}/rope_ab_${tag}_960x1664_f${FRAMES}_s${STEPS}_${TS}.log"

    local cmd=(torchrun --standalone --nnodes=1 --nproc_per_node=4 -m ltx2_ascend.cli
        --quality-preset hq
        --pipeline two-stage-hq
        --tensor-parallel
        --device 0
        --devices 0,1,2,3
        --checkpoint "${CHECKPOINT}"
        --spatial-upscaler "${UPSCALER}"
        --distilled-lora "${DISTILLED_LORA}"
        --gemma-root "${GEMMA_ROOT}"
        --prompt "${PROMPT}"
        --height "${HEIGHT}" --width "${WIDTH}" --frames "${FRAMES}" --fps "${FPS}"
        --steps "${STEPS}" --seed "${SEED}"
        --output "${out}")

    {
        echo "=== PASS ${tag} (LTX2_ASCEND_ROPE='${rope_mode}') ==="
        echo "CMDLINE: LTX2_ASCEND_ROPE='${rope_mode}' LTX2_GUIDANCE_FP32=${LTX2_GUIDANCE_FP32} LTX2_ASCEND_PROFILE=${LTX2_ASCEND_PROFILE} ${cmd[*]}"
        echo "OUT=${out}"
        echo "---"
    } > "${log}"

    echo "=== Running pass ${tag} (rope='${rope_mode}') -> ${out}"
    set +e
    if [ -n "${rope_mode}" ]; then
        LTX2_ASCEND_ROPE="${rope_mode}" /usr/bin/time -v "${cmd[@]}" 2>&1 | tee -a "${log}"
    else
        /usr/bin/time -v "${cmd[@]}" 2>&1 | tee -a "${log}"
    fi
    local status="${PIPESTATUS[0]}"
    set -e
    echo "STATUS=${status}" | tee -a "${log}"
    echo "LOG=${log}"
    echo "OUT=${out}"
    return "${status}"
}

echo "############ PASS A: control, LTX2_ASCEND_ROPE=eager ############"
run_pass "A_eager" "eager"

echo "############ PASS B: candidate, LTX2_ASCEND_ROPE=npu ############"
run_pass "B_npu" "npu"

echo ""
echo "=== A/B summary: diffusion_stage.loop per pass ==="
for f in "${OUTPUT_DIR}"/rope_ab_*_960x1664_f${FRAMES}_s${STEPS}_${TS}.log; do
    echo "--- ${f}"
    grep -E "\[profile\] diffusion_stage\.loop|Maximum resident set size|STATUS=|rope backend:" "${f}" || true
done

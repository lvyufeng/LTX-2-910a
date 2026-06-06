# LTX-2-910a

Inference-only Ascend 910A adaptation of Lightricks LTX-2.

This fork keeps the upstream `ltx-core` and `ltx-pipelines` runtime code, then adds Ascend-safe device selection and an FP16-first CLI. Training and CUDA-only optimization paths are intentionally out of scope.

## Hardware assumptions

- Current performance target: 8x Ascend 910A-class devices, 32GB HBM each.
- The validated high-quality path uses split-stage tensor parallelism: stage 1 stays resident on global ranks/devices `0,1,2,3`; stage 2, upsampler, and decode/output stay on ranks/devices `4,5,6,7`.
- Split-stage output defaults to the stage-2 leader, global rank `4`, so final latents do not need to move back to rank0.
- CANN runtime is available and `torch_npu` should be installed in the active Python environment.
- 910A inference uses FP16 transformer compute with fp32 precision kept for prompt embeddings processing, guidance combine, and video/audio decode. BF16/FP8/TensorRT-LLM paths are not selected for NPU execution.
- The 4-card tensor-parallel path remains available for regression/fallback, but the current speed target is the 8-card split-stage resident path.

## Ascend runtime setup

Activate the Python environment, source the CANN runtime environment, then prepend the project sources to the existing `PYTHONPATH` before launching inference. Do **not** replace `PYTHONPATH` with project paths only: CANN's `set_env.sh` adds Python paths that `torch_npu` may still need during lazy NPU initialization, even though this runtime disables NPU JIT compilation with `torch.npu.set_compile_mode(jit_compile=False)`.

```bash
conda activate ltx2-npu
if [ -f /usr/local/Ascend/cann-9.0.0/set_env.sh ]; then
  source /usr/local/Ascend/cann-9.0.0/set_env.sh
fi
export PYTHONPATH=packages/ltx-core/src:packages/ltx-pipelines/src:packages/ltx2-ascend/src:${PYTHONPATH:-}
python -m ltx2_ascend.cli --probe-only --device 0
```

If the probe fails during `torch.npu.set_device`, first confirm the CANN runtime has been sourced, the `cann_pythonpath` diagnostic is not missing CANN paths, and that `jit_compile=False` is reported before treating it as a model issue.

## Probe

```bash
export PYTHONPATH=packages/ltx-core/src:packages/ltx-pipelines/src:packages/ltx2-ascend/src:${PYTHONPATH:-}
python -m ltx2_ascend.cli --probe-only --device 0
```

## Quality inference entry point

Use `--quality-preset hq` and `--pipeline two-stage-hq` when judging visual quality. This path uses the full dev checkpoint, the spatial upscaler, the distilled LoRA on both stages, HQ guidance defaults, and 15 denoising steps.

The current validated performance path is 8-card split-stage TP-HQ with resident models: stage 1 on ranks `0,1,2,3`, stage 2 on ranks `4,5,6,7`, and output on rank `4`.

```bash
export PYTHONPATH=packages/ltx-core/src:packages/ltx-pipelines/src:packages/ltx2-ascend/src:${PYTHONPATH:-}
LTX2_GUIDANCE_FP32=1 \
/home/mseco/miniconda3/envs/ltx2-npu/bin/torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  -m ltx2_ascend.cli \
  --quality-preset hq \
  --pipeline two-stage-hq \
  --tensor-parallel \
  --tp-stage-split 0,1,2,3:4,5,6,7 \
  --devices 0,1,2,3,4,5,6,7 \
  --resident-models \
  --checkpoint /mnt/data/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --gemma-root /mnt/data/gemma-3-12b-it-qat-q4_0-unquantized \
  --spatial-upscaler /mnt/data/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --distilled-lora /mnt/data/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
  --prompt "A cinematic shot of a calm mountain lake at sunrise." \
  --height 960 --width 1664 --frames 121 --fps 24 \
  --steps 15 --repeat 2 \
  --output outputs/sample_hq.mp4
```

`--quality-preset hq` defaults to `1088x1920`, 121 frames, 15 steps, STG disabled, and HQ LoRA strengths `0.25/0.5`. The measured performance baseline below uses `960x1664x121` because it is the current validated full-size TP-HQ regression target on this 910A host.

## Current performance status

Validated high-quality target: `960x1664`, `121` frames, `24` fps, `15` HQ steps, `--repeat 2`, no profiling instrumentation in the timing run.

| Path | Full-size repeat timing | Notes |
|------|-------------------------|-------|
| Earlier 8-card split-stage resident baseline | run2 `323.14s` | After TP projection/reduction and in-place transformer/runtime optimizations. |
| Current 8-card split-stage resident path | run1 `627.79s`, run2 `319.72s` | Prompt embeddings repeat cache enabled by default; run2 excludes model-build cost and reuses resident stage models. |

Current profiled run2 breakdown with prompt cache enabled:

- rank0 / stage1 diffusion loop: about `193.26s`.
- rank4 / stage2 diffusion loop: about `105.05s`.
- prompt encoder on repeat: about `0.001s` after cache hit.
- video upsampler and audio decode are no longer material bottlenecks in resident repeat timing.

Default-enabled validated optimizations include 8-card split-stage HCCL subgroups, resident stage transformers, 8-rank TP Gemma text encoder with rank0 fp32 embeddings processor broadcast, prompt embeddings repeat cache, CANN `npu_rotary_mul` RoPE, TP q/k RMSNorm paired reductions, TP QKV/KV projection fusion, scoped TP-HQ chunk policy, output-bias in-place adds, inference-only residual/gate/AdaLN in-place updates, and small-batch output RGB-to-YUV conversion. `LTX2_PROMPT_EMBEDDINGS_CACHE=0` disables the prompt cache for debugging.

Optimizations that were tested but are **not** defaults because they were slower, unstable, unsupported on this 910A/CANN stack, or not fully validated include CANN fused attention, the current AscendC streaming-attention prototype, full-eager threshold expansion, packed FF reductions, RMSNorm temporary reuse, in-place TP RMSNorm scale, attention-score temporary mutation, and NPU batch guidance. Performance changes should only become defaults after unchanged video/audio quality and full-size validation.

## Memory-conscious standard inference

For a cheaper quality run, use the standard LTX-2.3 defaults. This keeps the one-stage pipeline unless you explicitly choose `--pipeline two-stage`, but it no longer uses smoke-test resolution or 4 denoising steps.

```bash
export PYTHONPATH=packages/ltx-core/src:packages/ltx-pipelines/src:packages/ltx2-ascend/src:${PYTHONPATH:-}
LTX2_ASCEND_ATTENTION=eager \
python -m ltx2_ascend.cli \
  --quality-preset standard \
  --pipeline one-stage \
  --device 0 \
  --devices 0,1,2,3 --layerwise \
  --checkpoint /mnt/data/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --gemma-root /mnt/data/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A cinematic shot of a calm mountain lake at sunrise." \
  --height 512 --width 768 --frames 121 --fps 24 \
  --steps 30 \
  --output outputs/sample_standard.mp4
```

Frame counts must be `8*k + 1`; valid examples include `9`, `17`, `97`, and `121`.

## CPU reference check

Use the original/shared one-stage LTX pipeline on CPU as a tiny correctness reference before comparing Ascend behavior. This is not a quality benchmark; it uses `32x32`, 9 frames, and 1 step to make the 22B checkpoint minimally feasible on CPU.

```bash
bash scripts/run_cpu_reference_one_stage.sh
bash scripts/run_ascend_reference_one_stage.sh
python scripts/compare_reference_outputs.py \
  --expected outputs/reference/cpu_one_stage_seed10_32x32_f9_s1.mp4 \
  --actual outputs/reference/ascend_shared_one_stage_seed10_32x32_f9_s1.mp4 \
  --require-same-shape \
  --max-frame-mean-abs-diff 8.0 \
  --max-frame-max-abs-diff 64.0
```

## Smoke test only

Use the smoke preset only for quick correctness/performance checks. It intentionally uses low resolution and 4 denoising steps, so the video may look bad even when the runtime is functioning correctly.

```bash
export PYTHONPATH=packages/ltx-core/src:packages/ltx-pipelines/src:packages/ltx2-ascend/src:${PYTHONPATH:-}
LTX2_ASCEND_ATTENTION=eager \
python -m ltx2_ascend.cli \
  --quality-preset smoke \
  --device 0 \
  --devices 0,1,2,3 --layerwise \
  --checkpoint /mnt/data/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --gemma-root /mnt/data/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt "A cinematic shot of a calm mountain lake at sunrise." \
  --output outputs/sample_smoke.mp4
```

`--spatial-upscaler` alone does not select a two-stage pipeline. Pass `--pipeline two-stage` or `--pipeline two-stage-hq`, or use `--quality-preset hq`, when you want the upscaler/refinement path.

`LTX2_ASCEND_ATTENTION=eager` is the correctness-first fallback. After a working end-to-end run, benchmark real Q/K/V shapes and replace this with an Ascend attention backend.

## 8-card split-stage TP-HQ mode

Use `--tensor-parallel --tp-stage-split 0,1,2,3:4,5,6,7 --devices 0,1,2,3,4,5,6,7 --resident-models` for the current validated high-quality performance path. TP collectives run inside stage-local 4-rank HCCL subgroups; cross-stage context/latent handoff uses NPU/HCCL tensor broadcasts in a fixed order. Host traffic is limited to shape/dtype metadata, logging, timing, and repeat barriers.

## 4-card fallback modes

The older `--devices 0,1,2,3 --layerwise` path remains useful for memory-conscious fallback and diagnostics. The 4-card tensor-parallel path without `--tp-stage-split` also remains supported for regression testing, but the current performance target is the 8-card resident split-stage path.

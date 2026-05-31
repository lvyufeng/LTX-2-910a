# LTX-2-910a

Inference-only Ascend 910A adaptation of Lightricks LTX-2.

This fork keeps the upstream `ltx-core` and `ltx-pipelines` runtime code, then adds Ascend-safe device selection and an FP16-first CLI. Training and CUDA-only optimization paths are intentionally out of scope.

## Hardware assumptions

- Current target: 4x Ascend 910A-class devices in one full-mesh group, 32GB HBM each, 128GB total.
- The second 4-card group can be used later for a second replica or larger sharding.
- CANN runtime is available and `torch_npu` should be installed in the active Python environment.
- 910A inference uses FP16. BF16/FP8/TensorRT-LLM paths are not selected for NPU execution.
- The default inference mesh is `(0,1,2,3)`.

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

```bash
export PYTHONPATH=packages/ltx-core/src:packages/ltx-pipelines/src:packages/ltx2-ascend/src:${PYTHONPATH:-}
LTX2_ASCEND_ATTENTION=eager \
python -m ltx2_ascend.cli \
  --quality-preset hq \
  --pipeline two-stage-hq \
  --device 0 \
  --devices 0,1,2,3 --layerwise \
  --checkpoint /mnt/data/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --gemma-root /mnt/data/gemma-3-12b-it-qat-q4_0-unquantized \
  --spatial-upscaler /mnt/data/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
  --distilled-lora /mnt/data/LTX-2.3/ltx-2.3-22b-distilled-lora-384-1.1.safetensors \
  --prompt "A cinematic shot of a calm mountain lake at sunrise." \
  --frames 121 --fps 24 \
  --steps 15 \
  --output outputs/sample_hq.mp4
```

`--quality-preset hq` defaults to `1088x1920`, 121 frames, 15 steps, STG disabled, and HQ LoRA strengths `0.25/0.5`. If full HQ resolution is too heavy for the current sharding setup, override `--height` and `--width` with smaller multiples of 64.

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

## 4-card full-mesh mode

Use `--devices 0,1,2,3 --layerwise` for the first performance target. This keeps the primary pipeline on `--device 0` but spreads transformer blocks across the 4-card full-mesh group. Activations move at block boundaries; after correctness is proven, the next optimization is replacing this with lower-overhead tensor/pipeline parallel kernels.

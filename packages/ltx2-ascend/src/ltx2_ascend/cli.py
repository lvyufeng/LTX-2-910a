from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ltx_core.accelerator import (
    DEFAULT_DTYPE,
    environment_report,
    get_default_device,
    hccs_group_for_device,
    npu_runtime_diagnostics,
    parse_device_list,
    require_npu_runtime,
)
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_pipelines.utils.args import ImageConditioningInput, parse_torch_dtype
from ltx_pipelines.utils.constants import (
    DEFAULT_NEGATIVE_PROMPT,
    LTX_2_3_HQ_PARAMS,
    LTX_2_3_PARAMS,
)

if TYPE_CHECKING:
    from ltx_pipelines.utils.constants import PipelineParams


_HQ_DISTILLED_LORA_STRENGTH_STAGE_1 = 0.25
_HQ_DISTILLED_LORA_STRENGTH_STAGE_2 = 0.5
_SMOKE_HEIGHT = 256
_SMOKE_WIDTH = 384
_SMOKE_FRAMES = 17
_SMOKE_FPS = 24.0
_SMOKE_STEPS = 4
_SMOKE_SEED = 0
_TWO_STAGE_PIPELINES = {"two-stage", "two-stage-hq", "distilled"}


def _parse_image(value: str) -> ImageConditioningInput:
    parts = value.rsplit(":", 2)
    path = value
    frame_idx = 0
    strength = 1.0
    if len(parts) == 2:
        try:
            frame_idx = int(parts[-1])
            path = parts[0]
        except ValueError:
            path = value
    elif len(parts) == 3:
        try:
            frame_idx = int(parts[-2])
            strength = float(parts[-1])
            path = parts[0]
        except ValueError as exc:
            raise ValueError(f"invalid image conditioning spec: {value!r}") from exc
    if not path:
        raise ValueError(f"invalid image conditioning spec with empty path: {value!r}")
    if frame_idx < 0:
        raise ValueError(f"image conditioning frame index must be >= 0: {value!r}")
    if strength <= 0:
        raise ValueError(f"image conditioning strength must be > 0: {value!r}")
    return ImageConditioningInput(path=path, frame_idx=frame_idx, strength=strength)


def _parse_images(values: list[str]) -> list[ImageConditioningInput]:
    return [_parse_image(value) for value in values]


def _default_dimensions(params: "PipelineParams", pipeline: str) -> tuple[int, int]:
    if pipeline in _TWO_STAGE_PIPELINES:
        return params.stage_2_height, params.stage_2_width
    return params.stage_1_height, params.stage_1_width


def _resolve_quality_defaults(args: argparse.Namespace) -> None:
    if args.quality_preset == "hq":
        params = LTX_2_3_HQ_PARAMS
        default_pipeline = "two-stage-hq"
    else:
        params = LTX_2_3_PARAMS
        default_pipeline = "one-stage"

    if args.pipeline is None:
        args.pipeline = default_pipeline

    if args.quality_preset == "smoke":
        default_height = _SMOKE_HEIGHT
        default_width = _SMOKE_WIDTH
        default_frames = _SMOKE_FRAMES
        default_fps = _SMOKE_FPS
        default_steps = _SMOKE_STEPS
        default_seed = _SMOKE_SEED
    else:
        default_height, default_width = _default_dimensions(params, args.pipeline)
        default_frames = params.num_frames
        default_fps = params.frame_rate
        default_steps = params.num_inference_steps
        default_seed = params.seed

    if args.height is None:
        args.height = default_height
    if args.width is None:
        args.width = default_width
    if args.frames is None:
        args.frames = default_frames
    if args.fps is None:
        args.fps = default_fps
    if args.steps is None:
        args.steps = default_steps
    elif args.pipeline == "distilled":
        args._steps_was_set = True
    if args.seed is None:
        args.seed = default_seed

    video_guider = params.video_guider_params
    audio_guider = params.audio_guider_params
    for name, value in (
        ("video_cfg", video_guider.cfg_scale),
        ("audio_cfg", audio_guider.cfg_scale),
        ("video_stg", video_guider.stg_scale),
        ("audio_stg", audio_guider.stg_scale),
        ("video_rescale", video_guider.rescale_scale),
        ("audio_rescale", audio_guider.rescale_scale),
        ("a2v", video_guider.modality_scale),
        ("v2a", audio_guider.modality_scale),
    ):
        if getattr(args, name) is None:
            setattr(args, name, value)
        elif args.pipeline == "distilled":
            setattr(args, f"_{name}_was_set", True)
    for name, value in (
        ("video_stg_block", list(video_guider.stg_blocks)),
        ("audio_stg_block", list(audio_guider.stg_blocks)),
    ):
        if getattr(args, name) is None:
            setattr(args, name, value)
        elif args.pipeline == "distilled":
            setattr(args, f"_{name}_was_set", True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ascend 910A FP16 inference CLI for LTX-2")
    parser.add_argument("--probe-only", action="store_true", help="print runtime/NPU environment and exit")
    parser.add_argument(
        "--quality-preset",
        choices=("standard", "hq", "smoke"),
        default="standard",
        help="generation defaults to use: standard quality, HQ two-stage quality, or fast smoke-test settings",
    )
    parser.add_argument(
        "--pipeline",
        choices=("one-stage", "two-stage", "two-stage-hq", "distilled"),
        default=None,
        help="pipeline to run; defaults to one-stage for standard/smoke and two-stage-hq for hq",
    )
    parser.add_argument("--device", type=int, default=0, help="primary NPU index to use for this process")
    parser.add_argument(
        "--text-encoder-device",
        default=None,
        help="device for Gemma text encoder; default uses the primary device",
    )
    parser.add_argument(
        "--devices",
        default="0,1,2,3",
        help="comma-separated NPU ids for layerwise/tensor-parallel sharding",
    )
    parser.add_argument("--layerwise", action="store_true", help="spread transformer blocks across --devices")
    parser.add_argument(
        "--text-encoder-layerwise",
        action="store_true",
        help="also shard Gemma text encoder layers across --devices",
    )
    parser.add_argument(
        "--tensor-parallel",
        action="store_true",
        help="run full-model HCCL tensor parallelism (launch with torchrun)",
    )
    parser.add_argument(
        "--resident-models",
        action="store_true",
        help="keep pipeline models loaded for repeated generations in one process",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "number of sequential generations in one process "
            "(seed increments per run; benefits from --resident-models)"
        ),
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=1,
        help=(
            "maximum transformer batch size; guided denoisers can batch up to 4 guidance passes "
            "when LTX2_NPU_BATCH_GUIDANCE=1"
        ),
    )
    parser.add_argument("--checkpoint", required=False, help="LTX-2 checkpoint safetensors path")
    parser.add_argument("--gemma-root", required=False, help="Gemma text encoder directory")
    parser.add_argument("--spatial-upscaler", required=False, help="LTX-2 spatial upscaler safetensors path")
    parser.add_argument(
        "--distilled-lora",
        action="append",
        default=[],
        help="distilled LoRA path[:strength] for two-stage pipelines, repeatable",
    )
    parser.add_argument(
        "--distilled-lora-strength-stage-1",
        type=float,
        default=_HQ_DISTILLED_LORA_STRENGTH_STAGE_1,
        help="two-stage-hq: distilled LoRA strength for stage 1",
    )
    parser.add_argument(
        "--distilled-lora-strength-stage-2",
        type=float,
        default=_HQ_DISTILLED_LORA_STRENGTH_STAGE_2,
        help="two-stage-hq: distilled LoRA strength for stage 2",
    )
    parser.add_argument("--lora", action="append", default=[], help="LoRA path[:strength], repeatable")
    parser.add_argument("--prompt", default="A cinematic shot of a calm mountain lake at sunrise.")
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--output", default="outputs/ltx2_ascend.mp4")
    parser.add_argument("--height", type=int, default=None, help="video height; preset default if omitted")
    parser.add_argument("--width", type=int, default=None, help="video width; preset default if omitted")
    parser.add_argument("--frames", type=int, default=None, help="number of frames; must be 8*k + 1")
    parser.add_argument("--fps", type=float, default=None, help="output frame rate; preset default if omitted")
    parser.add_argument("--steps", type=int, default=None, help="denoising steps; preset default if omitted")
    parser.add_argument("--seed", type=int, default=None, help="random seed; preset default if omitted")
    parser.add_argument(
        "--text-encoder-dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help=(
            "Override Gemma text encoder dtype independently from the inference dtype. "
            "Useful for CPU references where Gemma is numerically unstable in float32."
        ),
    )
    parser.add_argument(
        "--embeddings-processor-device",
        default=None,
        help=(
            "Override the prompt embeddings processor device independently from the inference device. "
            "Use cpu to align Ascend runs with CPU reference prompt processing."
        ),
    )
    parser.add_argument(
        "--embeddings-processor-dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help=(
            "Override the prompt embeddings processor dtype independently from the inference dtype. "
            "Use float32 on CPU to avoid NPU/FP16 prompt projection drift."
        ),
    )
    parser.add_argument(
        "--video-decoder-device",
        default=None,
        help=(
            "Override the video VAE decoder device independently from the inference device. "
            "Use cpu for fp32 decode when NPU/FP16 VAE decode is too noisy."
        ),
    )
    parser.add_argument(
        "--video-decoder-dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help="Override the video VAE decoder dtype independently from the inference dtype.",
    )
    parser.add_argument(
        "--random-draw-device",
        default=None,
        help=(
            "Draw stochastic tensors on this device before copying to the inference device. "
            "Use cpu for cross-device reference comparisons."
        ),
    )
    parser.add_argument(
        "--random-draw-dtype",
        choices=("float32", "float16", "bfloat16"),
        default=None,
        help=(
            "Draw stochastic tensors with this dtype before casting to the inference dtype. "
            "Use float16 to align CPU and FP16 NPU reference noise at the NPU precision."
        ),
    )
    parser.add_argument("--image", action="append", default=[], help="conditioning image path[:frame_idx[:strength]]")
    parser.add_argument("--enhance-prompt", action="store_true")
    parser.add_argument("--no-tiling", action="store_true")
    parser.add_argument("--video-cfg", type=float, default=None, help="video CFG scale; preset default if omitted")
    parser.add_argument("--audio-cfg", type=float, default=None, help="audio CFG scale; preset default if omitted")
    parser.add_argument("--video-stg", type=float, default=None, help="video STG scale; preset default if omitted")
    parser.add_argument("--audio-stg", type=float, default=None, help="audio STG scale; preset default if omitted")
    parser.add_argument(
        "--video-rescale",
        type=float,
        default=None,
        help="video rescale scale; preset default if omitted",
    )
    parser.add_argument(
        "--audio-rescale",
        type=float,
        default=None,
        help="audio rescale scale; preset default if omitted",
    )
    parser.add_argument(
        "--a2v",
        type=float,
        default=None,
        help="audio-to-video guidance scale; preset default if omitted",
    )
    parser.add_argument(
        "--v2a",
        type=float,
        default=None,
        help="video-to-audio guidance scale; preset default if omitted",
    )
    parser.add_argument(
        "--video-stg-block",
        action="append",
        type=int,
        default=None,
        help="video STG block; repeatable",
    )
    parser.add_argument(
        "--audio-stg-block",
        action="append",
        type=int,
        default=None,
        help="audio STG block; repeatable",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    args = build_parser().parse_args(argv)
    _resolve_quality_defaults(args)
    return args


def _parse_loras(values: list[str]) -> list[object]:
    from ltx_core.loader import LTXV_LORA_COMFY_RENAMING_MAP, LoraPathStrengthAndSDOps

    loras = []
    for value in values:
        if ":" in value:
            path, strength = value.rsplit(":", 1)
            loras.append(LoraPathStrengthAndSDOps(path, float(strength), LTXV_LORA_COMFY_RENAMING_MAP))
        else:
            loras.append(LoraPathStrengthAndSDOps(value, 1.0, LTXV_LORA_COMFY_RENAMING_MAP))
    return loras


def _guider_params(
    cfg: float,
    stg: float,
    rescale: float,
    modality: float,
    stg_blocks: list[int],
) -> MultiModalGuiderParams:
    return MultiModalGuiderParams(
        cfg_scale=cfg,
        stg_scale=stg,
        rescale_scale=rescale,
        modality_scale=modality,
        skip_step=0,
        stg_blocks=stg_blocks,
    )


def _validate_preset_pipeline_compatibility(args: argparse.Namespace) -> None:
    if args.quality_preset == "hq" and args.pipeline != "two-stage-hq":
        raise SystemExit("--quality-preset hq requires --pipeline two-stage-hq")
    if args.pipeline == "two-stage-hq" and args.quality_preset != "hq":
        raise SystemExit("--pipeline two-stage-hq requires --quality-preset hq")
    if args.quality_preset == "smoke" and args.pipeline in ("two-stage", "two-stage-hq"):
        logging.warning("smoke preset with %s is for mechanics only, not visual quality", args.pipeline)
    ignored = [
        flag
        for flag in (
            "steps",
            "video_cfg",
            "audio_cfg",
            "video_stg",
            "audio_stg",
            "video_rescale",
            "audio_rescale",
            "a2v",
            "v2a",
            "video_stg_block",
            "audio_stg_block",
        )
        if getattr(args, f"_{flag}_was_set", False)
    ]
    if args.pipeline == "distilled" and ignored:
        ignored_flags = ", ".join("--" + flag.replace("_", "-") for flag in ignored)
        logging.warning("--pipeline distilled ignores these explicit flags: %s", ignored_flags)


def _validate_generation_args(args: argparse.Namespace) -> None:
    from ltx_pipelines.utils.helpers import assert_num_frames, assert_resolution

    _validate_preset_pipeline_compatibility(args)

    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")
    if args.height < 1:
        raise SystemExit("--height must be >= 1")
    if args.width < 1:
        raise SystemExit("--width must be >= 1")
    if args.fps <= 0:
        raise SystemExit("--fps must be > 0")
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")

    try:
        assert_num_frames(args.frames)
        assert_resolution(args.height, args.width, is_two_stage=args.pipeline in _TWO_STAGE_PIPELINES)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None


def _warn_for_low_quality_settings(args: argparse.Namespace) -> None:
    if args.quality_preset == "smoke" or args.pipeline == "distilled":
        return

    if args.quality_preset == "hq":
        expected_steps = LTX_2_3_HQ_PARAMS.num_inference_steps
    else:
        expected_steps = LTX_2_3_PARAMS.num_inference_steps

    if args.steps < expected_steps:
        logging.warning(
            "--steps=%d is below the %s preset default (%d); video quality may be poor",
            args.steps,
            args.quality_preset,
            expected_steps,
        )


def _log_generation_profile(args: argparse.Namespace, tiling_enabled: bool) -> None:
    logging.info(
        "generation profile: preset=%s pipeline=%s size=%dx%d frames=%d fps=%s steps=%d seed=%d tiling=%s",
        args.quality_preset,
        args.pipeline,
        args.height,
        args.width,
        args.frames,
        args.fps,
        args.steps,
        args.seed,
        "on" if tiling_enabled else "off",
    )
    logging.info(
        "video guidance: cfg=%s stg=%s rescale=%s a2v=%s stg_blocks=%s",
        args.video_cfg,
        args.video_stg,
        args.video_rescale,
        args.a2v,
        args.video_stg_block,
    )
    logging.info(
        "audio guidance: cfg=%s stg=%s rescale=%s v2a=%s stg_blocks=%s",
        args.audio_cfg,
        args.audio_stg,
        args.audio_rescale,
        args.v2a,
        args.audio_stg_block,
    )


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.probe_only:
        print(environment_report(args.device))
        print("\nAscend NPU runtime diagnostics:")
        print("\n".join(npu_runtime_diagnostics(args.device)))
        return

    from ltx_core.distributed.hccl import destroy_process_group, init_hccl_if_requested, is_rank0, local_rank
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    from ltx_pipelines.distilled import DistilledPipeline
    from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
    from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
    from ltx_pipelines.ti2vid_two_stages_hq import TI2VidTwoStagesHQPipeline
    from ltx_pipelines.utils.media_io import encode_video
    from ltx_pipelines.utils.types import OffloadMode

    _validate_generation_args(args)
    _warn_for_low_quality_settings(args)

    required = ["checkpoint", "gemma_root"]
    if args.pipeline in ("two-stage", "two-stage-hq", "distilled"):
        required.append("spatial_upscaler")
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        missing_args = ", ".join("--" + name.replace("_", "-") for name in missing)
        raise SystemExit(f"missing required arguments for inference: {missing_args}")

    if args.pipeline in ("two-stage", "two-stage-hq") and not args.distilled_lora:
        raise SystemExit(f"--pipeline {args.pipeline} requires --distilled-lora <path[:strength]>")

    if args.tensor_parallel and args.layerwise:
        raise SystemExit("--tensor-parallel and --layerwise are mutually exclusive")

    tp_device_ids = parse_device_list(args.devices, default_count=4)
    preflight_device_indices = {args.device}
    if args.layerwise or args.tensor_parallel:
        preflight_device_indices.update(device.index or 0 for device in tp_device_ids if device.type == "npu")
    try:
        for device_index in sorted(preflight_device_indices):
            require_npu_runtime(device_index)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
    if args.tensor_parallel:
        rank = local_rank()
        if rank >= len(tp_device_ids):
            raise SystemExit(f"LOCAL_RANK {rank} exceeds --devices list {tp_device_ids}")
        device_index = tp_device_ids[rank].index or 0
        device = init_hccl_if_requested(True, device_index=device_index)
        group = hccs_group_for_device(device_index)
        logging.info(
            "rank %d using %s with HCCS group %s and dtype %s (tensor-parallel)",
            rank,
            device,
            group,
            DEFAULT_DTYPE,
        )
    else:
        device = get_default_device(args.device)
        if device.type == "npu":
            group = hccs_group_for_device(args.device)
            logging.info("using %s with HCCS group %s and dtype %s", device, group, DEFAULT_DTYPE)
        else:
            logging.info("using %s and dtype %s", device, DEFAULT_DTYPE)

    layerwise_devices = tp_device_ids if args.layerwise else None
    if layerwise_devices:
        logging.info("using layerwise transformer devices: %s", ",".join(str(d) for d in layerwise_devices))

    loras = tuple(_parse_loras(args.lora))
    distilled_loras = list(_parse_loras(args.distilled_lora))
    text_encoder_device = torch.device(args.text_encoder_device) if args.text_encoder_device else None
    text_encoder_layerwise_devices = layerwise_devices if args.text_encoder_layerwise else None
    random_draw_device = torch.device(args.random_draw_device) if args.random_draw_device else None
    random_draw_dtype = parse_torch_dtype(args.random_draw_dtype)
    text_encoder_dtype = parse_torch_dtype(args.text_encoder_dtype)
    embeddings_processor_device = (
        torch.device(args.embeddings_processor_device) if args.embeddings_processor_device else None
    )
    embeddings_processor_dtype = parse_torch_dtype(args.embeddings_processor_dtype)
    video_decoder_device = torch.device(args.video_decoder_device) if args.video_decoder_device else None
    video_decoder_dtype = parse_torch_dtype(args.video_decoder_dtype)
    if text_encoder_dtype is not None:
        logging.info("using text encoder dtype %s", text_encoder_dtype)
    if embeddings_processor_device is not None or embeddings_processor_dtype is not None:
        logging.info(
            "using embeddings processor device %s dtype %s",
            embeddings_processor_device or device,
            embeddings_processor_dtype or DEFAULT_DTYPE,
        )
    if video_decoder_device is not None or video_decoder_dtype is not None:
        logging.info(
            "using video decoder device %s dtype %s",
            video_decoder_device or device,
            video_decoder_dtype or DEFAULT_DTYPE,
        )
    if args.pipeline == "one-stage":
        pipeline = TI2VidOneStagePipeline(
            checkpoint_path=args.checkpoint,
            gemma_root=args.gemma_root,
            loras=loras,
            device=device,
            quantization=None,
            compilation_config=None,
            offload_mode=OffloadMode.NONE,
            layerwise_devices=layerwise_devices,
            text_encoder_device=text_encoder_device,
            text_encoder_dtype=text_encoder_dtype,
            embeddings_processor_device=embeddings_processor_device,
            embeddings_processor_dtype=embeddings_processor_dtype,
            video_decoder_device=video_decoder_device,
            video_decoder_dtype=video_decoder_dtype,
            text_encoder_layerwise_devices=text_encoder_layerwise_devices,
            tensor_parallel=args.tensor_parallel,
            resident_models=args.resident_models,
            random_draw_device=random_draw_device,
            random_draw_dtype=random_draw_dtype,
        )
    elif args.pipeline == "two-stage":
        pipeline = TI2VidTwoStagesPipeline(
            checkpoint_path=args.checkpoint,
            distilled_lora=distilled_loras,
            spatial_upsampler_path=args.spatial_upscaler,
            gemma_root=args.gemma_root,
            loras=loras,
            device=device,
            quantization=None,
            compilation_config=None,
            offload_mode=OffloadMode.NONE,
            layerwise_devices=layerwise_devices,
            text_encoder_device=text_encoder_device,
            text_encoder_dtype=text_encoder_dtype,
            embeddings_processor_device=embeddings_processor_device,
            embeddings_processor_dtype=embeddings_processor_dtype,
            text_encoder_layerwise_devices=text_encoder_layerwise_devices,
            tensor_parallel=args.tensor_parallel,
            resident_models=args.resident_models,
            random_draw_device=random_draw_device,
            random_draw_dtype=random_draw_dtype,
        )
    elif args.pipeline == "two-stage-hq":
        pipeline = TI2VidTwoStagesHQPipeline(
            checkpoint_path=args.checkpoint,
            distilled_lora=distilled_loras,
            distilled_lora_strength_stage_1=args.distilled_lora_strength_stage_1,
            distilled_lora_strength_stage_2=args.distilled_lora_strength_stage_2,
            spatial_upsampler_path=args.spatial_upscaler,
            gemma_root=args.gemma_root,
            loras=loras,
            device=device,
            quantization=None,
            compilation_config=None,
            offload_mode=OffloadMode.NONE,
            layerwise_devices=layerwise_devices,
            text_encoder_device=text_encoder_device,
            text_encoder_dtype=text_encoder_dtype,
            embeddings_processor_device=embeddings_processor_device,
            embeddings_processor_dtype=embeddings_processor_dtype,
            text_encoder_layerwise_devices=text_encoder_layerwise_devices,
            tensor_parallel=args.tensor_parallel,
            resident_models=args.resident_models,
        )
    else:
        if args.tensor_parallel:
            raise SystemExit("--tensor-parallel currently supports only --pipeline one-stage/two-stage/two-stage-hq")
        pipeline = DistilledPipeline(
            distilled_checkpoint_path=args.checkpoint,
            gemma_root=args.gemma_root,
            spatial_upsampler_path=args.spatial_upscaler,
            loras=loras,
            device=device,
            quantization=None,
            compilation_config=None,
            offload_mode=OffloadMode.NONE,
            layerwise_devices=layerwise_devices,
            text_encoder_dtype=text_encoder_dtype,
            embeddings_processor_device=embeddings_processor_device,
            embeddings_processor_dtype=embeddings_processor_dtype,
        )

    tiling_config = None if args.no_tiling else TilingConfig.default()
    _log_generation_profile(args, tiling_enabled=tiling_config is not None)
    video_chunks_number = get_video_chunks_number(args.frames, tiling_config) if tiling_config is not None else 1
    try:
        images = _parse_images(args.image)
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    base_output_path = Path(args.output)

    for run_idx in range(args.repeat):
        start = time.perf_counter()
        seed = args.seed + run_idx
        if args.pipeline in ("one-stage", "two-stage", "two-stage-hq"):
            video_guider_params = _guider_params(
                args.video_cfg,
                args.video_stg,
                args.video_rescale,
                args.a2v,
                args.video_stg_block,
            )
            audio_guider_params = _guider_params(
                args.audio_cfg,
                args.audio_stg,
                args.audio_rescale,
                args.v2a,
                args.audio_stg_block,
            )
            video, audio = pipeline(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                seed=seed,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                frame_rate=args.fps,
                num_inference_steps=args.steps,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=args.enhance_prompt,
                max_batch_size=args.max_batch_size,
            )
        else:
            video, audio = pipeline(
                prompt=args.prompt,
                seed=seed,
                height=args.height,
                width=args.width,
                num_frames=args.frames,
                frame_rate=args.fps,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=args.enhance_prompt,
            )

        if not (args.tensor_parallel and not is_rank0()):
            output_path = base_output_path
            if args.repeat > 1:
                output_name = f"{base_output_path.stem}_{run_idx:03d}{base_output_path.suffix}"
                output_path = base_output_path.with_name(output_name)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            encode_video(
                video=video,
                fps=args.fps,
                audio=audio,
                output_path=str(output_path),
                video_chunks_number=video_chunks_number,
            )
            logging.info(
                "wrote %s in %.2fs (run %d/%d, seed=%d)",
                output_path,
                time.perf_counter() - start,
                run_idx + 1,
                args.repeat,
                seed,
            )

        if args.tensor_parallel:
            torch.distributed.barrier()

    if args.tensor_parallel:
        destroy_process_group()


if __name__ == "__main__":
    main()

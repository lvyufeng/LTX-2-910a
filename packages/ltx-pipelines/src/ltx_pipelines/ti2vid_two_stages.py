import logging
import os
from collections.abc import Iterator

import torch

from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.distributed.hccl import is_rank0
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.random import make_generator
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_2_stage_arg_parser,
    detect_checkpoint_path,
    parse_torch_dtype,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    STAGE_2_DISTILLED_SIGMAS,
    detect_params,
)
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    get_device,
    profile_section,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at half of the target resolution with CFG guidance (assuming
    full model is used), then Stage 2 upsamples by 2x and refines using a distilled
    LoRA for higher quality output. Supports optional image conditioning via the
    images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        layerwise_devices: list[torch.device] | None = None,
        text_encoder_device: torch.device | None = None,
        text_encoder_layerwise_devices: list[torch.device] | None = None,
        text_encoder_dtype: torch.dtype | None = None,
        embeddings_processor_device: torch.device | None = None,
        embeddings_processor_dtype: torch.dtype | None = None,
        video_decoder_device: torch.device | None = None,
        video_decoder_dtype: torch.dtype | None = None,
        audio_decoder_device: torch.device | None = None,
        audio_decoder_dtype: torch.dtype | None = None,
        tensor_parallel: bool = False,
        resident_models: bool = False,
        dtype: torch.dtype | None = None,
        random_draw_device: torch.device | None = None,
        random_draw_dtype: torch.dtype | None = None,
    ):
        self.device = device or get_device()
        self.dtype = dtype or torch.float16
        self.random_draw_device = random_draw_device
        self.random_draw_dtype = random_draw_dtype
        self._tensor_parallel = tensor_parallel
        self._scheduler = LTX2Scheduler()

        self.prompt_encoder = PromptEncoder(
            checkpoint_path,
            gemma_root,
            self.dtype,
            self.device,
            registry=registry,
            offload_mode=offload_mode,
            text_encoder_device=text_encoder_device,
            text_encoder_dtype=text_encoder_dtype,
            embeddings_processor_device=embeddings_processor_device,
            embeddings_processor_dtype=embeddings_processor_dtype,
            layerwise_devices=text_encoder_layerwise_devices,
            tensor_parallel=tensor_parallel,
            resident=resident_models,
        )
        self.image_conditioner = ImageConditioner(checkpoint_path, self.dtype, self.device, registry=registry, resident=resident_models)
        self.upsampler = VideoUpsampler(
            checkpoint_path,
            spatial_upsampler_path,
            self.dtype,
            self.device,
            registry=registry,
            resident=resident_models and not tensor_parallel,
            tensor_parallel=tensor_parallel,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path,
            video_decoder_dtype or self.dtype,
            video_decoder_device or self.device,
            registry=registry,
            resident=resident_models,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path,
            audio_decoder_dtype or self.dtype,
            audio_decoder_device or self.device,
            registry=registry,
            resident=resident_models,
        )

        self.stage_1 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            layerwise_devices=layerwise_devices,
            tensor_parallel=tensor_parallel,
            resident=resident_models and not tensor_parallel,
        )
        self.stage_2 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=(*tuple(loras), *distilled_lora),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            layerwise_devices=layerwise_devices,
            tensor_parallel=tensor_parallel,
            resident=resident_models and not tensor_parallel,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = make_generator(
            seed,
            self.device,
            draw_device=self.random_draw_device,
            draw_dtype=self.random_draw_dtype,
        )
        noiser = GaussianNoiser(generator=generator)

        with profile_section("prompt_encoder", self.device):
            ctx_p, ctx_n = self.prompt_encoder(
                [prompt, negative_prompt],
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
                enhance_prompt_seed=seed,
            )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Stage 1: Generate video at half resolution with CFG guidance.
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        if images and (not self._tensor_parallel or is_rank0()):
            with profile_section("image_conditioner.stage1", self.device):
                stage_1_conditionings = self.image_conditioner(
                    lambda enc: combined_image_conditionings(
                        images=images,
                        height=stage_1_output_shape.height,
                        width=stage_1_output_shape.width,
                        video_encoder=enc,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
        else:
            stage_1_conditionings = []

        sigmas = (
            stage_1_sigmas if stage_1_sigmas is not None else self._scheduler.execute(steps=num_inference_steps)
        ).to(dtype=torch.float32, device=self.device)

        with profile_section("diffusion_stage.stage1", self.device):
            video_state, audio_state = self.stage_1(
                denoiser=FactoryGuidedDenoiser(
                    v_context=v_context_p,
                    a_context=a_context_p,
                    video_guider_factory=create_multimodal_guider_factory(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider_factory=create_multimodal_guider_factory(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                ),
                sigmas=sigmas,
                noiser=noiser,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
                audio=ModalitySpec(context=a_context_p),
                max_batch_size=max_batch_size,
            )

        if os.getenv("LTX2_DECODE_STAGE1", "").lower() in {"1", "true", "yes", "on"}:
            logging.warning("LTX2_DECODE_STAGE1 is set; decoding stage 1 half-res latent before upsampler for diagnostics")
            if self._tensor_parallel and not is_rank0():
                return iter(()), None
            decoded_video = self.video_decoder(video_state.latent[:1], tiling_config, generator)
            decoded_audio = self.audio_decoder(audio_state.latent) if audio_state is not None else None
            return decoded_video, decoded_audio

        # Stage 2: Upsample and refine the video at higher resolution with distilled LoRA.
        if self._tensor_parallel:
            with profile_section("video_upsampler", self.device):
                # After tensor-parallel stage 1 every rank already owns the same
                # full video latent. Do not force an extra rank0 broadcast here;
                # HCCL can time out creating a second communicator after the
                # diffusion all-reduces.
                upscaled_video_latent = self.upsampler.distributed_tensor_parallel(video_state.latent)
        else:
            with profile_section("video_upsampler", self.device):
                upscaled_video_latent = self.upsampler(video_state.latent[:1])
        video_state = None
        cleanup_memory()

        if os.getenv("LTX2_SKIP_STAGE2", "").lower() in {"1", "true", "yes", "on"}:
            logging.warning("LTX2_SKIP_STAGE2 is set; decoding upsampled latent before stage 2 for diagnostics")
            if self._tensor_parallel and not is_rank0():
                return iter(()), None
            decoded_video = self.video_decoder(upscaled_video_latent, tiling_config, generator)
            decoded_audio = self.audio_decoder(audio_state.latent) if audio_state is not None else None
            return decoded_video, decoded_audio

        stage_2_noise_scale = float(stage_2_sigmas[0].detach().cpu())
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        if images and (not self._tensor_parallel or is_rank0()):
            with profile_section("image_conditioner.stage2", self.device):
                stage_2_conditionings = self.image_conditioner(
                    lambda enc: combined_image_conditionings(
                        images=images,
                        height=height,
                        width=width,
                        video_encoder=enc,
                        dtype=self.dtype,
                        device=self.device,
                    )
                )
        else:
            stage_2_conditionings = []

        with profile_section("diffusion_stage.stage2", self.device):
            video_state, audio_state = self.stage_2(
                denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
                sigmas=stage_2_sigmas,
                noiser=noiser,
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(
                    context=v_context_p,
                    conditionings=stage_2_conditionings,
                    noise_scale=stage_2_noise_scale,
                    initial_latent=upscaled_video_latent,
                ),
                audio=ModalitySpec(
                    context=a_context_p,
                    noise_scale=stage_2_noise_scale,
                    initial_latent=audio_state.latent if audio_state is not None else None,
                ),
            )

        if self._tensor_parallel and not is_rank0():
            return iter(()), None

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        with profile_section("audio_decode", self.device):
            decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = default_2_stage_arg_parser(params=params, reference_runtime_args=True)
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else None
    dtype = parse_torch_dtype(args.dtype)
    text_encoder_dtype = parse_torch_dtype(args.text_encoder_dtype)
    random_draw_device = torch.device(args.random_draw_device) if args.random_draw_device else None
    random_draw_dtype = parse_torch_dtype(args.random_draw_dtype)
    pipeline = TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        device=device,
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        dtype=dtype,
        text_encoder_dtype=text_encoder_dtype,
        random_draw_device=random_draw_device,
        random_draw_dtype=random_draw_dtype,
    )
    logging.info("Using device: %s", pipeline.device)
    logging.info("Using dtype: %s", pipeline.dtype)
    if text_encoder_dtype is not None:
        logging.info("Using text encoder dtype: %s", text_encoder_dtype)
    if random_draw_device is not None or random_draw_dtype is not None:
        logging.info("Using random draw device: %s", random_draw_device or pipeline.device)
        logging.info("Using random draw dtype: %s", random_draw_dtype or pipeline.dtype)
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        tiling_config=tiling_config,
        max_batch_size=args.max_batch_size,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()

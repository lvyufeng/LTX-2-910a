import logging
from collections.abc import Iterator

import torch

from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.distributed.hccl import broadcast_tensor, is_rank0
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.debug import dump_tensor
from ltx_core.random import make_generator
from ltx_core.types import Audio
from ltx_pipelines.utils import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
)
from ltx_pipelines.utils.helpers import profile_section
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_1_stage_arg_parser,
    detect_checkpoint_path,
    parse_torch_dtype,
)
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode


class TI2VidOneStagePipeline:
    """
    Single-stage text/image-to-video generation pipeline.
    Generates video at the target resolution in a single diffusion pass with
    classifier-free guidance (CFG). Supports optional image conditioning via
    the images parameter.
    Assumes full non distilled model is provided in the checkpoint_path.
    """

    def __init__(
        self,
        checkpoint_path: str,
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
        tensor_parallel: bool = False,
        resident_models: bool = False,
        dtype: torch.dtype | None = None,
        random_draw_device: torch.device | None = None,
        random_draw_dtype: torch.dtype | None = None,
    ):
        self.dtype = dtype or torch.float16
        self.device = device or get_device()
        self.random_draw_device = random_draw_device
        self.random_draw_dtype = random_draw_dtype
        self._tensor_parallel = tensor_parallel
        self._scheduler = LTX2Scheduler()
        self.prompt_encoder = PromptEncoder(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            dtype=self.dtype,
            device=self.device,
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
        self.image_conditioner = ImageConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            resident=resident_models,
        )
        self.stage = DiffusionStage(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            compilation_config=compilation_config,
            offload_mode=offload_mode,
            layerwise_devices=layerwise_devices,
            tensor_parallel=tensor_parallel,
            resident=resident_models,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path=checkpoint_path,
            dtype=video_decoder_dtype or self.dtype,
            device=video_decoder_device or self.device,
            registry=registry,
            resident=resident_models,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
            resident=resident_models,
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
        enhance_prompt: bool = False,
        tiling_config: TilingConfig | None = None,
        max_batch_size: int = 1,
        sigmas: torch.Tensor | None = None,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=False)

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
        dump_tensor("prompt.video_context_positive", v_context_p)
        dump_tensor("prompt.audio_context_positive", a_context_p)
        dump_tensor("prompt.video_context_negative", v_context_n)
        dump_tensor("prompt.audio_context_negative", a_context_n)

        if images and (not self._tensor_parallel or is_rank0()):
            with profile_section("image_conditioner", self.device):
                stage_1_conditionings = self.image_conditioner(
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
            stage_1_conditionings = []

        sigmas = (sigmas if sigmas is not None else self._scheduler.execute(steps=num_inference_steps)).to(
            dtype=torch.float32, device=self.device
        )

        video_guider_factory = create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=v_context_n,
        )
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=a_context_n,
        )

        with profile_section("diffusion_stage", self.device):
            video_state, audio_state = self.stage(
                denoiser=FactoryGuidedDenoiser(
                    v_context=v_context_p,
                    a_context=a_context_p,
                    video_guider_factory=video_guider_factory,
                    audio_guider_factory=audio_guider_factory,
                ),
                sigmas=sigmas,
                noiser=noiser,
                width=width,
                height=height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(
                    context=v_context_p,
                    conditionings=stage_1_conditionings,
                ),
                audio=ModalitySpec(
                    context=a_context_p,
                ),
                max_batch_size=max_batch_size,
            )

        if self._tensor_parallel and not is_rank0():
            return iter(()), None

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator=generator)
        with profile_section("audio_decode", self.device):
            decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = default_1_stage_arg_parser(params=params, reference_runtime_args=True)
    args = parser.parse_args()
    device = torch.device(args.device) if args.device else None
    dtype = parse_torch_dtype(args.dtype)
    text_encoder_dtype = parse_torch_dtype(args.text_encoder_dtype)
    embeddings_processor_device = (
        torch.device(args.embeddings_processor_device) if args.embeddings_processor_device else None
    )
    embeddings_processor_dtype = parse_torch_dtype(args.embeddings_processor_dtype)
    random_draw_device = torch.device(args.random_draw_device) if args.random_draw_device else None
    random_draw_dtype = parse_torch_dtype(args.random_draw_dtype)
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        device=device,
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
        dtype=dtype,
        text_encoder_dtype=text_encoder_dtype,
        embeddings_processor_device=embeddings_processor_device,
        embeddings_processor_dtype=embeddings_processor_dtype,
        random_draw_device=random_draw_device,
        random_draw_dtype=random_draw_dtype,
    )
    logging.info("Using device: %s", pipeline.device)
    logging.info("Using dtype: %s", pipeline.dtype)
    if text_encoder_dtype is not None:
        logging.info("Using text encoder dtype: %s", text_encoder_dtype)
    if embeddings_processor_device is not None or embeddings_processor_dtype is not None:
        logging.info("Using embeddings processor device: %s", embeddings_processor_device or pipeline.device)
        logging.info("Using embeddings processor dtype: %s", embeddings_processor_dtype or pipeline.dtype)
    if random_draw_device is not None or random_draw_dtype is not None:
        logging.info("Using random draw device: %s", random_draw_device or pipeline.device)
        logging.info("Using random draw dtype: %s", random_draw_dtype or pipeline.dtype)
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
        max_batch_size=args.max_batch_size,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=1,
    )


if __name__ == "__main__":
    main()

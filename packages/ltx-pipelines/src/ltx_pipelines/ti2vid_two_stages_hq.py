import logging
import sys
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass

import torch

from ltx_core.components.diffusion_steps import Res2sDiffusionStep
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.distributed.hccl import (
    HCCLGroup,
    broadcast_tensor_from_global_rank,
    create_hccl_group,
    is_rank0,
    rank,
)
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.transformer.compiling import CompilationConfig
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput, hq_2_stage_arg_parser
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    LTX_2_3_HQ_PARAMS,
    STAGE_2_DISTILLED_SIGMAS,
)
from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    get_device,
    profile_section,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.samplers import res2s_audio_video_denoising_loop
from ltx_pipelines.utils.types import ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TwoStageTPPlacement:
    """Rank placement for split-stage two-stage-HQ tensor parallelism."""

    stage1_ranks: tuple[int, ...]
    stage2_ranks: tuple[int, ...]
    output_rank: int | None = None


@dataclass(frozen=True)
class _TwoStageTPGroups:
    stage1: HCCLGroup
    stage2: HCCLGroup
    output_rank: int


class TI2VidTwoStagesHQPipeline:
    @staticmethod
    def _build_split_tp_groups(placement: TwoStageTPPlacement) -> _TwoStageTPGroups:
        stage1 = create_hccl_group("tp_stage1", placement.stage1_ranks)
        stage2 = create_hccl_group("tp_stage2", placement.stage2_ranks)
        output_rank = stage2.leader_global_rank if placement.output_rank is None else int(placement.output_rank)
        if output_rank not in stage2.ranks:
            raise ValueError("split-stage TP output rank must belong to the stage2 rank group")
        logger.info(
            "split-stage TP-HQ enabled: stage1 ranks=%s stage2 ranks=%s output_rank=%d",
            ",".join(str(r) for r in stage1.ranks),
            ",".join(str(r) for r in stage2.ranks),
            output_rank,
        )
        return _TwoStageTPGroups(stage1=stage1, stage2=stage2, output_rank=output_rank)

    def _is_stage1_rank(self) -> bool:
        return self._stage1_group is None or self._stage1_group.is_member

    def _is_stage2_rank(self) -> bool:
        return self._stage2_group is None or self._stage2_group.is_member

    def _is_output_rank(self) -> bool:
        return rank() == self._output_rank if self._split_stage_tp else is_rank0()

    def _handoff_tensor(self, value: torch.Tensor | None, *, source_global_rank: int) -> torch.Tensor | None:
        return broadcast_tensor_from_global_rank(value, source_global_rank=source_global_rank, device=self.device)

    """
    Two-stage text/image-to-video generation pipeline using the res_2s sampler.
    Same structure as :class:`TI2VidTwoStagesPipeline`: stage 1 generates video at
    half of the target resolution with CFG guidance (assuming  full model is used),
    then Stage 2 upsamples by 2x and refines using a distilled LoRA for higher
    quality output.
    Uses the res_2s second-order sampler instead of Euler, allowing fewer
    steps for comparable quality. Supports optional image conditioning via
    the images parameter.
    """

    def __init__(  # noqa: PLR0913
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        distilled_lora_strength_stage_1: float,
        distilled_lora_strength_stage_2: float,
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...],
        device: torch.device | None = None,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        layerwise_devices: list[torch.device] | None = None,
        text_encoder_device: torch.device | None = None,
        text_encoder_dtype: torch.dtype | None = None,
        embeddings_processor_device: torch.device | None = None,
        embeddings_processor_dtype: torch.dtype | None = None,
        video_decoder_device: torch.device | None = None,
        video_decoder_dtype: torch.dtype | None = None,
        audio_decoder_device: torch.device | None = None,
        audio_decoder_dtype: torch.dtype | None = None,
        text_encoder_layerwise_devices: list[torch.device] | None = None,
        tensor_parallel: bool = False,
        resident_models: bool = False,
        tensor_parallel_placement: TwoStageTPPlacement | None = None,
    ):
        self.device = device or get_device()
        self.dtype = torch.float16
        if tensor_parallel_placement is not None and not tensor_parallel:
            raise ValueError("split-stage TP placement requires tensor_parallel=True")
        self._tensor_parallel = tensor_parallel
        self._split_tp_groups = self._build_split_tp_groups(tensor_parallel_placement) if tensor_parallel_placement is not None else None
        self._stage1_group = self._split_tp_groups.stage1 if self._split_tp_groups is not None else None
        self._stage2_group = self._split_tp_groups.stage2 if self._split_tp_groups is not None else None
        self._split_stage_tp = self._split_tp_groups is not None
        self._output_rank = self._split_tp_groups.output_rank if self._split_tp_groups is not None else 0
        self._scheduler = LTX2Scheduler()
        stage1_active = not self._split_stage_tp or self._is_stage1_rank()
        stage2_active = not self._split_stage_tp or self._is_stage2_rank()
        # In split-stage mode, keep prompt encoding on the full 8-rank default
        # HCCL group instead of pinning Gemma to the stage1 4-rank group.  This
        # uses the whole machine's memory bandwidth/capacity and cuts Gemma's
        # per-card resident shard in half, leaving room for each stage transformer.
        prompt_active = True
        prompt_tensor_parallel = tensor_parallel
        prompt_tensor_parallel_group = None
        stage1_tensor_parallel = tensor_parallel and stage1_active
        stage2_tensor_parallel = tensor_parallel and stage2_active
        tp_resident = resident_models and (not tensor_parallel or self._split_stage_tp)

        distilled_lora_stage_1 = LoraPathStrengthAndSDOps(
            path=distilled_lora[0].path,
            strength=distilled_lora_strength_stage_1,
            sd_ops=distilled_lora[0].sd_ops,
        )
        distilled_lora_stage_2 = LoraPathStrengthAndSDOps(
            path=distilled_lora[0].path,
            strength=distilled_lora_strength_stage_2,
            sd_ops=distilled_lora[0].sd_ops,
        )

        self.prompt_encoder = (
            PromptEncoder(
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
                tensor_parallel=prompt_tensor_parallel,
                resident=resident_models and prompt_active,
                resident_text_encoder=resident_models and prompt_active,
                resident_embeddings_processor=resident_models and prompt_active,
                embeddings_processor_rank0_only=self._split_stage_tp,
                tensor_parallel_group=prompt_tensor_parallel_group,
            )
            if prompt_active
            else None
        )
        self.image_conditioner_stage1 = (
            ImageConditioner(checkpoint_path, self.dtype, self.device, registry=registry, resident=resident_models)
            if stage1_active
            else None
        )
        self.image_conditioner_stage2 = (
            ImageConditioner(checkpoint_path, self.dtype, self.device, registry=registry, resident=resident_models)
            if stage2_active
            else None
        )
        self.image_conditioner = self.image_conditioner_stage1 or self.image_conditioner_stage2
        self.upsampler = (
            VideoUpsampler(
                checkpoint_path,
                spatial_upsampler_path,
                self.dtype,
                self.device,
                registry=registry,
                resident=tp_resident and stage2_active,
                tensor_parallel=stage2_tensor_parallel,
                tensor_parallel_group=self._stage2_group,
            )
            if stage2_active
            else None
        )
        self.video_decoder = (
            VideoDecoder(
                checkpoint_path,
                video_decoder_dtype or self.dtype,
                video_decoder_device or self.device,
                registry=registry,
                resident=resident_models,
            )
            if stage2_active
            else None
        )
        self.audio_decoder = (
            AudioDecoder(
                checkpoint_path,
                audio_decoder_dtype or self.dtype,
                audio_decoder_device or self.device,
                registry=registry,
                resident=resident_models,
            )
            if stage2_active
            else None
        )

        self.stage_1 = (
            DiffusionStage(
                checkpoint_path,
                self.dtype,
                self.device,
                loras=(*loras, distilled_lora_stage_1),
                quantization=quantization,
                registry=registry,
                compilation_config=compilation_config,
                offload_mode=offload_mode,
                layerwise_devices=layerwise_devices,
                tensor_parallel=stage1_tensor_parallel,
                resident=tp_resident and stage1_active,
                tensor_parallel_group=self._stage1_group,
            )
            if stage1_active
            else None
        )
        self.stage_2 = (
            DiffusionStage(
                checkpoint_path,
                self.dtype,
                self.device,
                loras=(*loras, distilled_lora_stage_2),
                quantization=quantization,
                registry=registry,
                compilation_config=compilation_config,
                offload_mode=offload_mode,
                layerwise_devices=layerwise_devices,
                tensor_parallel=stage2_tensor_parallel,
                resident=tp_resident and stage2_active,
                tensor_parallel_group=self._stage2_group,
            )
            if stage2_active
            else None
        )

    def _call_split_stage(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None,
        enhance_prompt: bool,
        max_batch_size: int,
        stage_1_sigmas: torch.Tensor | None,
        stage_2_sigmas: torch.Tensor,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert self._stage1_group is not None
        assert self._stage2_group is not None
        stage1_source = self._stage1_group.leader_global_rank

        latent_generator = torch.Generator(device=self.device).manual_seed(seed)
        decode_generator = torch.Generator(device=self.device).manual_seed(seed + 2_000_000)
        noiser = GaussianNoiser(generator=latent_generator)
        dtype = torch.float16
        stepper = Res2sDiffusionStep()

        stage_stack = ExitStack()
        stage_stack.__enter__()
        try:
            stage1_transformer = None
            if self._is_stage1_rank():
                assert self.stage_1 is not None
                stage1_transformer = stage_stack.enter_context(self.stage_1.model_context(video_tools=None))
            stage2_transformer = None
            if self._is_stage2_rank():
                assert self.stage_2 is not None
                stage2_transformer = stage_stack.enter_context(self.stage_2.model_context(video_tools=None))

            # Build prompt encoder after the resident diffusion transformers are
            # already sharded.  This avoids the transient peak where an 8-way
            # resident Gemma shard and an in-progress stage transformer build
            # compete for the last few MiB on stage1 cards.
            v_context_p = a_context_p = v_context_n = a_context_n = None
            assert self.prompt_encoder is not None
            with profile_section("prompt_encoder", self.device):
                ctx_p, ctx_n = self.prompt_encoder(
                    [prompt, negative_prompt],
                    enhance_first_prompt=enhance_prompt,
                    enhance_prompt_image=images[0][0] if len(images) > 0 else None,
                    enhance_prompt_seed=seed,
                )
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

            stage_1_output_shape = VideoPixelShape(
                batch=1,
                frames=num_frames,
                width=width // 2,
                height=height // 2,
                fps=frame_rate,
            )
            video_state = None
            audio_state = None
            if self._is_stage1_rank():
                stage_1_conditionings = []
                if images and is_rank0(self._stage1_group):
                    assert self.image_conditioner_stage1 is not None
                    with profile_section("image_conditioner.stage1", self.device):
                        stage_1_conditionings = self.image_conditioner_stage1(
                            lambda enc: combined_image_conditionings(
                                images=images,
                                height=stage_1_output_shape.height,
                                width=stage_1_output_shape.width,
                                video_encoder=enc,
                                dtype=dtype,
                                device=self.device,
                            )
                        )

                if stage_1_sigmas is None:
                    empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_output_shape).to_torch_shape())
                    stage_1_sigmas = self._scheduler.execute(latent=empty_latent, steps=num_inference_steps)
                sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)
                assert self.stage_1 is not None
                assert stage1_transformer is not None
                with profile_section("diffusion_stage.stage1", self.device):
                    video_state, audio_state = self.stage_1.run(
                        stage1_transformer,
                        denoiser=GuidedDenoiser(
                            v_context=v_context_p,
                            a_context=a_context_p,
                            video_guider=MultiModalGuider(
                                params=video_guider_params,
                                negative_context=v_context_n,
                            ),
                            audio_guider=MultiModalGuider(
                                params=audio_guider_params,
                                negative_context=a_context_n,
                            ),
                        ),
                        sigmas=sigmas,
                        noiser=noiser,
                        stepper=stepper,
                        width=stage_1_output_shape.width,
                        height=stage_1_output_shape.height,
                        frames=num_frames,
                        fps=frame_rate,
                        video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
                        audio=ModalitySpec(context=a_context_p),
                        loop=res2s_audio_video_denoising_loop,
                        max_batch_size=max_batch_size,
                        loop_kwargs={"noise_seed": seed, "noise_seed_substep": seed + 10_000},
                    )

            stage1_video_latent = self._handoff_tensor(
                video_state.latent if video_state is not None else None,
                source_global_rank=stage1_source,
            )
            stage1_audio_latent = self._handoff_tensor(
                audio_state.latent if audio_state is not None else None,
                source_global_rank=stage1_source,
            )
            video_state = None
            audio_state = None
            cleanup_memory()

            if self._is_stage2_rank():
                assert self.upsampler is not None
                with profile_section("video_upsampler", self.device):
                    upscaled_video_latent = self.upsampler.distributed_tensor_parallel(stage1_video_latent)
            else:
                upscaled_video_latent = None
            stage1_video_latent = None
            cleanup_memory()

            stage_2_noise_scale = float(stage_2_sigmas[0].detach().cpu())
            stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
            if self._is_stage2_rank():
                stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
                stage_2_conditionings = []
                if images and is_rank0(self._stage2_group):
                    assert self.image_conditioner_stage2 is not None
                    with profile_section("image_conditioner.stage2", self.device):
                        stage_2_conditionings = self.image_conditioner_stage2(
                            lambda enc: combined_image_conditionings(
                                images=images,
                                height=stage_2_output_shape.height,
                                width=stage_2_output_shape.width,
                                video_encoder=enc,
                                dtype=dtype,
                                device=self.device,
                            )
                        )

                assert self.stage_2 is not None
                assert stage2_transformer is not None
                with profile_section("diffusion_stage.stage2", self.device):
                    video_state, audio_state = self.stage_2.run(
                        stage2_transformer,
                        denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
                        sigmas=stage_2_sigmas,
                        noiser=noiser,
                        stepper=stepper,
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
                            initial_latent=stage1_audio_latent,
                        ),
                        loop=res2s_audio_video_denoising_loop,
                        loop_kwargs={"noise_seed": seed + 1_000_000, "noise_seed_substep": seed + 1_010_000},
                    )

            video_latent = video_state.latent if video_state is not None else None
            audio_latent = audio_state.latent if audio_state is not None else None
            video_state = None
            audio_state = None
            upscaled_video_latent = None
            stage1_audio_latent = None
            v_context_p = a_context_p = v_context_n = a_context_n = None
            cleanup_memory()

            if not self._is_output_rank():
                return iter(()), None

            assert video_latent is not None
            assert audio_latent is not None
            assert self.video_decoder is not None
            assert self.audio_decoder is not None
            decoded_video = self.video_decoder(video_latent, tiling_config, decode_generator)
            with profile_section("audio_decode", self.device):
                decoded_audio = self.audio_decoder(audio_latent)
            return decoded_video, decoded_audio
        finally:
            stage_stack.__exit__(*sys.exc_info())

    @torch.inference_mode()
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
        video_guider_params: MultiModalGuiderParams,
        audio_guider_params: MultiModalGuiderParams,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        max_batch_size: int = 1,
        stage_1_sigmas: torch.Tensor | None = None,
        stage_2_sigmas: torch.Tensor = STAGE_2_DISTILLED_SIGMAS,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)
        if self._split_stage_tp:
            return self._call_split_stage(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=enhance_prompt,
                max_batch_size=max_batch_size,
                stage_1_sigmas=stage_1_sigmas,
                stage_2_sigmas=stage_2_sigmas,
            )

        latent_generator = torch.Generator(device=self.device).manual_seed(seed)
        decode_generator = torch.Generator(device=self.device).manual_seed(seed + 2_000_000)
        noiser = GaussianNoiser(generator=latent_generator)
        dtype = torch.float16

        with profile_section("prompt_encoder", self.device):
            ctx_p, ctx_n = self.prompt_encoder(
                [prompt, negative_prompt],
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
                enhance_prompt_seed=seed,
            )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Stage 1: Generate video at half resolution with CFG guidance using res2s sampler.
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
                        dtype=dtype,
                        device=self.device,
                    )
                )
        else:
            stage_1_conditionings = []

        stepper = Res2sDiffusionStep()

        if stage_1_sigmas is None:
            empty_latent = torch.empty(VideoLatentShape.from_pixel_shape(stage_1_output_shape).to_torch_shape())
            stage_1_sigmas = self._scheduler.execute(latent=empty_latent, steps=num_inference_steps)
        sigmas = stage_1_sigmas.to(dtype=torch.float32, device=self.device)

        with profile_section("diffusion_stage.stage1", self.device):
            video_state, audio_state = self.stage_1(
                denoiser=GuidedDenoiser(
                    v_context=v_context_p,
                    a_context=a_context_p,
                    video_guider=MultiModalGuider(
                        params=video_guider_params,
                        negative_context=v_context_n,
                    ),
                    audio_guider=MultiModalGuider(
                        params=audio_guider_params,
                        negative_context=a_context_n,
                    ),
                ),
                sigmas=sigmas,
                noiser=noiser,
                stepper=stepper,
                width=stage_1_output_shape.width,
                height=stage_1_output_shape.height,
                frames=num_frames,
                fps=frame_rate,
                video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
                audio=ModalitySpec(context=a_context_p),
                loop=res2s_audio_video_denoising_loop,
                max_batch_size=max_batch_size,
                loop_kwargs={"noise_seed": seed, "noise_seed_substep": seed + 10_000},
            )

        # Stage 2: Upsample and refine the video at higher resolution with distilled LoRA.
        if self._tensor_parallel:
            with profile_section("video_upsampler", self.device):
                # After tensor-parallel stage 1 every rank already owns the same
                # full video latent. Passing None on nonzero ranks forces an extra
                # object-broadcast communicator here, which can time out on HCCL
                # after the diffusion all-reduces. Feed the local latent on all
                # ranks and let the TP upsampler run collectively.
                upscaled_video_latent = self.upsampler.distributed_tensor_parallel(video_state.latent)
        else:
            with profile_section("video_upsampler", self.device):
                upscaled_video_latent = self.upsampler(video_state.latent[:1])
        video_state = None
        cleanup_memory()

        stage_2_noise_scale = float(stage_2_sigmas[0].detach().cpu())
        stage_2_sigmas = stage_2_sigmas.to(dtype=torch.float32, device=self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        if images and (not self._tensor_parallel or is_rank0()):
            with profile_section("image_conditioner.stage2", self.device):
                stage_2_conditionings = self.image_conditioner(
                    lambda enc: combined_image_conditionings(
                        images=images,
                        height=stage_2_output_shape.height,
                        width=stage_2_output_shape.width,
                        video_encoder=enc,
                        dtype=dtype,
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
                stepper=stepper,
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
                loop=res2s_audio_video_denoising_loop,
                loop_kwargs={"noise_seed": seed + 1_000_000, "noise_seed_substep": seed + 1_010_000},
            )

        if self._tensor_parallel and not is_rank0():
            return iter(()), None

        decoded_video = self.video_decoder(video_state.latent, tiling_config, decode_generator)
        with profile_section("audio_decode", self.device):
            decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = hq_2_stage_arg_parser(params=LTX_2_3_HQ_PARAMS)
    args = parser.parse_args()
    pipeline = TI2VidTwoStagesHQPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        distilled_lora_strength_stage_1=args.distilled_lora_strength_stage_1,
        distilled_lora_strength_stage_2=args.distilled_lora_strength_stage_2,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        compilation_config=args.compile,
        offload_mode=args.offload_mode,
    )
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

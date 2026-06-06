"""Pipeline blocks — each block owns its model lifecycle.
Blocks build a model on each ``__call__``, use it, then free GPU memory.
This eliminates manual ``del model; cleanup_memory()`` in pipelines and
removes the need for :class:`ModelLedger`.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import logging
import os
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import replace
from typing import Callable, TypeVar

import torch

from ltx_core.accelerator import is_npu_device, synchronize
from ltx_core.batch_split import BatchSplitAdapter
from ltx_core.block_streaming import DISK_CPU_SLOTS, StreamingModelBuilder
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import Noiser
from ltx_core.components.patchifiers import AudioPatchifier, VideoLatentPatchifier
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.debug import dump_tensor
from ltx_core.distributed.hccl import (
    HCCLGroup,
    broadcast_tensor,
    broadcast_tensor_like,
    is_distributed,
    is_rank0,
    rank,
    world_size,
)
from ltx_core.loader import SDOps
from ltx_core.model.transformer.ascend_sharding import build_gemma_layerwise_device_map_op, build_layerwise_device_map_op
from ltx_core.model.transformer.ascend_tensor_parallel import build_hccl_tensor_parallel_op
from ltx_core.loader.attention_ops import set_attention_module_op
from ltx_core.loader.fuse_loras import bf16_fuse_rule
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import BuilderProtocol, LoraPathStrengthAndSDOps, ModelBuilderProtocol
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    VocoderConfigurator,
)
from ltx_core.model.audio_vae import (
    decode_audio as vae_decode_audio,
)
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModel,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.transformer.attention import (
    AscendChunkedAttention,
    AttentionCallable,
    AttentionFunction,
)
from ltx_core.model.transformer.compiling import (
    CompilationConfig,
    build_compile_transformer_op,
    modify_sd_ops_for_compilation,
)
from ltx_core.model.upsampler import LatentUpsamplerConfigurator, upsample_video
from ltx_core.model.upsampler.ascend_tensor_parallel import build_hccl_upsampler_tensor_parallel_op
from ltx_core.model.video_vae import (
    MEMORY_EFFICIENT_DECODE,
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    TilingConfig,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.quantization import QuantizationPolicy, fp8_cast_fuse_rule
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoder,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.text_encoders.gemma.ascend_tensor_parallel import (
    build_hccl_embeddings_processor_tensor_parallel_op,
    build_hccl_gemma_tensor_parallel_op,
)
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor, EmbeddingsProcessorOutput
from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_core.random import GeneratorLike
from ltx_core.types import Audio, AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.utils import find_matching_file
from ltx_pipelines.utils.gpu_model import gpu_model
from ltx_pipelines.utils.helpers import (
    cleanup_memory,
    create_noised_state,
    generate_enhanced_prompt,
    profile_section,
    state_with_conditionings,
)
from ltx_pipelines.utils.samplers import euler_denoising_loop
from ltx_pipelines.utils.types import Denoiser, ModalitySpec, OffloadMode

logger = logging.getLogger(__name__)

_EXPERIMENTAL_PRECISION_ENV = "LTX2_ASCEND_EXPERIMENTAL_PRECISION"
_VIDEO_DECODER_AUTOCAST_ENV = "LTX2_ASCEND_VIDEO_DECODER_AUTOCAST"
_PROMPT_EMBEDDINGS_CACHE_ENV = "LTX2_PROMPT_EMBEDDINGS_CACHE"
_TRUTHY_ENV_VALUES = {"1", "true", "yes", "on"}
_FALSY_ENV_VALUES = {"0", "false", "no", "off"}
_LOGGED_EXPERIMENTAL_PRECISION_SCOPES: set[str] = set()

T = TypeVar("T")
_M = TypeVar("_M", bound=torch.nn.Module)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").lower() in _TRUTHY_ENV_VALUES


def _env_disabled(name: str) -> bool:
    return os.getenv(name, "").lower() in _FALSY_ENV_VALUES


def _prompt_embeddings_cache_enabled() -> bool:
    return not _env_disabled(_PROMPT_EMBEDDINGS_CACHE_ENV)


def _prompt_embeddings_cache_key(
    prompts: list[str],
    *,
    enhance_first_prompt: bool,
    enhance_prompt_image: str | None,
    enhance_prompt_seed: int,
    dtype: torch.dtype,
    text_encoder_dtype: torch.dtype,
    embeddings_processor_dtype: torch.dtype,
    device: torch.device,
    text_encoder_device: torch.device,
    embeddings_processor_device: torch.device,
) -> tuple[object, ...]:
    return (
        tuple(prompts),
        bool(enhance_first_prompt),
        enhance_prompt_image if enhance_first_prompt else None,
        int(enhance_prompt_seed) if enhance_first_prompt else None,
        str(dtype),
        str(text_encoder_dtype),
        str(embeddings_processor_dtype),
        str(device),
        str(text_encoder_device),
        str(embeddings_processor_device),
    )


def _clone_prompt_output(output: EmbeddingsProcessorOutput) -> EmbeddingsProcessorOutput:
    return EmbeddingsProcessorOutput(
        video_encoding=output.video_encoding.clone(),
        audio_encoding=None if output.audio_encoding is None else output.audio_encoding.clone(),
        attention_mask=output.attention_mask.clone(),
    )


def _clone_prompt_outputs(outputs: tuple[EmbeddingsProcessorOutput, ...]) -> list[EmbeddingsProcessorOutput]:
    return [_clone_prompt_output(output) for output in outputs]


def _experimental_precision_enabled(name: str) -> bool:
    return _env_enabled(_EXPERIMENTAL_PRECISION_ENV) and _env_enabled(name)


def _log_experimental_precision_once(scope: str, message: str) -> None:
    if scope in _LOGGED_EXPERIMENTAL_PRECISION_SCOPES:
        return
    _LOGGED_EXPERIMENTAL_PRECISION_SCOPES.add(scope)
    logger.warning("Experimental precision enabled: %s", message)


def _chain_quantization(
    sd_ops: SDOps,
    module_ops: tuple[ModuleOps, ...],
    quantization: QuantizationPolicy,
) -> tuple[SDOps, tuple[ModuleOps, ...]]:
    chained_sd_ops = sd_ops
    if quantization.sd_ops is not None:
        chained_sd_ops = SDOps(
            name=f"sd_ops_chain_{sd_ops.name}+{quantization.sd_ops.name}",
            mapping=(*sd_ops.mapping, *quantization.sd_ops.mapping),
        )
    return chained_sd_ops, (*module_ops, *quantization.module_ops)


def _gemma_output_dtype_op(dtype: torch.dtype) -> ModuleOps:
    """Set GemmaTextEncoder.encode() output hidden-state dtype after construction."""

    def set_output_dtype(module: torch.nn.Module) -> torch.nn.Module:
        if isinstance(module, GemmaTextEncoder):
            module._dtype = dtype
        return module

    return ModuleOps(
        name=f"GemmaOutputDtype[{dtype}]",
        matcher=lambda module: isinstance(module, GemmaTextEncoder),
        mutator=set_output_dtype,
    )


def _apply_compile_ops(
    sd_ops: SDOps,
    module_ops: tuple[ModuleOps, ...],
    loras: tuple[LoraPathStrengthAndSDOps, ...],
    number_of_layers: int,
    compilation_config: CompilationConfig,
) -> tuple[SDOps, tuple[ModuleOps, ...], tuple[LoraPathStrengthAndSDOps, ...]]:
    """Rewrite sd_ops/module_ops/LoRAs for compiled blocks (params land under ``_orig_mod``)."""
    sd_ops = modify_sd_ops_for_compilation(sd_ops, number_of_layers)
    compile_op = build_compile_transformer_op(compilation_config)
    module_ops = (*module_ops, compile_op)
    loras = tuple(
        LoraPathStrengthAndSDOps(
            lora.path,
            lora.strength,
            modify_sd_ops_for_compilation(lora.sd_ops, number_of_layers),
        )
        for lora in loras
    )
    return sd_ops, module_ops, loras


@contextmanager
def _streaming_model(
    builder: StreamingModelBuilder,
    offload_mode: OffloadMode,
    target_device: torch.device,
    dtype: torch.dtype,
) -> Iterator:
    """Build a streaming wrapper, yield it, then tear down and free memory."""
    cpu_slots_count = DISK_CPU_SLOTS if offload_mode == OffloadMode.DISK else None
    wrapped = builder.build(
        target_device=target_device,
        dtype=dtype,
        cpu_slots_count=cpu_slots_count,
    )
    try:
        yield wrapped
    finally:
        wrapped.teardown()
        wrapped.to("meta")
        cleanup_memory()


def _build_state(
    spec: ModalitySpec,
    tools: LatentTools,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
) -> LatentState:
    """Create a noised latent state from a modality spec and tools."""
    state = create_noised_state(
        tools=tools,
        conditionings=spec.conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=spec.noise_scale,
        initial_latent=spec.initial_latent,
    )
    if spec.frozen:
        state = replace(state, denoise_mask=torch.zeros_like(state.denoise_mask))
    return state


def _group_src(group: HCCLGroup | None) -> int:
    return group.leader_global_rank if group is not None else 0


def _broadcast_state(state: LatentState | None, device: torch.device, group: HCCLGroup | None = None) -> LatentState | None:
    src = _group_src(group)
    latent = broadcast_tensor(None if state is None else state.latent, device=device, src=src, group=group)
    if latent is None:
        return None
    return LatentState(
        latent=latent,
        denoise_mask=broadcast_tensor(state.denoise_mask if state is not None else None, device=device, src=src, group=group),
        positions=broadcast_tensor(state.positions if state is not None else None, device=device, src=src, group=group),
        clean_latent=broadcast_tensor(state.clean_latent if state is not None else None, device=device, src=src, group=group),
        attention_mask=broadcast_tensor(None if state is None else state.attention_mask, device=device, src=src, group=group),
    )


def _broadcast_bool(value: bool, device: torch.device, group: HCCLGroup | None = None) -> bool:
    src = _group_src(group)
    flag = torch.tensor([1 if value else 0], device=device, dtype=torch.int64) if is_rank0(group) else None
    flag = broadcast_tensor_like(flag, shape=(1,), dtype=torch.int64, device=device, src=src, group=group)
    return bool(flag.item())


def _sync_state_latent(state: LatentState, device: torch.device, group: HCCLGroup | None = None) -> LatentState:
    latent = broadcast_tensor_like(
        state.latent if is_rank0(group) else None,
        shape=state.latent.shape,
        dtype=state.latent.dtype,
        device=device,
        src=_group_src(group),
        group=group,
    )
    return replace(state, latent=latent)


def _build_state_with_synchronized_noise(
    spec: ModalitySpec,
    tools: LatentTools,
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    group: HCCLGroup | None = None,
) -> LatentState:
    """Build state on all ranks but draw random noise only on the TP group leader."""
    state = tools.create_initial_state(device, dtype, spec.initial_latent)
    state = state_with_conditionings(state, spec.conditionings, tools)
    if is_rank0(group):
        state = noiser(state, spec.noise_scale)
    state = _sync_state_latent(state, device, group)
    if spec.frozen:
        state = replace(state, denoise_mask=torch.zeros_like(state.denoise_mask))
    return state


def _cleanup_iter(it: Iterator[torch.Tensor], model: torch.nn.Module) -> Iterator[torch.Tensor]:
    """Wrap an iterator to clean up *model* memory once it is exhausted or abandoned."""
    with gpu_model(model):
        yield from it


@contextmanager
def _profiled_gpu_model(model: _M, *, teardown_profile: str | None = None) -> Iterator[_M]:
    """Like ``gpu_model()``, with optional teardown profiling under ``LTX2_ASCEND_PROFILE``.

    This preserves the existing model lifecycle exactly: yield the already-built
    model, then synchronize, move parameters/buffers to meta, and run
    ``cleanup_memory()``. The optional profile section only measures that teardown
    cost; it does not change model math or residency.
    """
    try:
        yield model
    finally:
        if teardown_profile is None:
            synchronize()
            model.to("meta")
            cleanup_memory()
        else:
            with profile_section(teardown_profile):
                synchronize()
                model.to("meta")
                cleanup_memory()


def _maybe_autocast_iter(
    it: Iterator[torch.Tensor],
    device: torch.device,
    *,
    env_name: str,
    scope: str,
) -> Iterator[torch.Tensor]:
    """Optionally keep an autocast scope active while a lazy iterator is consumed."""
    if not _experimental_precision_enabled(env_name) or device.type != "npu":
        yield from it
        return
    _log_experimental_precision_once(scope, f"{scope} uses NPU float16 autocast")
    with torch.autocast(device_type="npu", dtype=torch.float16):
        yield from it


def _dump_first_chunk(it: Iterator[torch.Tensor]) -> Iterator[torch.Tensor]:
    """Pass-through iterator that dumps the first decoded chunk for debug comparison."""
    for index, chunk in enumerate(it):
        if index == 0:
            dump_tensor("video_decoder.first_chunk", chunk)
        yield chunk


# ---------------------------------------------------------------------------
# DiffusionStage
# ---------------------------------------------------------------------------


class DiffusionStage:
    """Owns transformer lifecycle. Builds on each call, frees on exit.
    Replaces the manual ``model_ledger.transformer()`` / ``del transformer``
    pattern in every pipeline.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        compilation_config: CompilationConfig | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        transformer_builder: ModelBuilderProtocol[LTXModel] | None = None,
        layerwise_devices: list[torch.device] | None = None,
        tensor_parallel: bool = False,
        resident: bool = False,
        tensor_parallel_group: HCCLGroup | None = None,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._resident = resident
        self._resident_transformer: X0Model | None = None
        self._quantization = quantization
        self._compilation_config = compilation_config
        if offload_mode != OffloadMode.NONE and is_npu_device(device):
            raise ValueError("text encoder offload is not supported on Ascend NPU yet")
        self._offload_mode = offload_mode
        configurator = (
            quantization.model_configurator
            if quantization is not None and quantization.model_configurator is not None
            else LTXModelConfigurator
        )
        if transformer_builder is not None:
            self._transformer_builder = transformer_builder
        else:
            self._transformer_builder = Builder(
                model_path=checkpoint_path,
                model_class_configurator=configurator,
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
                loras=tuple(loras),
                registry=registry or DummyRegistry(),
            )
        self._tensor_parallel = tensor_parallel
        self._tensor_parallel_group = tensor_parallel_group
        if tensor_parallel:
            if not is_npu_device(device):
                raise ValueError("HCCL tensor parallelism requires an Ascend NPU device")
            if not is_distributed() or world_size(tensor_parallel_group) <= 1:
                raise ValueError("HCCL tensor parallelism requires torchrun with world_size > 1")
            self._transformer_builder = self._transformer_builder.with_module_ops(
                (
                    *self._transformer_builder.module_ops,
                    build_hccl_tensor_parallel_op(
                        rank=rank(tensor_parallel_group),
                        world_size=world_size(tensor_parallel_group),
                        device=device,
                        process_group=tensor_parallel_group.process_group if tensor_parallel_group is not None else None,
                        label=tensor_parallel_group.name if tensor_parallel_group is not None else None,
                    ),
                )
            )
        if layerwise_devices is not None and len(layerwise_devices) > 1:
            if tensor_parallel:
                raise ValueError("layerwise sharding and tensor parallelism are mutually exclusive")
            self._transformer_builder = self._transformer_builder.with_module_ops(
                (*self._transformer_builder.module_ops, build_layerwise_device_map_op(layerwise_devices))
            )

        if offload_mode != OffloadMode.NONE and is_npu_device(self._device):
            raise ValueError("block streaming/offload is not supported on Ascend NPU yet")
        if offload_mode != OffloadMode.NONE:
            if compilation_config is not None:
                raise ValueError("torch.compile is not supported with layer streaming")
            # WeightsProvider currently only supports plain bf16 + fp8_cast LoRA fusion
            # (no companion-key emission). Quantization policies that emit
            # companion keys (e.g. ``.weight_scale``) cannot be streamed yet.
            if quantization is not None and quantization.fuse_rule is not fp8_cast_fuse_rule:
                raise ValueError(
                    "Block streaming is not supported with this quantization policy "
                    "(only bf16 and fp8_cast are currently supported)."
                )
            streaming_sd_ops: SDOps = LTXV_MODEL_COMFY_RENAMING_MAP
            streaming_module_ops: tuple[ModuleOps, ...] = ()
            if quantization is not None:
                streaming_sd_ops, streaming_module_ops = _chain_quantization(
                    streaming_sd_ops, streaming_module_ops, quantization
                )
            self._streaming_builder = StreamingModelBuilder(
                model_class_configurator=configurator,
                model_path=checkpoint_path,
                model_sd_ops=streaming_sd_ops,
                module_ops=streaming_module_ops,
                loras=tuple(loras),
                registry=registry or DummyRegistry(),
                fuse_rule=quantization.fuse_rule if quantization is not None else bf16_fuse_rule,
                blocks_attr="transformer_blocks",
                blocks_prefix="transformer_blocks",
            )

    def with_attention(self, attention: AttentionFunction | AttentionCallable | None) -> "DiffusionStage":
        """Return a new ``DiffusionStage`` that pins the transformer build to ``attention``.
        Functional: never mutates ``self``. The returned stage shares all other
        configuration with the original; only the underlying builders' ``module_ops``
        gain a ``set_attention_module_op(attention)`` entry so subsequent transformer
        builds use that kernel. ``attention=None`` is a no-op (returns ``self``).
        """
        if attention is None:
            return self
        op = set_attention_module_op(attention)
        new = copy.copy(self)
        new._transformer_builder = self._transformer_builder.with_module_ops(
            (*self._transformer_builder.module_ops, op),
        )
        if self._offload_mode != OffloadMode.NONE:
            new._streaming_builder = dataclasses.replace(
                self._streaming_builder,
                module_ops=(*self._streaming_builder.module_ops, op),
            )
        return new

    def _build_transformer(self, *, device: torch.device | None = None, **kwargs: object) -> X0Model:
        target = device or self._device
        sd_ops = self._transformer_builder.model_sd_ops
        module_ops = self._transformer_builder.module_ops
        loras = self._transformer_builder.loras
        if self._compilation_config is not None:
            number_of_layers = self._transformer_builder.model_config()["transformer"]["num_layers"]
            sd_ops, module_ops, loras = _apply_compile_ops(
                sd_ops, module_ops, loras, number_of_layers, self._compilation_config
            )
        if self._quantization is not None:
            sd_ops, module_ops = _chain_quantization(sd_ops, module_ops, self._quantization)

        builder = self._transformer_builder.with_module_ops(module_ops).with_sd_ops(sd_ops).with_loras(loras)
        if self._quantization is not None:
            builder = builder.with_fuse_rule(self._quantization.fuse_rule)
        model = builder.build(device=target, dtype=self._dtype, **kwargs)
        wrapped = X0Model(model)
        if getattr(model, "tensor_parallel", False) or getattr(model, "block_device_map", None):
            return wrapped.eval()
        return wrapped.to(target).eval()

    @contextmanager
    def _streaming_transformer_ctx(self) -> Iterator[X0Model]:
        with _streaming_model(
            self._streaming_builder, self._offload_mode, self._device, self._dtype
        ) as streaming_wrapper:
            yield X0Model(streaming_wrapper).eval()

    def _transformer_ctx(self, **kwargs: object) -> AbstractContextManager:
        if self._offload_mode != OffloadMode.NONE:
            return self._streaming_transformer_ctx()
        if self._resident:
            if self._resident_transformer is None:
                with profile_section("diffusion_stage.transformer_build", self._device):
                    self._resident_transformer = self._build_transformer(**kwargs)
            return nullcontext(self._resident_transformer)
        with profile_section("diffusion_stage.transformer_build", self._device):
            transformer = self._build_transformer(**kwargs)
        return _profiled_gpu_model(transformer, teardown_profile="diffusion_stage.transformer_teardown")

    def model_context(self, **kwargs: object) -> AbstractContextManager:
        """Build the transformer, yield it, then free its memory on exit.
        Keyword arguments are forwarded to the underlying builder (e.g.
        ``video_tools`` required by ``TiledDataParallelBuilder``).
        """
        return self._transformer_ctx(**kwargs)

    def run(  # noqa: PLR0913
        self,
        transformer: object,
        denoiser: Denoiser,
        sigmas: torch.Tensor,
        noiser: Noiser,
        width: int,
        height: int,
        frames: int,
        fps: float,
        video: ModalitySpec | None = None,
        audio: ModalitySpec | None = None,
        stepper: DiffusionStepProtocol | None = None,
        loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
        max_batch_size: int = 1,
        loop_kwargs: dict[str, object] | None = None,
    ) -> tuple[LatentState | None, LatentState | None]:
        """Run denoising with a pre-built transformer.
        Same semantics as ``__call__`` but accepts a pre-built transformer so
        the model can be shared across multiple calls (e.g. tiled inference
        inside a single ``model_context()`` block). Audio supports
        ``ModalitySpec(frozen=True)`` to keep the latent unchanged throughout
        denoising while still providing cross-modal context to the transformer.
        Returns ``(video_state | None, audio_state | None)`` with cleared
        conditionings and unpatchified latents for present modalities.
        """
        if video is None and audio is None:
            raise ValueError("At least one of `video` or `audio` must be provided")

        if loop is None:
            loop = euler_denoising_loop
        if stepper is None:
            stepper = EulerDiffusionStep()

        pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)

        video_state: LatentState | None = None
        video_tools: LatentTools | None = None
        if video is not None:
            with profile_section("diffusion_stage.build_video_state", self._device):
                v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
                video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)
                has_conditionings = bool(video.conditionings)
                if self._tensor_parallel:
                    has_conditionings = _broadcast_bool(has_conditionings, self._device, self._tensor_parallel_group)
                if self._tensor_parallel and not has_conditionings:
                    video_state = _build_state_with_synchronized_noise(
                        video,
                        video_tools,
                        noiser,
                        self._dtype,
                        self._device,
                        self._tensor_parallel_group,
                    )
                else:
                    if not self._tensor_parallel or is_rank0(self._tensor_parallel_group):
                        video_state = _build_state(video, video_tools, noiser, self._dtype, self._device)
                    if self._tensor_parallel:
                        video_state = _broadcast_state(video_state, self._device, self._tensor_parallel_group)

        audio_state: LatentState | None = None
        audio_tools: LatentTools | None = None
        if audio is not None:
            with profile_section("diffusion_stage.build_audio_state", self._device):
                a_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
                audio_tools = AudioLatentTools(AudioPatchifier(patch_size=1), a_shape)
                has_conditionings = bool(audio.conditionings)
                if self._tensor_parallel:
                    has_conditionings = _broadcast_bool(has_conditionings, self._device, self._tensor_parallel_group)
                if self._tensor_parallel and not has_conditionings:
                    audio_state = _build_state_with_synchronized_noise(
                        audio,
                        audio_tools,
                        noiser,
                        self._dtype,
                        self._device,
                        self._tensor_parallel_group,
                    )
                else:
                    if not self._tensor_parallel or is_rank0(self._tensor_parallel_group):
                        audio_state = _build_state(audio, audio_tools, noiser, self._dtype, self._device)
                    if self._tensor_parallel:
                        audio_state = _broadcast_state(audio_state, self._device, self._tensor_parallel_group)

        if video_state is not None:
            dump_tensor("diffusion_stage.initial_video_latent", video_state.latent)
        if audio_state is not None:
            dump_tensor("diffusion_stage.initial_audio_latent", audio_state.latent)

        wrapped = BatchSplitAdapter(transformer, max_batch_size=max_batch_size)  # type: ignore[arg-type]
        effective_loop_kwargs = dict(loop_kwargs or {})
        if self._tensor_parallel and "tp_group" in inspect.signature(loop).parameters:
            effective_loop_kwargs.setdefault("tp_group", self._tensor_parallel_group)
        with profile_section("diffusion_stage.loop", self._device):
            video_state, audio_state = loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                transformer=wrapped,
                denoiser=denoiser,
                **effective_loop_kwargs,
            )

        if video_state is not None:
            dump_tensor("diffusion_stage.denoised_video_latent", video_state.latent)
        if audio_state is not None:
            dump_tensor("diffusion_stage.denoised_audio_latent", audio_state.latent)

        with profile_section("diffusion_stage.postprocess_state", self._device):
            if video_state is not None and video_tools is not None:
                video_state = video_tools.clear_conditioning(video_state)
                video_state = video_tools.unpatchify(video_state)
            if audio_state is not None and audio_tools is not None:
                audio_state = audio_tools.clear_conditioning(audio_state)
                audio_state = audio_tools.unpatchify(audio_state)

        return video_state, audio_state

    def __call__(  # noqa: PLR0913
        self,
        denoiser: Denoiser,
        sigmas: torch.Tensor,
        noiser: Noiser,
        width: int,
        height: int,
        frames: int,
        fps: float,
        video: ModalitySpec | None = None,
        audio: ModalitySpec | None = None,
        stepper: DiffusionStepProtocol | None = None,
        loop: Callable[..., tuple[LatentState | None, LatentState | None]] | None = None,
        max_batch_size: int = 1,
        loop_kwargs: dict[str, object] | None = None,
    ) -> tuple[LatentState | None, LatentState | None]:
        """Build transformer -> run denoising loop -> free transformer.
        Returns ``(video_state | None, audio_state | None)`` with cleared
        conditionings and unpatchified latents for present modalities.
        """
        # Build video_tools up front so it can be forwarded to the transformer
        # context (required by TiledDataParallelBuilder in multi-GPU mode).
        # `run()` rebuilds its own tools internally; the duplication is cheap.
        video_tools: LatentTools | None = None
        if video is not None:
            pixel_shape = VideoPixelShape(batch=1, frames=frames, height=height, width=width, fps=fps)
            v_shape = VideoLatentShape.from_pixel_shape(pixel_shape)
            video_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), v_shape, fps)

        mode = "streaming" if self._offload_mode != OffloadMode.NONE else "standard"
        logger.info("Building transformer (%s) from %s", mode, self._checkpoint_path)
        with self._transformer_ctx(video_tools=video_tools) as transformer:
            logger.info(
                "Running denoising loop (%d steps, %dx%d %d frames @ %.1f fps)",
                len(sigmas) - 1,
                width,
                height,
                frames,
                fps,
            )
            return self.run(
                transformer,
                denoiser,
                sigmas,
                noiser,
                width,
                height,
                frames,
                fps,
                video,
                audio,
                stepper,
                loop,
                max_batch_size,
                loop_kwargs,
            )


# ---------------------------------------------------------------------------
# PromptEncoder
# ---------------------------------------------------------------------------


class PromptEncoder:
    """Owns text encoder + embeddings processor lifecycle.
    Loads Gemma, encodes prompts, frees Gemma, then loads the embeddings
    processor to produce final outputs.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        offload_mode: OffloadMode = OffloadMode.NONE,
        text_encoder_builder: BuilderProtocol | None = None,
        text_encoder_device: torch.device | None = None,
        text_encoder_dtype: torch.dtype | None = None,
        embeddings_processor_device: torch.device | None = None,
        embeddings_processor_dtype: torch.dtype | None = None,
        layerwise_devices: list[torch.device] | None = None,
        tensor_parallel: bool = False,
        resident: bool = False,
        resident_text_encoder: bool | None = None,
        resident_embeddings_processor: bool | None = None,
        embeddings_processor_rank0_only: bool = False,
        tensor_parallel_group: HCCLGroup | None = None,
    ) -> None:
        self._gemma_root = gemma_root
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._text_encoder_dtype = text_encoder_dtype or dtype
        self._embeddings_processor_dtype = embeddings_processor_dtype or dtype
        self._device = device
        self._text_encoder_device = text_encoder_device or device
        self._tensor_parallel = tensor_parallel
        self._tensor_parallel_group = tensor_parallel_group
        self._resident = resident
        self._resident_text_encoder_enabled = resident if resident_text_encoder is None else resident_text_encoder
        self._resident_embeddings_processor_enabled = (
            resident if resident_embeddings_processor is None else resident_embeddings_processor
        )
        self._embeddings_processor_rank0_only = embeddings_processor_rank0_only
        self._resident_text_encoder: torch.nn.Module | None = None
        self._resident_embeddings_processor: EmbeddingsProcessor | None = None
        self._prompt_embeddings_cache: dict[tuple[object, ...], tuple[EmbeddingsProcessorOutput, ...]] = {}
        self._tp_prompt_rank0_only = (
            tensor_parallel
            and os.getenv("LTX2_TP_PROMPT_RANK0_ONLY", "").lower() in {"1", "true", "yes", "on"}
        )
        if tensor_parallel:
            if not is_npu_device(device):
                raise ValueError("Gemma HCCL tensor parallelism requires an Ascend NPU device")
            if not is_distributed() or world_size(tensor_parallel_group) <= 1:
                raise ValueError("Gemma HCCL tensor parallelism requires torchrun with world_size > 1")
        if embeddings_processor_device is not None:
            self._embeddings_processor_device = embeddings_processor_device
        elif self._device.type == "npu" and self._text_encoder_device == self._device and not tensor_parallel:
            self._embeddings_processor_device = torch.device("npu", (self._device.index or 0) + 1)
        else:
            self._embeddings_processor_device = self._device
        self._offload_mode = offload_mode

        if text_encoder_builder is not None:
            if offload_mode != OffloadMode.NONE:
                raise ValueError(
                    "text_encoder_builder cannot be used with offload_mode != OffloadMode.NONE "
                    "because no streaming text encoder builder is available."
                )
            self._text_encoder_builder = text_encoder_builder
            self._streaming_text_encoder_builder = None
        else:
            module_ops = (*module_ops_from_gemma_root(gemma_root), _gemma_output_dtype_op(self._text_encoder_dtype))
            disable_tp_text_encoder = os.getenv("LTX2_DISABLE_TP_TEXT_ENCODER", "").lower() in {"1", "true", "yes", "on"}
            if tensor_parallel and not disable_tp_text_encoder:
                module_ops = (
                    *module_ops,
                    build_hccl_gemma_tensor_parallel_op(
                        rank=rank(tensor_parallel_group),
                        world_size=world_size(tensor_parallel_group),
                        device=self._text_encoder_device,
                        process_group=tensor_parallel_group.process_group if tensor_parallel_group is not None else None,
                        label=tensor_parallel_group.name if tensor_parallel_group is not None else None,
                    ),
                )
            elif layerwise_devices is not None and len(layerwise_devices) > 1:
                module_ops = (*module_ops, build_gemma_layerwise_device_map_op(layerwise_devices))
            model_folder = find_matching_file(gemma_root, "model*.safetensors").parent
            weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]
            self._text_encoder_builder = Builder(
                model_path=tuple(weight_paths),
                model_class_configurator=GemmaTextEncoderConfigurator,
                model_sd_ops=GEMMA_LLM_KEY_OPS,
                module_ops=(GEMMA_MODEL_OPS, *module_ops),
                registry=registry or DummyRegistry(),
            )
            self._streaming_text_encoder_builder = StreamingModelBuilder(
                model_path=tuple(weight_paths),
                model_class_configurator=GemmaTextEncoderConfigurator,
                model_sd_ops=GEMMA_LLM_KEY_OPS,
                module_ops=(GEMMA_MODEL_OPS, *module_ops),
                registry=registry or DummyRegistry(),
                blocks_attr="model.model.language_model.layers",
                blocks_prefix="model.model.language_model.layers",
            )
        embeddings_module_ops: tuple[ModuleOps, ...] = ()
        # Keep the prompt embeddings processor as a full fp32 replica on each NPU
        # by default.  The fp16 path has been verified to corrupt prompt context,
        # and the HCCL-sharded processor is a separate numerical surface from the
        # transformer TP path.  It is small enough to replicate; set
        # LTX2_TP_EMBEDDINGS_PROCESSOR=1 only for explicit experiments.
        if tensor_parallel and os.getenv("LTX2_TP_EMBEDDINGS_PROCESSOR", "").lower() in {"1", "true", "yes", "on"}:
            embeddings_module_ops = (
                build_hccl_embeddings_processor_tensor_parallel_op(
                    rank=rank(tensor_parallel_group),
                    world_size=world_size(tensor_parallel_group),
                    device=self._embeddings_processor_device,
                    process_group=tensor_parallel_group.process_group if tensor_parallel_group is not None else None,
                    label=tensor_parallel_group.name if tensor_parallel_group is not None else None,
                ),
            )
        elif is_npu_device(self._embeddings_processor_device):
            chunked = AscendChunkedAttention()
            embeddings_module_ops = (set_attention_module_op(chunked, chunked),)
        self._embeddings_processor_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
            module_ops=embeddings_module_ops,
            registry=registry or DummyRegistry(),
        )

    def _build_text_encoder(self) -> torch.nn.Module:
        """Build the Gemma text encoder (non-streaming path)."""
        with profile_section("prompt_encoder.text_encoder_build", self._text_encoder_device):
            text_encoder = self._text_encoder_builder.build(device=self._text_encoder_device, dtype=self._text_encoder_dtype).eval()
        if isinstance(text_encoder, GemmaTextEncoder):
            text_encoder._dtype = self._text_encoder_dtype
        return text_encoder

    def _build_embeddings_processor(self) -> EmbeddingsProcessor:
        """Build the embeddings processor on the configured processor device."""
        with profile_section("prompt_encoder.embeddings_processor_build", self._embeddings_processor_device):
            return self._embeddings_processor_builder.build(
                device=self._embeddings_processor_device,
                dtype=self._embeddings_processor_dtype,
            ).eval()

    def _text_encoder_ctx(self) -> AbstractContextManager:
        if self._offload_mode != OffloadMode.NONE:
            return _streaming_model(self._streaming_text_encoder_builder, self._offload_mode, self._text_encoder_device, self._text_encoder_dtype)
        if self._resident_text_encoder_enabled:
            if self._resident_text_encoder is None:
                self._resident_text_encoder = self._build_text_encoder()
            return nullcontext(self._resident_text_encoder)
        return gpu_model(self._build_text_encoder())

    def __call__(
        self,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
    ) -> list[EmbeddingsProcessorOutput]:
        """Encode *prompts* through Gemma -> embeddings processor, freeing each model after use."""
        cache_key = _prompt_embeddings_cache_key(
            prompts,
            enhance_first_prompt=enhance_first_prompt,
            enhance_prompt_image=enhance_prompt_image,
            enhance_prompt_seed=enhance_prompt_seed,
            dtype=self._dtype,
            text_encoder_dtype=self._text_encoder_dtype,
            embeddings_processor_dtype=self._embeddings_processor_dtype,
            device=self._device,
            text_encoder_device=self._text_encoder_device,
            embeddings_processor_device=self._embeddings_processor_device,
        )
        cache_enabled = _prompt_embeddings_cache_enabled()
        cached = self._prompt_embeddings_cache.get(cache_key) if cache_enabled else None
        if cached is not None:
            logger.info("Prompt embeddings cache hit")
            return _clone_prompt_outputs(cached)

        raw_outputs = None
        if self._tp_prompt_rank0_only and not is_rank0(self._tensor_parallel_group):
            logger.info("Skipping prompt text encoder on nonzero TP rank")
        else:
            logger.info("Building text encoder from %s", self._gemma_root)
            with self._text_encoder_ctx() as text_encoder:
                if enhance_first_prompt:
                    prompts = list(prompts)
                    prompts[0] = generate_enhanced_prompt(
                        text_encoder, prompts[0], enhance_prompt_image, seed=enhance_prompt_seed
                    )
                raw_outputs = []
                for prompt_index, prompt in enumerate(prompts):
                    with profile_section(f"prompt_encoder.text_encode.{prompt_index}", self._text_encoder_device):
                        raw_outputs.append(text_encoder.encode(prompt))
                    with profile_section(f"prompt_encoder.text_encode_cleanup.{prompt_index}", self._text_encoder_device):
                        cleanup_memory()
            logger.info("Text encoder done, building embeddings processor from %s", self._checkpoint_path)
            with profile_section("prompt_encoder.after_text_cleanup", self._text_encoder_device):
                cleanup_memory()

        def move_raw_output(raw_output: tuple[tuple[torch.Tensor, ...], torch.Tensor]) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
            hidden_states, attention_mask = raw_output
            hidden_states = tuple(
                t.to(device=self._embeddings_processor_device, dtype=self._embeddings_processor_dtype)
                for t in hidden_states
            )
            return hidden_states, attention_mask.to(device=self._embeddings_processor_device)

        if self._tp_prompt_rank0_only and not is_rank0(self._tensor_parallel_group):
            result = []
            for _ in prompts:
                video_encoding = broadcast_tensor(
                    None,
                    device=self._device,
                    src=_group_src(self._tensor_parallel_group),
                    group=self._tensor_parallel_group,
                )
                audio_encoding = broadcast_tensor(
                    None,
                    device=self._device,
                    src=_group_src(self._tensor_parallel_group),
                    group=self._tensor_parallel_group,
                )
                attention_mask = broadcast_tensor(
                    None,
                    device=self._device,
                    src=_group_src(self._tensor_parallel_group),
                    group=self._tensor_parallel_group,
                )
                result.append(
                    EmbeddingsProcessorOutput(
                        video_encoding=video_encoding,
                        audio_encoding=audio_encoding,
                        attention_mask=attention_mask,
                    )
                )
            cleanup_memory()
            if cache_enabled:
                self._prompt_embeddings_cache.clear()
                self._prompt_embeddings_cache[cache_key] = tuple(_clone_prompt_outputs(tuple(result)))
            logger.info("Prompt encoding received from rank0")
            return result

        if self._embeddings_processor_rank0_only and not is_rank0(self._tensor_parallel_group):
            result = []
            for _ in prompts:
                video_encoding = broadcast_tensor(
                    None,
                    device=self._device,
                    src=_group_src(self._tensor_parallel_group),
                    group=self._tensor_parallel_group,
                )
                audio_encoding = broadcast_tensor(
                    None,
                    device=self._device,
                    src=_group_src(self._tensor_parallel_group),
                    group=self._tensor_parallel_group,
                )
                attention_mask = broadcast_tensor(
                    None,
                    device=self._device,
                    src=_group_src(self._tensor_parallel_group),
                    group=self._tensor_parallel_group,
                )
                result.append(
                    EmbeddingsProcessorOutput(
                        video_encoding=video_encoding,
                        audio_encoding=audio_encoding,
                        attention_mask=attention_mask,
                    )
                )
            cleanup_memory()
            if cache_enabled:
                self._prompt_embeddings_cache.clear()
                self._prompt_embeddings_cache[cache_key] = tuple(_clone_prompt_outputs(tuple(result)))
            logger.info("Prompt embeddings received from rank0")
            return result

        if self._resident_embeddings_processor_enabled:
            if self._resident_embeddings_processor is None:
                self._resident_embeddings_processor = self._build_embeddings_processor()
            embeddings_processor_ctx = nullcontext(self._resident_embeddings_processor)
        else:
            embeddings_processor_ctx = gpu_model(self._build_embeddings_processor())
        with embeddings_processor_ctx as embeddings_processor:
            result = []
            for raw_index, raw_output in enumerate(raw_outputs):
                with profile_section(f"prompt_encoder.move_raw_output.{raw_index}", self._embeddings_processor_device):
                    hidden_states, attention_mask = move_raw_output(raw_output)
                with profile_section(f"prompt_encoder.embeddings_process.{raw_index}", self._embeddings_processor_device):
                    output = embeddings_processor.process_hidden_states(hidden_states, attention_mask)
                with profile_section(f"prompt_encoder.embeddings_output_move.{raw_index}", self._embeddings_processor_device):
                    prompt_output = EmbeddingsProcessorOutput(
                        video_encoding=output.video_encoding.contiguous().to(device=self._device, dtype=self._dtype),
                        audio_encoding=(
                            None
                            if output.audio_encoding is None
                            else output.audio_encoding.contiguous().to(device=self._device, dtype=output.audio_encoding.dtype)
                        ),
                        attention_mask=output.attention_mask.to(self._device),
                    )
                del output, hidden_states, attention_mask
                with profile_section(f"prompt_encoder.embeddings_cleanup.{raw_index}", self._embeddings_processor_device):
                    cleanup_memory()
                if self._tp_prompt_rank0_only or self._embeddings_processor_rank0_only:
                    broadcast_tensor(
                        prompt_output.video_encoding,
                        device=self._device,
                        src=_group_src(self._tensor_parallel_group),
                        group=self._tensor_parallel_group,
                    )
                    broadcast_tensor(
                        prompt_output.audio_encoding,
                        device=self._device,
                        src=_group_src(self._tensor_parallel_group),
                        group=self._tensor_parallel_group,
                    )
                    broadcast_tensor(
                        prompt_output.attention_mask,
                        device=self._device,
                        src=_group_src(self._tensor_parallel_group),
                        group=self._tensor_parallel_group,
                    )
                result.append(prompt_output)
        with profile_section("prompt_encoder.final_cleanup", self._device):
            cleanup_memory()
        if cache_enabled:
            self._prompt_embeddings_cache.clear()
            self._prompt_embeddings_cache[cache_key] = tuple(_clone_prompt_outputs(tuple(result)))
        logger.info("Prompt encoding complete")
        return result


# ---------------------------------------------------------------------------
# ImageConditioner
# ---------------------------------------------------------------------------


class ImageConditioner:
    """Owns video encoder lifecycle.
    Builds the encoder, passes it to the user-supplied callable, then frees it.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        resident: bool = False,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._resident = resident
        self._resident_encoder: VideoEncoder | None = None
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def _build_encoder(self) -> VideoEncoder:
        return self._encoder_builder.build(device=self._device, dtype=self._dtype).eval()

    def __call__(self, fn: Callable[[VideoEncoder], T]) -> T:
        """Build video encoder → call *fn(encoder)* → free encoder."""
        if self._resident:
            if self._resident_encoder is None:
                self._resident_encoder = self._build_encoder()
            return fn(self._resident_encoder)
        with gpu_model(self._build_encoder()) as encoder:
            return fn(encoder)


# ---------------------------------------------------------------------------
# VideoUpsampler
# ---------------------------------------------------------------------------


class VideoUpsampler:
    """Owns video encoder + spatial upsampler lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        upsampler_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        resident: bool = False,
        tensor_parallel: bool = False,
        tensor_parallel_group: HCCLGroup | None = None,
    ) -> None:
        self._upsampler_path = upsampler_path
        self._dtype = dtype
        self._device = device
        self._resident = resident
        self._tensor_parallel = tensor_parallel
        self._tensor_parallel_group = tensor_parallel_group
        self._resident_encoder: VideoEncoder | None = None
        self._resident_upsampler: torch.nn.Module | None = None
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VideoEncoderConfigurator,
            model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        use_tp_upsampler = tensor_parallel and os.getenv("LTX2_DISABLE_TP_UPSAMPLER", "").lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }
        module_ops = (
            (
                build_hccl_upsampler_tensor_parallel_op(
                    rank=rank(tensor_parallel_group),
                    world_size=world_size(tensor_parallel_group),
                    device=self._device,
                    process_group=tensor_parallel_group.process_group if tensor_parallel_group is not None else None,
                    label=tensor_parallel_group.name if tensor_parallel_group is not None else None,
                ),
            )
            if use_tp_upsampler
            else ()
        )
        if tensor_parallel and not use_tp_upsampler:
            logger.warning(
                "LTX2_DISABLE_TP_UPSAMPLER is set; running full-replica NPU upsampler on every rank for diagnostics"
            )
        self._upsampler_builder = Builder(
            model_path=upsampler_path,
            model_class_configurator=LatentUpsamplerConfigurator,
            registry=registry or DummyRegistry(),
            module_ops=module_ops,
        )

    def __call__(self, latent: torch.Tensor) -> torch.Tensor:
        """Upsample *latent* using video encoder + spatial upsampler, then free both."""
        logger.info("Building video encoder + spatial upsampler from %s", self._upsampler_path)
        if self._resident:
            if self._resident_encoder is None:
                with profile_section("video_upsampler.encoder_build", self._device):
                    self._resident_encoder = self._encoder_builder.build(device=self._device, dtype=self._dtype).eval()
            if self._resident_upsampler is None:
                with profile_section("video_upsampler.upsampler_build", self._device):
                    self._resident_upsampler = self._upsampler_builder.build(device=self._device, dtype=self._dtype).eval()
            with profile_section("video_upsampler.upsample", self._device):
                return upsample_video(latent=latent, video_encoder=self._resident_encoder, upsampler=self._resident_upsampler)
        with profile_section("video_upsampler.encoder_build", self._device):
            encoder = self._encoder_builder.build(device=self._device, dtype=self._dtype).eval()
        with profile_section("video_upsampler.upsampler_build", self._device):
            upsampler = self._upsampler_builder.build(device=self._device, dtype=self._dtype).eval()
        with (
            _profiled_gpu_model(encoder, teardown_profile="video_upsampler.encoder_teardown") as encoder,
            _profiled_gpu_model(upsampler, teardown_profile="video_upsampler.upsampler_teardown") as upsampler,
        ):
            with profile_section("video_upsampler.upsample", self._device):
                return upsample_video(latent=latent, video_encoder=encoder, upsampler=upsampler)

    def distributed_tensor_parallel(self, latent: torch.Tensor | None) -> torch.Tensor | None:
        """Run latent upsampling with the HCCL tensor-parallel upsampler wrapper."""
        if latent is None:
            raise ValueError("latent is required for tensor-parallel upsampling on every rank")
        return self(latent)


# ---------------------------------------------------------------------------
# VideoDecoder
# ---------------------------------------------------------------------------


class VideoDecoder:
    """Owns video decoder lifecycle.
    Returns an iterator that cleans up the decoder after all chunks are consumed.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        memory_efficient: bool = True,
        decoder_builder: BuilderProtocol | None = None,
        resident: bool = False,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._resident = resident
        self._resident_decoder: torch.nn.Module | None = None
        if decoder_builder is not None:
            self._decoder_builder = decoder_builder
        else:
            self._decoder_builder = Builder(
                model_path=checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=registry or DummyRegistry(),
                module_ops=(MEMORY_EFFICIENT_DECODE,) if memory_efficient and not is_npu_device(device) else (),
            )

    def __call__(
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None = None,
        generator: GeneratorLike = None,
    ) -> Iterator[torch.Tensor]:
        """Decode *latent* to pixel-space video chunks. Decoder freed after exhaustion."""
        logger.info("Building video decoder from %s", self._checkpoint_path)
        dump_tensor("video_decoder.input_latent", latent)
        latent = latent.to(device=self._device, dtype=self._dtype)
        if self._resident:
            if self._resident_decoder is None:
                with profile_section("video_decoder.build", self._device):
                    self._resident_decoder = self._decoder_builder.build(device=self._device, dtype=self._dtype).eval()
            chunks = self._resident_decoder.decode_video(latent, tiling_config, generator)
            chunks = _maybe_autocast_iter(
                chunks,
                self._device,
                env_name=_VIDEO_DECODER_AUTOCAST_ENV,
                scope="video decoder",
            )
            return _dump_first_chunk(chunks)
        with profile_section("video_decoder.build", self._device):
            decoder = self._decoder_builder.build(device=self._device, dtype=self._dtype).eval()
        chunks = decoder.decode_video(latent, tiling_config, generator)
        chunks = _maybe_autocast_iter(
            chunks,
            self._device,
            env_name=_VIDEO_DECODER_AUTOCAST_ENV,
            scope="video decoder",
        )
        return _cleanup_iter(_dump_first_chunk(chunks), decoder)


# ---------------------------------------------------------------------------
# AudioDecoder
# ---------------------------------------------------------------------------


class AudioDecoder:
    """Owns audio decoder + vocoder lifecycle."""

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
        resident: bool = False,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._dtype = dtype
        self._device = device
        self._resident = resident
        self._resident_decoder: torch.nn.Module | None = None
        self._resident_vocoder: torch.nn.Module | None = None
        self._decoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioDecoderConfigurator,
            model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )
        self._vocoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=VocoderConfigurator,
            model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def __call__(self, latent: torch.Tensor) -> Audio:
        """Decode audio *latent* through VAE decoder + vocoder, then free both."""
        logger.info("Building audio decoder + vocoder from %s", self._checkpoint_path)
        latent = latent.to(device=self._device, dtype=self._dtype)
        if self._resident:
            if self._resident_decoder is None:
                with profile_section("audio_decode.decoder_build", self._device):
                    self._resident_decoder = self._decoder_builder.build(device=self._device, dtype=self._dtype).eval()
            if self._resident_vocoder is None:
                with profile_section("audio_decode.vocoder_build", self._device):
                    self._resident_vocoder = self._vocoder_builder.build(device=self._device, dtype=self._dtype).eval()
            with profile_section("audio_decode.decode", self._device):
                return vae_decode_audio(latent, self._resident_decoder, self._resident_vocoder)
        with profile_section("audio_decode.decoder_build", self._device):
            decoder = self._decoder_builder.build(device=self._device, dtype=self._dtype).eval()
        with profile_section("audio_decode.vocoder_build", self._device):
            vocoder = self._vocoder_builder.build(device=self._device, dtype=self._dtype).eval()
        with (
            _profiled_gpu_model(decoder, teardown_profile="audio_decode.decoder_teardown") as decoder,
            _profiled_gpu_model(vocoder, teardown_profile="audio_decode.vocoder_teardown") as vocoder,
        ):
            with profile_section("audio_decode.decode", self._device):
                return vae_decode_audio(latent, decoder, vocoder)


# ---------------------------------------------------------------------------
# AudioEncoder
# ---------------------------------------------------------------------------


class AudioConditioner:
    """Owns audio encoder lifecycle.
    Builds the encoder, passes it to the user-supplied callable, then frees it.
    Mirrors :class:`ImageConditioner` for the audio modality.
    """

    def __init__(
        self,
        checkpoint_path: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: Registry | None = None,
    ) -> None:
        self._dtype = dtype
        self._device = device
        self._encoder_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=AudioEncoderConfigurator,
            model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
            registry=registry or DummyRegistry(),
        )

    def __call__(self, fn: Callable[[torch.nn.Module], T]) -> T:
        """Build audio encoder → call *fn(encoder)* → free encoder."""
        with gpu_model(self._encoder_builder.build(device=self._device, dtype=self._dtype).eval()) as encoder:
            return fn(encoder)

import functools
from pathlib import Path

import torch
from transformers import AutoImageProcessor, Gemma3ForConditionalGeneration, Gemma3Processor

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
from ltx_core.utils import find_matching_file


class GemmaTextEncoder(torch.nn.Module):
    """Pure Gemma text encoder — runs the LLM and returns raw hidden states.
    Prompt enhancement (generate) is also supported since the full
    Gemma3ForConditionalGeneration model (including lm_head) is loaded.
    """

    def __init__(
        self,
        model: Gemma3ForConditionalGeneration | None = None,
        tokenizer: LTXVGemmaTokenizer | None = None,
        processor: Gemma3Processor | None = None,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self._dtype = dtype

    def encode(
        self,
        text: str,
        padding_side: str = "left",  # noqa: ARG002
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """Run Gemma LLM and return raw hidden states + attention mask.
        Calls the inner model (self.model.model) to skip lm_head logits computation (~500 MiB saving).
        Returns:
            (hidden_states, attention_mask) where hidden_states is a tuple of per-layer tensors.
        """
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=self.model.device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=self.model.device)
        language_model = self.model.model.language_model
        layer_device_map = getattr(self, "gemma_layer_device_map", None)
        if layer_device_map is not None:
            original_forward = language_model.forward

            def forward_with_layer_moves(*args, **kwargs):
                original_layers = language_model.layers
                language_model.layers = _LayerwiseModuleList(original_layers, layer_device_map)
                try:
                    return original_forward(*args, **kwargs)
                finally:
                    language_model.layers = original_layers

            language_model.forward = forward_with_layer_moves
            try:
                outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            finally:
                language_model.forward = original_forward
        else:
            outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = []
        for idx, tensor in enumerate(outputs.hidden_states):
            try:
                hidden_states.append(tensor.to(device="cpu", dtype=self._dtype))
                if self.model.device.type == "npu" and hasattr(torch, "npu"):
                    torch.npu.synchronize(self.model.device)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"failed to collect Gemma hidden state {idx} from {tensor.device} "
                    f"with shape {tuple(tensor.shape)} to CPU"
                ) from exc
        del outputs
        return tuple(hidden_states), attention_mask.to(device="cpu")

    # --- Prompt enhancement methods ---

    def _enhance(
        self,
        messages: list[dict[str, str]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        text = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        model_inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        ).to(self.model.device)
        pad_token_id = self.processor.tokenizer.pad_token_id if self.processor.tokenizer.pad_token_id is not None else 0
        model_inputs = _pad_inputs_for_attention_alignment(model_inputs, pad_token_id=pad_token_id)

        with torch.inference_mode(), torch.random.fork_rng(devices=[self.model.device]):
            torch.manual_seed(seed)
            outputs = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
            )
            generated_ids = outputs[0][len(model_inputs.input_ids[0]) :]
            enhanced_prompt = self.processor.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return enhanced_prompt

    def enhance_t2v(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for T2V generation."""
        system_prompt = system_prompt or self.default_gemma_t2v_system_prompt

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"user prompt: {prompt}"},
        ]

        return self._enhance(messages, max_new_tokens=max_new_tokens, seed=seed)

    def enhance_i2v(
        self,
        prompt: str,
        image: torch.Tensor,
        max_new_tokens: int = 512,
        system_prompt: str | None = None,
        seed: int = 10,
    ) -> str:
        """Enhance a text prompt for I2V generation using a reference image."""
        system_prompt = system_prompt or self.default_gemma_i2v_system_prompt
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"User Raw Input Prompt: {prompt}."},
                ],
            },
        ]
        return self._enhance(messages, image=image, max_new_tokens=max_new_tokens, seed=seed)

    @functools.cached_property
    def default_gemma_i2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_i2v_system_prompt.txt")

    @functools.cached_property
    def default_gemma_t2v_system_prompt(self) -> str:
        return _load_system_prompt("gemma_t2v_system_prompt.txt")


# --- Standalone utility functions ---


@functools.lru_cache(maxsize=2)
def _load_system_prompt(prompt_name: str) -> str:
    with open(Path(__file__).parent / "prompts" / f"{prompt_name}", "r") as f:
        return f.read()


def _cat_with_padding(
    tensor: torch.Tensor,
    padding_length: int,
    value: int | float,
) -> torch.Tensor:
    """Concatenate a tensor with a padding tensor of the given value."""
    return torch.cat(
        [
            tensor,
            torch.full(
                (1, padding_length),
                value,
                dtype=tensor.dtype,
                device=tensor.device,
            ),
        ],
        dim=1,
    )


def _pad_inputs_for_attention_alignment(
    model_inputs: dict[str, torch.Tensor],
    pad_token_id: int = 0,
    alignment: int = 8,
) -> dict[str, torch.Tensor]:
    """Pad sequence length to multiple of alignment for Flash Attention compatibility."""
    seq_len = model_inputs.input_ids.shape[1]
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    padding_length = padded_len - seq_len

    if padding_length > 0:
        model_inputs["input_ids"] = _cat_with_padding(model_inputs.input_ids, padding_length, pad_token_id)
        model_inputs["attention_mask"] = _cat_with_padding(model_inputs.attention_mask, padding_length, 0)
        if "token_type_ids" in model_inputs and model_inputs["token_type_ids"] is not None:
            model_inputs["token_type_ids"] = _cat_with_padding(model_inputs["token_type_ids"], padding_length, 0)

    return model_inputs


def module_ops_from_gemma_root(gemma_root: str) -> tuple[ModuleOps, ...]:
    tokenizer_root = str(find_matching_file(gemma_root, "tokenizer.model").parent)
    processor_root = str(find_matching_file(gemma_root, "preprocessor_config.json").parent)

    def load_tokenizer(module: GemmaTextEncoder) -> GemmaTextEncoder:
        module.tokenizer = LTXVGemmaTokenizer(tokenizer_root, 1024)
        return module

    def load_processor(module: GemmaTextEncoder) -> GemmaTextEncoder:
        image_processor = AutoImageProcessor.from_pretrained(processor_root, local_files_only=True, use_fast=False)
        if not module.tokenizer:
            raise ValueError("Tokenizer model operation must be performed before processor model operation")
        module.processor = Gemma3Processor(image_processor=image_processor, tokenizer=module.tokenizer.tokenizer)
        return module

    tokenizer_load_ops = ModuleOps(
        "TokenizerLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.tokenizer is None,
        mutator=load_tokenizer,
    )
    processor_load_ops = ModuleOps(
        "ProcessorLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.processor is None,
        mutator=load_processor,
    )
    return (tokenizer_load_ops, processor_load_ops)


class _LayerwiseModuleList(torch.nn.ModuleList):
    def __init__(self, modules: torch.nn.ModuleList | list[torch.nn.Module], device_map: dict[int, torch.device], offset: int = 0):
        super().__init__(modules)
        self._device_map = device_map
        self._offset = offset

    def __iter__(self):
        for local_idx, layer in enumerate(super().__iter__()):
            idx = self._offset + local_idx
            target = self._device_map.get(idx)
            if target is None:
                yield layer
            else:
                yield _LayerDeviceWrapper(layer, target)

    def __getitem__(self, idx):
        modules = list(self._modules.values())
        if isinstance(idx, slice):
            indices = list(range(len(modules)))[idx]
            layers = modules[idx]
            if not indices:
                return _LayerwiseModuleList([], self._device_map, self._offset)
            if len(indices) == 1:
                step = 1
            else:
                step = indices[1] - indices[0]
            if step == 1:
                return _LayerwiseModuleList(layers, self._device_map, self._offset + indices[0])
            remapped = {local_idx: self._device_map.get(self._offset + absolute_idx) for local_idx, absolute_idx in enumerate(indices)}
            return _LayerwiseModuleList(layers, remapped, 0)
        layer = modules[idx]
        absolute_idx = self._offset + (idx if idx >= 0 else len(modules) + idx)
        target = self._device_map.get(absolute_idx)
        return layer if target is None else _LayerDeviceWrapper(layer, target)


class _LayerDeviceWrapper(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module, device: torch.device):
        super().__init__()
        self.layer = layer
        self.device = device

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("layer"), name)

    def _move_tensor(self, value, *, label: str = "tensor"):
        if isinstance(value, torch.Tensor):
            if value.device == self.device:
                return value.contiguous() if not value.is_contiguous() else value
            try:
                source = value.contiguous() if not value.is_contiguous() else value
                return source.to(self.device)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"failed to move Gemma layer {label} from {value.device} to {self.device}; "
                    f"shape={tuple(value.shape)} dtype={value.dtype} contiguous={value.is_contiguous()} stride={value.stride()}"
                ) from exc
        return value

    def _move_output(self, value):
        if isinstance(value, torch.Tensor):
            return self._move_tensor(value, label="output")
        if isinstance(value, tuple):
            return tuple(self._move_output(item) for item in value)
        if isinstance(value, list):
            return [self._move_output(item) for item in value]
        return value

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        hidden_states = self._move_tensor(hidden_states, label="hidden_states")
        kwargs = dict(kwargs)
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = self._move_tensor(kwargs["attention_mask"], label="attention_mask")
        needed_position_key = "position_embeddings_local" if getattr(self.layer.self_attn, "is_sliding", False) else "position_embeddings_global"
        if needed_position_key in kwargs:
            value = kwargs[needed_position_key]
            if isinstance(value, tuple):
                moved = tuple(self._move_tensor(item, label=f"{needed_position_key}[{idx}]") for idx, item in enumerate(value))
            else:
                moved = self._move_tensor(value, label=needed_position_key)
            kwargs["position_embeddings_global"] = moved
            kwargs["position_embeddings_local"] = moved
        output = self.layer(hidden_states, *args, **kwargs)
        return self._move_output(output)

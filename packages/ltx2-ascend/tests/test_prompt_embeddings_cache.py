from contextlib import nullcontext

import torch

from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_pipelines.utils import blocks


class _FakeTextEncoder:
    def __init__(self, calls: list[str]) -> None:
        self._calls = calls

    def encode(self, prompt: str):
        self._calls.append(prompt)
        value = float(len(self._calls))
        hidden = torch.full((1, 2, 3), value, dtype=torch.float32)
        mask = torch.ones((1, 2), dtype=torch.int64)
        return (hidden,), mask


class _FakeEmbeddingsProcessor:
    def __init__(self, calls: list[float]) -> None:
        self._calls = calls

    def process_hidden_states(self, hidden_states, attention_mask):
        value = float(hidden_states[0][0, 0, 0].item())
        self._calls.append(value)
        return EmbeddingsProcessorOutput(
            video_encoding=torch.full((1, 2, 4), value, dtype=torch.float32),
            audio_encoding=torch.full((1, 2, 2), value + 10.0, dtype=torch.float32),
            attention_mask=attention_mask.clone(),
        )


def _fake_prompt_encoder(monkeypatch):
    monkeypatch.setattr(blocks, "cleanup_memory", lambda: None)
    monkeypatch.setattr(blocks, "profile_section", lambda *args, **kwargs: nullcontext())

    text_calls: list[str] = []
    process_calls: list[float] = []
    encoder = blocks.PromptEncoder.__new__(blocks.PromptEncoder)
    encoder._gemma_root = "gemma"
    encoder._checkpoint_path = "checkpoint"
    encoder._dtype = torch.float16
    encoder._text_encoder_dtype = torch.float16
    encoder._embeddings_processor_dtype = torch.float32
    encoder._device = torch.device("cpu")
    encoder._text_encoder_device = torch.device("cpu")
    encoder._embeddings_processor_device = torch.device("cpu")
    encoder._tensor_parallel_group = None
    encoder._prompt_embeddings_cache = {}
    encoder._tp_prompt_rank0_only = False
    encoder._embeddings_processor_rank0_only = False
    encoder._resident_embeddings_processor_enabled = True
    encoder._resident_embeddings_processor = _FakeEmbeddingsProcessor(process_calls)
    encoder._text_encoder_ctx = lambda: nullcontext(_FakeTextEncoder(text_calls))
    return encoder, text_calls, process_calls


def test_prompt_encoder_reuses_cached_embeddings_and_returns_clones(monkeypatch):
    monkeypatch.delenv("LTX2_PROMPT_EMBEDDINGS_CACHE", raising=False)
    encoder, text_calls, process_calls = _fake_prompt_encoder(monkeypatch)

    first = encoder(["positive", "negative"])
    first[0].video_encoding.add_(1000)
    second = encoder(["positive", "negative"])

    assert text_calls == ["positive", "negative"]
    assert process_calls == [1.0, 2.0]
    torch.testing.assert_close(second[0].video_encoding, torch.full((1, 2, 4), 1.0, dtype=torch.float16))
    assert first[0].video_encoding.data_ptr() != second[0].video_encoding.data_ptr()


def test_prompt_embeddings_cache_can_be_disabled(monkeypatch):
    monkeypatch.setenv("LTX2_PROMPT_EMBEDDINGS_CACHE", "0")
    encoder, text_calls, process_calls = _fake_prompt_encoder(monkeypatch)

    encoder(["positive", "negative"])
    encoder(["positive", "negative"])

    assert text_calls == ["positive", "negative", "positive", "negative"]
    assert process_calls == [1.0, 2.0, 3.0, 4.0]


def test_prompt_embeddings_cache_key_scopes_enhanced_prompt_inputs():
    base = dict(
        prompts=["prompt"],
        dtype=torch.float16,
        text_encoder_dtype=torch.float16,
        embeddings_processor_dtype=torch.float32,
        device=torch.device("npu:0"),
        text_encoder_device=torch.device("npu:0"),
        embeddings_processor_device=torch.device("npu:0"),
    )

    plain_a = blocks._prompt_embeddings_cache_key(
        **base,
        enhance_first_prompt=False,
        enhance_prompt_image="image-a.png",
        enhance_prompt_seed=1,
    )
    plain_b = blocks._prompt_embeddings_cache_key(
        **base,
        enhance_first_prompt=False,
        enhance_prompt_image="image-b.png",
        enhance_prompt_seed=2,
    )
    enhanced_a = blocks._prompt_embeddings_cache_key(
        **base,
        enhance_first_prompt=True,
        enhance_prompt_image="image-a.png",
        enhance_prompt_seed=1,
    )
    enhanced_b = blocks._prompt_embeddings_cache_key(
        **base,
        enhance_first_prompt=True,
        enhance_prompt_image="image-a.png",
        enhance_prompt_seed=2,
    )

    assert plain_a == plain_b
    assert enhanced_a != enhanced_b
    assert plain_a != enhanced_a

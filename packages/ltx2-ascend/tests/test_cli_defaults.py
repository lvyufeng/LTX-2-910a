import os
from types import SimpleNamespace

import pytest

from ltx_pipelines.utils.constants import LTX_2_3_HQ_PARAMS, LTX_2_3_PARAMS
from ltx_pipelines.utils.helpers import assert_num_frames
from ltx2_ascend.cli import _parse_image, _set_default_attention_chunk, _validate_generation_args, parse_args


def test_standard_preset_uses_ltx_23_defaults():
    args = parse_args([])

    assert args.quality_preset == "standard"
    assert args.pipeline == "one-stage"
    assert args.height == LTX_2_3_PARAMS.stage_1_height
    assert args.width == LTX_2_3_PARAMS.stage_1_width
    assert args.frames == LTX_2_3_PARAMS.num_frames
    assert args.fps == LTX_2_3_PARAMS.frame_rate
    assert args.steps == LTX_2_3_PARAMS.num_inference_steps
    assert args.seed == LTX_2_3_PARAMS.seed
    assert args.video_cfg == LTX_2_3_PARAMS.video_guider_params.cfg_scale
    assert args.audio_cfg == LTX_2_3_PARAMS.audio_guider_params.cfg_scale
    assert args.video_stg_block == LTX_2_3_PARAMS.video_guider_params.stg_blocks
    assert args.audio_stg_block == LTX_2_3_PARAMS.audio_guider_params.stg_blocks


def test_hq_preset_uses_hq_defaults_and_lora_strengths():
    args = parse_args(["--quality-preset", "hq"])

    assert args.pipeline == "two-stage-hq"
    assert args.height == LTX_2_3_HQ_PARAMS.stage_2_height
    assert args.width == LTX_2_3_HQ_PARAMS.stage_2_width
    assert args.frames == LTX_2_3_HQ_PARAMS.num_frames
    assert args.steps == LTX_2_3_HQ_PARAMS.num_inference_steps
    assert args.video_stg == LTX_2_3_HQ_PARAMS.video_guider_params.stg_scale
    assert args.audio_stg == LTX_2_3_HQ_PARAMS.audio_guider_params.stg_scale
    assert args.video_stg_block == []
    assert args.audio_stg_block == []
    assert args.distilled_lora_strength_stage_1 == 0.25
    assert args.distilled_lora_strength_stage_2 == 0.5


def test_smoke_preset_preserves_fast_settings():
    args = parse_args(["--quality-preset", "smoke"])

    assert args.pipeline == "one-stage"
    assert args.height == 256
    assert args.width == 384
    assert args.frames == 17
    assert args.fps == 24.0
    assert args.steps == 4
    assert args.seed == 0
    assert args.video_stg_block == [28]
    assert args.audio_stg_block == [28]


def test_tp_hq_attention_chunk_default_preserves_validated_hq_override(monkeypatch):
    monkeypatch.delenv("LTX2_ASCEND_ATTENTION_CHUNK", raising=False)
    monkeypatch.delenv("LTX2_ASCEND_ATTENTION_CHUNK_MAX_MB", raising=False)
    monkeypatch.delenv("LTX2_ASCEND_TP_HQ_SHAPE_POLICY", raising=False)
    monkeypatch.delenv("LTX2_ASCEND_SOFTMAX_FP16", raising=False)
    args = SimpleNamespace(tensor_parallel=True, pipeline="two-stage-hq")

    _set_default_attention_chunk(args, SimpleNamespace(type="npu"))

    assert os.environ["LTX2_ASCEND_ATTENTION_CHUNK"] == "1536"
    assert os.environ["LTX2_ASCEND_ATTENTION_CHUNK_MAX_MB"] == "1600"
    assert os.environ["LTX2_ASCEND_TP_HQ_SHAPE_POLICY"] == "1"
    assert os.environ["LTX2_ASCEND_SOFTMAX_FP16"] == "1"


def test_tp_hq_attention_chunk_preserves_env_override(monkeypatch):
    monkeypatch.setenv("LTX2_ASCEND_ATTENTION_CHUNK", "512")
    monkeypatch.setenv("LTX2_ASCEND_ATTENTION_CHUNK_MAX_MB", "768")
    monkeypatch.setenv("LTX2_ASCEND_SOFTMAX_FP16", "0")
    monkeypatch.delenv("LTX2_ASCEND_TP_HQ_SHAPE_POLICY", raising=False)
    args = SimpleNamespace(tensor_parallel=True, pipeline="two-stage-hq")

    _set_default_attention_chunk(args, SimpleNamespace(type="npu"))

    assert os.environ["LTX2_ASCEND_ATTENTION_CHUNK"] == "512"
    assert os.environ["LTX2_ASCEND_ATTENTION_CHUNK_MAX_MB"] == "768"
    assert os.environ["LTX2_ASCEND_SOFTMAX_FP16"] == "0"
    assert "LTX2_ASCEND_TP_HQ_SHAPE_POLICY" not in os.environ


def test_tp_hq_attention_defaults_are_scoped_to_npu_tensor_parallel_hq(monkeypatch):
    for name in (
        "LTX2_ASCEND_ATTENTION_CHUNK",
        "LTX2_ASCEND_ATTENTION_CHUNK_MAX_MB",
        "LTX2_ASCEND_TP_HQ_SHAPE_POLICY",
        "LTX2_ASCEND_SOFTMAX_FP16",
    ):
        monkeypatch.delenv(name, raising=False)

    _set_default_attention_chunk(SimpleNamespace(tensor_parallel=False, pipeline="two-stage-hq"), SimpleNamespace(type="npu"))
    _set_default_attention_chunk(SimpleNamespace(tensor_parallel=True, pipeline="two-stage"), SimpleNamespace(type="npu"))
    _set_default_attention_chunk(SimpleNamespace(tensor_parallel=True, pipeline="two-stage-hq"), SimpleNamespace(type="cpu"))

    assert "LTX2_ASCEND_ATTENTION_CHUNK" not in os.environ
    assert "LTX2_ASCEND_ATTENTION_CHUNK_MAX_MB" not in os.environ
    assert "LTX2_ASCEND_TP_HQ_SHAPE_POLICY" not in os.environ
    assert "LTX2_ASCEND_SOFTMAX_FP16" not in os.environ


def test_user_overrides_survive_preset_resolution():
    args = parse_args(
        [
            "--quality-preset",
            "standard",
            "--pipeline",
            "two-stage",
            "--height",
            "512",
            "--width",
            "768",
            "--frames",
            "97",
            "--steps",
            "22",
            "--video-stg-block",
            "12",
            "--video-stg-block",
            "13",
        ]
    )

    assert args.pipeline == "two-stage"
    assert args.height == 512
    assert args.width == 768
    assert args.frames == 97
    assert args.steps == 22
    assert args.video_stg_block == [12, 13]


def test_hq_preset_rejects_non_hq_pipeline():
    args = parse_args(["--quality-preset", "hq", "--pipeline", "one-stage"])

    with pytest.raises(SystemExit, match="requires --pipeline two-stage-hq"):
        _validate_generation_args(args)


def test_two_stage_hq_requires_hq_preset():
    args = parse_args(["--quality-preset", "standard", "--pipeline", "two-stage-hq"])

    with pytest.raises(SystemExit, match="requires --quality-preset hq"):
        _validate_generation_args(args)


def test_smoke_two_stage_warns(caplog):
    args = parse_args(["--quality-preset", "smoke", "--pipeline", "two-stage"])

    _validate_generation_args(args)

    assert "not visual quality" in caplog.text


def test_parse_image_supports_colon_paths():
    image = _parse_image("/tmp/path:with:colon.png:17:0.5")

    assert image.path == "/tmp/path:with:colon.png"
    assert image.frame_idx == 17
    assert image.strength == 0.5


@pytest.mark.parametrize("spec", ["", "/tmp/a.png:-1:0.5", "/tmp/a.png:1:0"])
def test_parse_image_rejects_invalid_specs(spec):
    with pytest.raises(ValueError):
        _parse_image(spec)


@pytest.mark.parametrize("frames", [9, 17, 97, 121])
def test_valid_frame_counts(frames):
    assert_num_frames(frames)


@pytest.mark.parametrize("frames", [0, 2, 16, 120])
def test_invalid_frame_counts(frames):
    with pytest.raises(ValueError):
        assert_num_frames(frames)

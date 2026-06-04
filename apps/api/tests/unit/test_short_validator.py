import pytest

from app.services.format_validator import ShortValidator


def _valid_short_scene(n, **overrides):
    base = {
        "scene_number": n,
        "narration_text": f"Narration text for scene {n} that is long enough.",
        "visual_prompt": f"Visual prompt for scene {n} that is long enough.",
        "asset_type": "video_clip" if n == 1 else "ken_burns",
        "kb_motion": None if n == 1 else "zoom_in",
        "target_duration_seconds": 5.0,
    }
    base.update(overrides)
    return base


def _valid_short_payload(**overrides):
    base = {
        "_format": "short",
        "_version": 1,
        "scenes": [
            _valid_short_scene(1, asset_type="video_clip", kb_motion=None),
            _valid_short_scene(2, asset_type="ken_burns", kb_motion="zoom_in"),
            _valid_short_scene(3, asset_type="ken_burns", kb_motion="pan_left"),
        ],
        "target_total_duration": 45.0,
        "visual_style": "Cinematic short form",
        "audio_direction": "Upbeat with vocal clarity",
        "music_mood": "synthwave_hype",
        "voice_id": "voice_123",
        "subtitle_preset": "CENTER_POP_YELLOW",
    }
    base.update(overrides)
    return base


@pytest.mark.unit
class TestShortValidator:
    def test_should_validate_correct_payload(self):
        validator = ShortValidator()
        payload = _valid_short_payload()

        result = validator.validate(payload)

        assert result.valid is True
        assert result.validated_payload is not None
        assert result.validated_payload["_format"] == "short"
        assert result.validated_payload["_version"] == 1
        assert result.error_message is None

    def test_should_upgrade_scene_1_when_zero_video_clips(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(1, asset_type="ken_burns", kb_motion="pan_left"),
                _valid_short_scene(2, asset_type="ken_burns", kb_motion="zoom_in"),
                _valid_short_scene(3, asset_type="ken_burns", kb_motion="pan_right"),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is True
        scenes = result.validated_payload["scenes"]
        assert scenes[0]["asset_type"] == "video_clip"
        assert scenes[0]["kb_motion"] is None
        assert scenes[1]["asset_type"] == "ken_burns"
        assert scenes[2]["asset_type"] == "ken_burns"

    def test_should_downgrade_excess_video_clips(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(1, asset_type="video_clip", kb_motion=None),
                _valid_short_scene(2, asset_type="video_clip", kb_motion=None),
                _valid_short_scene(3, asset_type="video_clip", kb_motion=None),
                _valid_short_scene(4, asset_type="video_clip", kb_motion=None),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is True
        scenes = result.validated_payload["scenes"]
        video_clip_scenes = [s for s in scenes if s["asset_type"] == "video_clip"]
        assert len(video_clip_scenes) == 2
        # Highest scene numbers should be downgraded first
        assert scenes[0]["asset_type"] == "video_clip"
        assert scenes[1]["asset_type"] == "video_clip"
        assert scenes[2]["asset_type"] == "ken_burns"
        assert scenes[2]["kb_motion"] == "zoom_in"
        assert scenes[3]["asset_type"] == "ken_burns"
        assert scenes[3]["kb_motion"] == "zoom_in"

    def test_kb_motion_enforced_after_autofix(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(1, asset_type="video_clip", kb_motion=None),
                _valid_short_scene(2, asset_type="video_clip", kb_motion=None),
                _valid_short_scene(3, asset_type="video_clip", kb_motion=None),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is True
        scenes = result.validated_payload["scenes"]
        assert scenes[0]["asset_type"] == "video_clip"
        assert scenes[0]["kb_motion"] is None
        assert scenes[1]["asset_type"] == "video_clip"
        assert scenes[1]["kb_motion"] is None
        assert scenes[2]["asset_type"] == "ken_burns"
        assert scenes[2]["kb_motion"] == "zoom_in"

    def test_should_reject_total_duration_below_15(self):
        validator = ShortValidator()
        payload = _valid_short_payload(target_total_duration=14.9)

        result = validator.validate(payload)

        assert result.valid is False
        assert "target_total_duration" in result.error_message.lower()

    def test_should_reject_total_duration_above_90(self):
        validator = ShortValidator()
        payload = _valid_short_payload(target_total_duration=90.1)

        result = validator.validate(payload)

        assert result.valid is False
        assert "target_total_duration" in result.error_message.lower()

    def test_should_reject_fewer_than_2_scenes(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(1, asset_type="video_clip", kb_motion=None),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "scenes" in result.error_message.lower()

    def test_should_reject_more_than_12_scenes(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(i, asset_type="ken_burns", kb_motion="zoom_in")
                for i in range(1, 14)
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "scenes" in result.error_message.lower()

    def test_should_reject_empty_narration_text(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(1, narration_text=""),
                _valid_short_scene(2, asset_type="ken_burns", kb_motion="zoom_in"),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "narration_text" in result.error_message.lower()

    def test_should_reject_empty_visual_prompt(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(1, visual_prompt=""),
                _valid_short_scene(2, asset_type="ken_burns", kb_motion="zoom_in"),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "visual_prompt" in result.error_message.lower()

    def test_should_reject_whitespace_only_narration(self):
        validator = ShortValidator()
        scene = _valid_short_scene(1, narration_text=" " * 15)
        payload = _valid_short_payload(
            scenes=[
                scene,
                _valid_short_scene(2, asset_type="ken_burns", kb_motion="zoom_in"),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "empty" in result.error_message.lower()
        assert "narration_text" in result.error_message.lower()

    def test_should_reject_whitespace_only_visual_prompt(self):
        validator = ShortValidator()
        scene = _valid_short_scene(1, visual_prompt=" " * 15)
        payload = _valid_short_payload(
            scenes=[
                scene,
                _valid_short_scene(2, asset_type="ken_burns", kb_motion="zoom_in"),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "empty" in result.error_message.lower()
        assert "visual_prompt" in result.error_message.lower()

    def test_should_reject_kb_motion_on_video_clip(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(1, asset_type="video_clip", kb_motion="zoom_in"),
                _valid_short_scene(2, asset_type="ken_burns", kb_motion="pan_left"),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "kb_motion" in result.error_message.lower()

    def test_should_reject_missing_kb_motion_on_ken_burns(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(1, asset_type="video_clip", kb_motion=None),
                _valid_short_scene(2, asset_type="ken_burns", kb_motion=None),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is False
        assert "kb_motion" in result.error_message.lower()

    def test_should_reject_wrong_format_discriminator(self):
        validator = ShortValidator()
        payload = _valid_short_payload(**{"_format": "blog"})

        result = validator.validate(payload)

        assert result.valid is False

    def test_should_handle_empty_dict(self):
        validator = ShortValidator()

        result = validator.validate({})

        assert result.valid is False
        assert result.error_message is not None

    def test_should_use_alias_names_in_output(self):
        validator = ShortValidator()
        payload = _valid_short_payload()

        result = validator.validate(payload)

        assert result.valid is True
        assert "_format" in result.validated_payload
        assert "_version" in result.validated_payload

    def test_should_pass_with_exactly_2_video_clips(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(1, asset_type="video_clip", kb_motion=None),
                _valid_short_scene(2, asset_type="video_clip", kb_motion=None),
                _valid_short_scene(3, asset_type="ken_burns", kb_motion="pan_left"),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is True
        scenes = result.validated_payload["scenes"]
        assert scenes[0]["asset_type"] == "video_clip"
        assert scenes[1]["asset_type"] == "video_clip"
        assert scenes[2]["asset_type"] == "ken_burns"

    def test_should_downgrade_highest_scene_numbers_first(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(10, asset_type="video_clip", kb_motion=None),
                _valid_short_scene(20, asset_type="video_clip", kb_motion=None),
                _valid_short_scene(30, asset_type="video_clip", kb_motion=None),
                _valid_short_scene(40, asset_type="video_clip", kb_motion=None),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is True
        scenes = result.validated_payload["scenes"]
        assert scenes[0]["asset_type"] == "video_clip"  # scene 10
        assert scenes[1]["asset_type"] == "video_clip"  # scene 20
        assert scenes[2]["asset_type"] == "ken_burns"  # scene 30
        assert scenes[3]["asset_type"] == "ken_burns"  # scene 40

    def test_short_scene_is_hook_default(self):
        validator = ShortValidator()
        payload = _valid_short_payload()

        result = validator.validate(payload)

        assert result.valid is True
        for scene in result.validated_payload["scenes"]:
            assert scene["is_hook"] is False

    def test_short_scene_is_hook_true(self):
        validator = ShortValidator()
        payload = _valid_short_payload(
            scenes=[
                _valid_short_scene(
                    1, asset_type="video_clip", kb_motion=None, is_hook=True
                ),
                _valid_short_scene(2, asset_type="ken_burns", kb_motion="zoom_in"),
                _valid_short_scene(3, asset_type="ken_burns", kb_motion="pan_left"),
            ]
        )

        result = validator.validate(payload)

        assert result.valid is True
        assert result.validated_payload["scenes"][0]["is_hook"] is True

    def test_short_format_payload_visual_style_theme(self):
        validator = ShortValidator()
        payload = _valid_short_payload(visual_style_theme="cinematic")

        result = validator.validate(payload)

        assert result.valid is True
        assert result.validated_payload["visual_style_theme"] == "cinematic"

    def test_short_format_payload_visual_style_theme_optional(self):
        validator = ShortValidator()
        payload = _valid_short_payload()

        assert (
            "visual_style_theme" not in payload
            or payload.get("visual_style_theme") is None
        )

        result = validator.validate(payload)

        assert result.valid is True

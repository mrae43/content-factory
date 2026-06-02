import pytest

from app.services.subtitles import (
    generate_ass_file,
    PLATFORM_ASPECT_RATIOS_SHORT,
)


def _alignment(words: list[tuple[str, float, float]]) -> list[dict]:
    return [{"word": w, "start": s, "end": e} for w, s, e in words]


def _scenes(narrations: list[str]) -> list[dict]:
    return [
        {
            "scene_number": i + 1,
            "narration_text": text,
            "visual_prompt": "test visual",
            "asset_type": "video_clip",
            "target_duration_seconds": 5.0,
        }
        for i, text in enumerate(narrations)
    ]


@pytest.mark.unit
def test_generate_ass_file_structure():
    alignment = _alignment([("Hello", 0.0, 0.5), ("world", 0.6, 1.0)])
    scenes = _scenes(["Hello world"])

    ass = generate_ass_file(alignment, scenes, "CENTER_POP_YELLOW", "tiktok")

    assert "[Script Info]" in ass
    assert "[V4+ Styles]" in ass
    assert "[Events]" in ass
    assert "Dialogue:" in ass
    assert "CENTER_POP_YELLOW" in ass


@pytest.mark.unit
def test_karaoke_tags_present():
    alignment = _alignment([("First", 0.0, 0.4), ("word", 0.5, 0.9)])
    scenes = _scenes(["First word"])

    ass = generate_ass_file(alignment, scenes, "CENTER_POP_YELLOW", "tiktok")

    assert "\\k" in ass
    assert "First" in ass
    assert "word" in ass


@pytest.mark.unit
def test_platform_resolution_in_header():
    for platform, (w, h) in PLATFORM_ASPECT_RATIOS_SHORT.items():
        alignment = _alignment([("test", 0.0, 1.0)])
        scenes = _scenes(["test"])

        ass = generate_ass_file(alignment, scenes, "CENTER_POP_YELLOW", platform)

        assert f"PlayResX: {w}" in ass
        assert f"PlayResY: {h}" in ass


@pytest.mark.unit
def test_all_preset_styles_exist():
    alignment = _alignment([("word", 0.0, 1.0)])
    scenes = _scenes(["word"])

    for preset in ["CENTER_POP_YELLOW", "CLEAN_WHITE_LOWER", "NEON_BOXED"]:
        ass = generate_ass_file(alignment, scenes, preset, "tiktok")
        assert f"Style: {preset}" in ass


@pytest.mark.unit
def test_multiple_scenes_generate_multiple_dialogues():
    alignment = _alignment(
        [
            ("Scene", 0.0, 0.5),
            ("one", 0.6, 1.0),
            ("Scene", 1.1, 1.5),
            ("two", 1.6, 2.0),
        ]
    )
    scenes = _scenes(["Scene one", "Scene two"])

    ass = generate_ass_file(alignment, scenes, "CENTER_POP_YELLOW", "tiktok")

    lines = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert len(lines) == 2


@pytest.mark.unit
def test_invalid_preset_raises():
    with pytest.raises(ValueError):
        generate_ass_file([], [], "INVALID_PRESET", "tiktok")


@pytest.mark.unit
def test_empty_alignment_returns_header_only():
    scenes = _scenes(["Hello world"])
    ass = generate_ass_file([], scenes, "CENTER_POP_YELLOW", "tiktok")
    assert "[Events]" in ass
    assert "Dialogue:" not in ass


@pytest.mark.unit
def test_karaoke_durations_in_centiseconds():
    alignment = _alignment([("Quick", 0.0, 0.15), ("test", 0.2, 0.5)])
    scenes = _scenes(["Quick test"])

    ass = generate_ass_file(alignment, scenes, "CENTER_POP_YELLOW", "tiktok")

    # 0.15s -> 15cs, 0.3s -> 30cs
    assert "\\k15}Quick" in ass
    assert "\\k30}test" in ass


@pytest.mark.unit
def test_ass_time_format():
    alignment = _alignment([("word", 61.23, 62.45)])
    scenes = _scenes(["word"])

    ass = generate_ass_file(alignment, scenes, "CENTER_POP_YELLOW", "tiktok")

    line = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")][0]
    assert "0:01:01.23" in line
    assert "0:01:02.45" in line


@pytest.mark.unit
def test_remaining_words_assigned_to_last_scene():
    alignment = _alignment(
        [
            ("one", 0.0, 0.5),
            ("two", 0.6, 1.0),
            ("three", 1.1, 1.5),
        ]
    )
    scenes = _scenes(["one two"])  # only 2 words expected

    ass = generate_ass_file(alignment, scenes, "CENTER_POP_YELLOW", "tiktok")

    lines = [ln for ln in ass.splitlines() if ln.startswith("Dialogue:")]
    assert len(lines) == 1
    assert "one" in lines[0]
    assert "two" in lines[0]
    assert "three" in lines[0]


@pytest.mark.unit
def test_neon_preset_has_cyan_outline():
    alignment = _alignment([("word", 0.0, 1.0)])
    scenes = _scenes(["word"])

    ass = generate_ass_file(alignment, scenes, "NEON_BOXED", "tiktok")
    assert "NEON_BOXED" in ass
    assert "&HFFFF00&" in ass  # cyan outline in BGR


@pytest.mark.unit
def test_clean_white_lower_alignment():
    alignment = _alignment([("word", 0.0, 1.0)])
    scenes = _scenes(["word"])

    ass = generate_ass_file(alignment, scenes, "CLEAN_WHITE_LOWER", "tiktok")
    assert "CLEAN_WHITE_LOWER" in ass
    # Lower third uses bottom-center alignment (2) with larger marginv
    # Style definition ends with ...,MarginV,Encoding
    assert ",50,1" in ass

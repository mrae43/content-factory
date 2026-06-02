"""Subtitle generation for SHORT format pipeline.

Provides ASS subtitle file generation with karaoke word highlighting
and re-exports of short-format configuration constants.
"""

import logging
from typing import Dict, List, Tuple

from app.services.short_config import (
    DEFAULT_SUBTITLE_PRESET_MAP,
    KB_MOTION_PRESETS,
    PLATFORM_ASPECT_RATIOS_SHORT,
)

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SUBTITLE_PRESET_MAP",
    "generate_ass_file",
    "KB_MOTION_PRESETS",
    "PLATFORM_ASPECT_RATIOS_SHORT",
]

# ---------------------------------------------------------------------------
# Subtitle style presets
# ---------------------------------------------------------------------------

_SUBTITLE_STYLES: Dict[str, Dict[str, object]] = {
    "CENTER_POP_YELLOW": {
        "fontname": "Arial",
        "fontsize": 64,
        "primary": "&H00FFFF&",
        "secondary": "&H404040&",
        "outline": "&H000000&",
        "back": "&H000000&",
        "bold": -1,
        "italic": 0,
        "underline": 0,
        "strikeout": 0,
        "scalex": 100,
        "scaley": 100,
        "spacing": 0,
        "angle": 0,
        "borderstyle": 1,
        "outline_size": 3,
        "shadow": 1,
        "alignment": 2,
        "marginl": 10,
        "marginr": 10,
        "marginv": 30,
        "encoding": 1,
    },
    "CLEAN_WHITE_LOWER": {
        "fontname": "Arial",
        "fontsize": 48,
        "primary": "&HFFFFFF&",
        "secondary": "&H808080&",
        "outline": "&H000000&",
        "back": "&H000000&",
        "bold": 0,
        "italic": 0,
        "underline": 0,
        "strikeout": 0,
        "scalex": 100,
        "scaley": 100,
        "spacing": 0,
        "angle": 0,
        "borderstyle": 1,
        "outline_size": 2,
        "shadow": 0,
        "alignment": 2,
        "marginl": 10,
        "marginr": 10,
        "marginv": 50,
        "encoding": 1,
    },
    "NEON_BOXED": {
        "fontname": "Arial",
        "fontsize": 56,
        "primary": "&HFFFFFF&",
        "secondary": "&H808000&",
        "outline": "&HFFFF00&",
        "back": "&H000000&",
        "bold": -1,
        "italic": 0,
        "underline": 0,
        "strikeout": 0,
        "scalex": 100,
        "scaley": 100,
        "spacing": 0,
        "angle": 0,
        "borderstyle": 1,
        "outline_size": 3,
        "shadow": 2,
        "alignment": 5,
        "marginl": 10,
        "marginr": 10,
        "marginv": 30,
        "encoding": 1,
    },
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format H:MM:SS.cc"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centis = int(round((seconds - int(seconds)) * 100))
    return f"{hours}:{minutes:02d}:{secs:02d}.{centis:02d}"


def _match_words_to_scenes(
    vocal_alignment_data: List[Dict],
    scenes: List[Dict],
) -> List[Tuple[int, List[Dict]]]:
    """Match word-level alignment entries to scenes by word count."""
    word_index = 0
    scene_groups: List[Tuple[int, List[Dict]]] = []
    for scene_idx, scene in enumerate(scenes):
        narration = scene.get("narration_text", "").strip()
        clean = narration.rstrip(".!?,; ")
        expected = len(clean.split()) if clean else 0
        matched: List[Dict] = []
        for _ in range(expected):
            if word_index < len(vocal_alignment_data):
                matched.append(vocal_alignment_data[word_index])
                word_index += 1
        scene_groups.append((scene_idx, matched))
    # Distribute any remaining words to the last scene
    if word_index < len(vocal_alignment_data) and scene_groups:
        last = len(scene_groups) - 1
        scene_groups[last] = (
            scene_groups[last][0],
            scene_groups[last][1] + vocal_alignment_data[word_index:],
        )
    return scene_groups


def _build_karaoke_text(words: List[Dict]) -> str:
    """Build a karaoke text line with {\\k} timing tags."""
    parts: List[str] = []
    for entry in words:
        word = entry.get("word", "")
        start = entry.get("start", 0.0)
        end = entry.get("end", 0.0)
        dur_cs = max(1, int((end - start) * 100))
        parts.append(f"{{\\k{dur_cs}}}{word}")
    return " ".join(parts)


def _build_ass_header(width: int, height: int, subtitle_preset: str) -> str:
    """Build the ASS file header including the selected style."""
    style = _SUBTITLE_STYLES[subtitle_preset]
    style_str = (
        f"{subtitle_preset},"
        f"{style['fontname']},"
        f"{style['fontsize']},"
        f"{style['primary']},"
        f"{style['secondary']},"
        f"{style['outline']},"
        f"{style['back']},"
        f"{style['bold']},"
        f"{style['italic']},"
        f"{style['underline']},"
        f"{style['strikeout']},"
        f"{style['scalex']},"
        f"{style['scaley']},"
        f"{style['spacing']},"
        f"{style['angle']},"
        f"{style['borderstyle']},"
        f"{style['outline_size']},"
        f"{style['shadow']},"
        f"{style['alignment']},"
        f"{style['marginl']},"
        f"{style['marginr']},"
        f"{style['marginv']},"
        f"{style['encoding']}"
    )
    return (
        "[Script Info]\n"
        "Title: Short Format Subtitles\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        "WrapStyle: 0\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, "
        "SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, "
        "MarginV, Encoding\n"
        f"Style: {style_str}\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, "
        "MarginR, MarginV, Effect, Text\n"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_ass_file(
    vocal_alignment_data: List[Dict],
    scenes: List[Dict],
    subtitle_preset: str,
    platform: str,
) -> str:
    """Generate an ASS subtitle file with karaoke highlighting."""
    if subtitle_preset not in _SUBTITLE_STYLES:
        raise ValueError(f"Unknown subtitle preset: {subtitle_preset}")

    width, height = PLATFORM_ASPECT_RATIOS_SHORT.get(platform, (1080, 1920))
    header = _build_ass_header(width, height, subtitle_preset)

    scene_groups = _match_words_to_scenes(vocal_alignment_data, scenes)
    events: List[str] = []
    for _scene_idx, words in scene_groups:
        if not words:
            continue
        start = words[0].get("start", 0.0)
        end = words[-1].get("end", 0.0)
        start_ass = _seconds_to_ass_time(start)
        end_ass = _seconds_to_ass_time(end)
        text = _build_karaoke_text(words)
        events.append(
            f"Dialogue: 0,{start_ass},{end_ass},{subtitle_preset},,0,0,0,,{text}"
        )

    return header + "\n".join(events)

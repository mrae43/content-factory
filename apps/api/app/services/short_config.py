"""Short-format pipeline constants and configuration helpers.

Placed here (instead of inside tts.py) to avoid circular imports
when other modules need platform defaults before the TTS provider
layer is loaded.
"""

from app.core.config import settings

# ---------------------------------------------------------------------------
# Platform defaults
# ---------------------------------------------------------------------------

DEFAULT_VOICE_MAP: dict[str, str] = {
    "tiktok": settings.default_voice_map.get("tiktok", ""),
    "instagram": settings.default_voice_map.get("instagram", ""),
    "youtube": settings.default_voice_map.get("youtube", ""),
}

DEFAULT_SUBTITLE_PRESET_MAP: dict[str, str] = {
    "tiktok": "CENTER_POP_YELLOW",
    "youtube": "CLEAN_WHITE_LOWER",
    "instagram": "NEON_BOXED",
}

# ---------------------------------------------------------------------------
# KB Motion preset → FFmpeg zoompan expressions
# ---------------------------------------------------------------------------

KB_MOTION_PRESETS: dict[str, dict[str, str]] = {
    "pan_left": {
        "zoompan": "z='1.001':x='iw*0.5-(iw*0.5-iw*0.3)*on/(duration*fps)':y='ih*0.5':d=duration*fps"
    },
    "pan_right": {
        "zoompan": "z='1.001':x='iw*0.3+(iw*0.5-iw*0.3)*on/(duration*fps)':y='ih*0.5':d=duration*fps"
    },
    "zoom_in": {
        "zoompan": "z='min(zoom+0.0015,1.5)':x='iw*0.5':y='ih*0.5':d=duration*fps"
    },
    "zoom_out": {
        "zoompan": "z='if(lte(zoom,1.0),1.5,max(zoom-0.0015,1.0))':x='iw*0.5':y='ih*0.5':d=duration*fps"
    },
    "static_zoom_in": {"zoompan": "z='1.001':x='iw*0.5':y='ih*0.5':d=duration*fps"},
}

# ---------------------------------------------------------------------------
# SHORT-specific platform dimensions
# ---------------------------------------------------------------------------

PLATFORM_ASPECT_RATIOS_SHORT: dict[str, tuple[int, int]] = {
    "tiktok": (1080, 1920),  # 9:16
    "instagram": (1080, 1350),  # 4:5
    "youtube": (1080, 1920),  # 9:16 (YouTube Shorts)
}

# ---------------------------------------------------------------------------
# Background Music — mood → filename mapping for S3 library
# ---------------------------------------------------------------------------

MUSIC_MOOD_MAP: dict[str, str] = {
    "dark_lofi": "01.mp3",
    "synthwave_hype": "01.mp3",
    "calm_informative": "01.mp3",
}

MUSIC_VOLUME: float = 0.1  # ~-20dB relative to voiceover

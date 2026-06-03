from pydantic import BaseModel, Field, ConfigDict, model_validator
from typing import List, Optional, Literal
from uuid import UUID


class VideoScene(BaseModel):
    scene_number: int = Field(ge=1)
    narration_text: str = Field(min_length=10)
    visual_prompt: str = Field(min_length=10)
    audio_cue: str = ""
    duration_seconds: float = Field(ge=3.0, le=60.0)


class VideoFormatPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    format: Literal["video"] = Field(alias="_format", default="video")
    version: int = Field(alias="_version", default=1)
    scenes: List[VideoScene] = Field(min_length=3)
    total_duration_seconds: float = Field(ge=60.0, le=300.0)
    visual_style: str = Field(min_length=5)
    audio_direction: str = ""
    unified_visual_prompt: str = Field(min_length=20)


class BlogSection(BaseModel):
    heading: str
    body: str
    key_takeaway: str
    sources_used: List[UUID] = Field(default_factory=list)
    word_count: int


class SeoMeta(BaseModel):
    meta_title: str
    meta_description: str
    keywords: List[str]
    canonical_url: Optional[str] = None


class BlogFormatPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    format: Literal["blog"] = Field(alias="_format", default="blog")
    version: int = Field(alias="_version", default=1)
    title: str
    subtitle: str
    sections: List[BlogSection]
    seo_meta: SeoMeta
    tags: List[str]
    call_to_action: str


class CarouselSlide(BaseModel):
    slide_number: int
    text: str
    visual_description: str
    hook_type: str
    sources_used: List[UUID] = Field(default_factory=list)
    image_url: Optional[str] = None


class CarouselFormatPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    format: Literal["carousel"] = Field(alias="_format", default="carousel")
    version: int = Field(alias="_version", default=1)
    slides: List[CarouselSlide]
    thread_title: str
    hashtags: List[str]
    cta_slide: str
    char_limit_violations: List[str] = Field(default_factory=list)


class ShortScene(BaseModel):
    scene_number: int = Field(..., ge=1)
    is_hook: bool = False
    narration_text: str = Field(..., min_length=10)
    visual_prompt: str = Field(..., min_length=10)
    asset_type: Literal["video_clip", "ken_burns"] = Field(...)
    kb_motion: Optional[
        Literal[
            "pan_left",
            "pan_right",
            "zoom_in",
            "zoom_out",
            "static_zoom_in",
        ]
    ] = Field(None)
    sfx_cue: Optional[str] = Field(None)
    target_duration_seconds: float = Field(..., ge=3.0, le=15.0)

    @model_validator(mode="after")
    def validate_kb_motion(self):
        if self.asset_type == "ken_burns" and self.kb_motion is None:
            raise ValueError("kb_motion is required when asset_type is 'ken_burns'")
        if self.asset_type == "video_clip" and self.kb_motion is not None:
            raise ValueError("kb_motion must be None when asset_type is 'video_clip'")
        return self


class ShortFormatPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    format: Literal["short"] = Field(alias="_format", default="short")
    version: int = Field(alias="_version", default=1)
    scenes: List[ShortScene] = Field(..., min_length=2, max_length=12)
    target_total_duration: float = Field(..., ge=15.0, le=90.0)
    visual_style: str = Field(..., min_length=5)
    audio_direction: str = Field(...)
    music_mood: str = Field(...)
    voice_id: str = Field(...)
    subtitle_preset: Literal[
        "CENTER_POP_YELLOW",
        "CLEAN_WHITE_LOWER",
        "NEON_BOXED",
    ] = Field(default="CENTER_POP_YELLOW")
    loop_hook: Optional[str] = Field(None)
    visual_style_theme: Optional[str] = Field(
        None, description="e.g., cinematic, minimalist, newsroom, documentary, dynamic"
    )

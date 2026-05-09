from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal
from uuid import UUID


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
    visual_prompt: str
    hook_type: str
    sources_used: List[UUID] = Field(default_factory=list)


class CarouselFormatPayload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    format: Literal["carousel"] = Field(alias="_format", default="carousel")
    version: int = Field(alias="_version", default=1)
    slides: List[CarouselSlide]
    thread_title: str
    hashtags: List[str]
    cta_slide: str
    char_limit_violations: List[str] = Field(default_factory=list)

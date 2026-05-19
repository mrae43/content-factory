import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.workers.agents import AgentActionStatus, AgentResult, BaseAgent
from app.schemas.formats import BlogSection, SeoMeta, CarouselSlide, VideoScene

logger = logging.getLogger(__name__)


def _build_hedge_block(hedge_index: list) -> str:
    if not hedge_index:
        return ""
    hedge_lines = "\n".join(
        f'  - "{c.get("claim_text", "")}" [{c.get("verdict", "UNCERTAIN")}]'
        for c in hedge_index
    )
    return (
        "UNCERTAIN CLAIMS — apply hedged language to each of these:\n"
        f"{hedge_lines}\n\n"
        "Hedging rules:\n"
        '  - Statistics:    "figures suggest..." / "estimates indicate..."\n'
        '  - Attributions:  "reportedly..." / "according to some sources..."\n'
        '  - Causal:        "research suggests a link between..." / "may contribute to..."\n'
        '  - Never: "studies prove", "it is a fact that", "definitively"\n'
    )


class BlogOutlineSection(BaseModel):
    heading: str
    key_points: List[str]
    sources_to_cite: List[str] = []


class BlogPlan(BaseModel):
    proposed_title: str
    proposed_subtitle: str
    sections: List[BlogOutlineSection]
    target_tags: List[str]
    cta_direction: str


class BlogFormatterOutput(BaseModel):
    title: str
    subtitle: str
    sections: List[BlogSection] = Field(min_length=1)
    seo_meta: SeoMeta
    tags: List[str]
    call_to_action: str


class CarouselOutlineSlide(BaseModel):
    slide_number: int
    purpose: str
    hook_type: str
    key_claim: str = ""


class CarouselPlan(BaseModel):
    thread_title: str
    slides: List[CarouselOutlineSlide]
    hashtags: List[str]
    cta_direction: str


class CarouselFormatterOutput(BaseModel):
    slides: List[CarouselSlide]
    thread_title: str
    hashtags: List[str]
    cta_slide: str


class VideoSceneOutline(BaseModel):
    scene_number: int
    purpose: str
    key_visual: str
    duration_estimate: float


class VideoPlan(BaseModel):
    proposed_title: str
    scene_outline: List[VideoSceneOutline]
    visual_style_direction: str


class VideoFormatterOutput(BaseModel):
    scenes: List[VideoScene]
    total_duration_seconds: float
    visual_style: str
    audio_direction: str


BLOG_PLAN_SYSTEM = (
    "You are the Blog Formatter Planner at the AI Content Factory.\n"
    "Your task is to PLAN a blog article structure from a verified video script.\n"
    "You do NOT write the article — you produce a structured outline.\n\n"
    "## PLANNING RULES\n"
    "1. Identify the 4-8 most important sections from the script.\n"
    "2. For each section, list 2-4 key points to cover.\n"
    "3. Note which verified claims or sources should be cited in each section.\n"
    "4. Propose an SEO-friendly title and subtitle.\n"
    "5. Suggest 3-8 tags.\n"
    "6. Define the call-to-action direction.\n"
    "7. Ensure every factual claim in the script is assigned to a section.\n"
    "8. Order sections for narrative flow: hook → context → depth → takeaway → CTA."
)

BLOG_PLAN_HUMAN = (
    "Plan a blog article outline from the following verified video script.\n\n"
    "<script>\n{script_content}\n</script>\n\n"
    "<refined_context>\n{refined_context}\n</refined_context>\n\n"
    "<verified_claims>\n{verified_claims}\n</verified_claims>\n\n"
    "<correction_hint>\n{correction_hint}\n</correction_hint>\n\n"
    "Produce a structured outline with section headings, key points per section, "
    "and source citations. Do NOT write the article content."
)

BLOG_FORMATTER_SYSTEM = (
    "You are the Blog Formatter Agent at the AI Content Factory.\n"
    "You receive a verified script, refined research context, verified claims, and a PLAN.\n"
    "Your task is to execute the plan — generate the full blog article following the outline.\n\n"
    "## RULES\n"
    "1. Every factual claim in the blog MUST trace to the verified claims or refined_context.\n"
    "2. Follow the plan's section structure — use the headings and key points as your guide.\n"
    "3. Each section must include a key_takeaway summarizing its core point.\n"
    "4. Include word_count for each section (count words in the body field).\n"
    "5. sources_used: list ResearchChunk UUIDs cited in each section.\n"
    "6. SEO: meta_title ≤ 60 chars, meta_description ≤ 160 chars, 5-10 keywords.\n"
    "7. Tags: use the plan's suggested tags, adjust if needed (3-8 tags).\n"
    "8. call_to_action: follow the plan's CTA direction.\n"
    "9. Write in an authoritative but accessible tone.\n"
    "10. Structure: 4-8 sections with clear headings.\n"
    "11. CRITICAL: Do NOT mention key_takeaway, word_count, or sources_used "
    "inside the body text. These fields are displayed separately in the UI — "
    "including them inline creates redundancy."
)

BLOG_FORMATTER_HUMAN = (
    "Execute the following blog plan to produce a complete article.\n\n"
    "<plan>\n{plan}\n</plan>\n\n"
    "<script>\n{script_content}\n</script>\n\n"
    "<refined_context>\n{refined_context}\n</refined_context>\n\n"
    "<verified_claims>\n{verified_claims}\n</verified_claims>\n\n"
    "<correction_hint>\n{correction_hint}\n</correction_hint>\n\n"
    "Follow the plan's section structure and key points. Generate the full article with SEO metadata."
)

CAROUSEL_PLAN_SYSTEM = (
    "You are the Carousel Formatter Planner at the AI Content Factory.\n"
    "Your task is to PLAN a social media carousel structure from a verified video script.\n"
    "You do NOT write the carousel — you produce a structured slide outline.\n\n"
    "## PLANNING RULES\n"
    "1. Design 8-12 slides with a clear narrative arc.\n"
    "2. First slide: hook (question, statistic, or bold statement).\n"
    "3. Last slide: CTA direction.\n"
    "4. Each slide has a purpose and hook_type.\n"
    "5. Note which verified claims map to which slides.\n"
    "6. Consider the platform's char limits when scoping slide content.\n"
    "7. Suggest 3-5 hashtags and a thread title."
)

CAROUSEL_PLAN_HUMAN = (
    "Plan a social media carousel outline from the following verified video script.\n\n"
    "<script>\n{script_content}\n</script>\n\n"
    "<refined_context>\n{refined_context}\n</refined_context>\n\n"
    "<verified_claims>\n{verified_claims}\n</verified_claims>\n\n"
    "<platform>\n{platform}\n</platform>\n\n"
    "<correction_hint>\n{correction_hint}\n</correction_hint>\n\n"
    "Produce a structured slide outline with purpose, hook_type, and key claim per slide. "
    "Do NOT write the slide text."
)

CAROUSEL_FORMATTER_SYSTEM = (
    "You are the Carousel Formatter Agent at the AI Content Factory.\n"
    "You receive a verified script, refined research context, verified claims, platform constraints, and a PLAN.\n"
    "Your task is to execute the plan — generate the full carousel following the outline.\n\n"
    "## RULES\n"
    "1. Every factual claim MUST trace to the verified claims or refined_context.\n"
    "2. Follow the plan's slide sequence and purposes.\n"
    "3. Each slide text MUST stay within platform character limits:\n"
    "   - Twitter: 280 chars per slide\n"
    "   - LinkedIn: 700 chars per slide\n"
    "   - Instagram: 2200 chars per slide\n"
    "   - Default (no platform): 500 chars per slide\n"
    "4. Include a visual_description for each slide — a caption describing the visual element.\n"
    "5. hook_type per slide: question, statistic, quote, visual, story, cta\n"
    "6. sources_used: list ResearchChunk UUIDs cited per slide.\n"
    "7. First slide must hook the reader (question or bold statement).\n"
    "8. Last slide is the CTA (use cta_slide field for this).\n"
    "9. Target 8-12 slides for a complete carousel.\n"
    "10. Each slide should be self-contained but flow as a narrative.\n"
    "11. Include 3-5 relevant hashtags."
)

CAROUSEL_FORMATTER_HUMAN = (
    "Execute the following carousel plan to produce a complete slide deck.\n\n"
    "<plan>\n{plan}\n</plan>\n\n"
    "<script>\n{script_content}\n</script>\n\n"
    "<refined_context>\n{refined_context}\n</refined_context>\n\n"
    "<verified_claims>\n{verified_claims}\n</verified_claims>\n\n"
    "<platform>\n{platform}\n</platform>\n\n"
    "<correction_hint>\n{correction_hint}\n</correction_hint>\n\n"
    "Follow the plan's slide sequence and purposes. Generate the full carousel with visual descriptions."
)

VIDEO_PLAN_SYSTEM = (
    "You are the Video Formatter Planner at the AI Content Factory.\n"
    "Your task is to PLAN a video scene structure from a verified narrative script.\n"
    "You do NOT produce the final scenes — you produce a structured scene outline.\n\n"
    "## PLANNING RULES\n"
    "1. Break the script into 3-8 distinct scenes with a clear narrative arc.\n"
    "2. First scene: hook — grab attention with a striking visual or provocative opening.\n"
    "3. Last scene: closer — end with a call-to-action or thought-provoking statement.\n"
    "4. Each scene needs a purpose, key visual description, and duration estimate.\n"
    "5. Note which verified claims map to which scenes.\n"
    "6. Total duration must be 60-300 seconds (1-5 minutes).\n"
    "7. Each scene should be 5-45 seconds.\n"
    "8. Define a visual style direction (cinematic, documentary, animated, etc.)."
)

VIDEO_PLAN_HUMAN = (
    "Plan a video scene outline from the following verified narrative script.\n\n"
    "<script>\n{script_content}\n</script>\n\n"
    "<refined_context>\n{refined_context}\n</refined_context>\n\n"
    "<verified_claims>\n{verified_claims}\n</verified_claims>\n\n"
    "<correction_hint>\n{correction_hint}\n</correction_hint>\n\n"
    "Produce a structured scene outline with scene number, purpose, key visual, "
    "and duration estimate per scene. Do NOT write the full narration or visual prompts."
)

VIDEO_FORMATTER_SYSTEM = (
    "You are the Video Formatter Agent at the AI Content Factory.\n"
    "You receive a verified narrative script, refined research context, verified claims, and a PLAN.\n"
    "Your task is to execute the plan — generate complete video scenes with narration and visual direction.\n\n"
    "## RULES\n"
    "1. Every factual claim MUST trace to the verified claims or refined_context.\n"
    "2. Follow the plan's scene sequence, purposes, and duration estimates.\n"
    "3. Each scene must have:\n"
    "   - narration_text: the voiceover narration (min 10 chars)\n"
    "   - visual_prompt: a detailed visual description for AI video generation (min 10 chars)\n"
    "   - audio_cue: background music or SFX direction\n"
    "   - duration_seconds: scene length (3-60 seconds)\n"
    "4. Total duration must be 60-300 seconds.\n"
    "5. Minimum 3 scenes.\n"
    "6. visual_style: describe the overall visual style of the video.\n"
    "7. audio_direction: describe the overall audio/music direction.\n"
    "8. Scene transitions should feel natural and maintain narrative flow.\n"
    "9. Write narration in a conversational, engaging tone.\n"
    "10. Visual prompts should be cinematic and specific (camera angles, lighting, colors)."
)

VIDEO_FORMATTER_HUMAN = (
    "Execute the following video plan to produce complete scenes.\n\n"
    "<plan>\n{plan}\n</plan>\n\n"
    "<script>\n{script_content}\n</script>\n\n"
    "<refined_context>\n{refined_context}\n</refined_context>\n\n"
    "<verified_claims>\n{verified_claims}\n</verified_claims>\n\n"
    "<correction_hint>\n{correction_hint}\n</correction_hint>\n\n"
    "Follow the plan's scene structure. Generate complete scenes with narration, visual prompts, "
    "and audio cues. Include overall visual_style and audio_direction."
)


class BlogFormatterAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        script_content = context.get("script_content", "")
        refined_context = context.get("refined_context", "")
        verified_claims = context.get("verified_claims", [])
        hedge_index = context.get("hedge_index", [])
        correction_hint = context.get("correction_hint", "")

        if not script_content:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No script content provided for blog formatting.",
                confidence_score=0.0,
            )

        if not refined_context:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No refined context provided for blog formatting.",
                confidence_score=0.0,
            )

        claims_text = "\n".join(
            f"- {c.get('claim_text', '')} [{c.get('verdict', 'UNKNOWN')}]: {c.get('evidence_text', 'N/A')}"
            for c in verified_claims
        )

        hedge_block = _build_hedge_block(hedge_index)
        blog_plan_system = (
            f"{hedge_block}{BLOG_PLAN_SYSTEM}" if hedge_block else BLOG_PLAN_SYSTEM
        )

        plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", blog_plan_system),
                ("human", BLOG_PLAN_HUMAN),
            ]
        )
        plan_chain = plan_prompt | self.llm.with_structured_output(BlogPlan)
        plan: BlogPlan = await plan_chain.ainvoke(
            {
                "script_content": script_content,
                "refined_context": refined_context,
                "verified_claims": claims_text,
                "correction_hint": correction_hint,
            }
        )
        plan_text = plan.model_dump_json(indent=2)
        logger.info("Blog plan produced: %d sections", len(plan.sections))

        blog_exec_system = (
            f"{hedge_block}{BLOG_FORMATTER_SYSTEM}"
            if hedge_block
            else BLOG_FORMATTER_SYSTEM
        )
        exec_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", blog_exec_system),
                ("human", BLOG_FORMATTER_HUMAN),
            ]
        )
        exec_chain = exec_prompt | self.llm.with_structured_output(BlogFormatterOutput)
        result: BlogFormatterOutput = await exec_chain.ainvoke(
            {
                "plan": plan_text,
                "script_content": script_content,
                "refined_context": refined_context,
                "verified_claims": claims_text,
                "correction_hint": correction_hint,
            }
        )

        payload = result.model_dump()
        payload["_format"] = "blog"
        payload["_version"] = 1

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload=payload,
            reasoning="Blog format generated via Plan→Execute from verified script and research context.",
            confidence_score=0.9,
            metadata={
                "model": self.model_name,
                "agent": "blog_formatter",
                "planned_sections": len(plan.sections),
                "generated_sections": len(result.sections),
            },
        )


class CarouselFormatterAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        script_content = context.get("script_content", "")
        refined_context = context.get("refined_context", "")
        verified_claims = context.get("verified_claims", [])
        hedge_index = context.get("hedge_index", [])
        platform = context.get("platform", "")
        correction_hint = context.get("correction_hint", "")

        if not script_content:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No script content provided for carousel formatting.",
                confidence_score=0.0,
            )

        if not refined_context:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No refined context provided for carousel formatting.",
                confidence_score=0.0,
            )

        claims_text = "\n".join(
            f"- {c.get('claim_text', '')} [{c.get('verdict', 'UNKNOWN')}]: {c.get('evidence_text', 'N/A')}"
            for c in verified_claims
        )

        hedge_block = _build_hedge_block(hedge_index)
        carousel_plan_system = (
            f"{hedge_block}{CAROUSEL_PLAN_SYSTEM}"
            if hedge_block
            else CAROUSEL_PLAN_SYSTEM
        )

        plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", carousel_plan_system),
                ("human", CAROUSEL_PLAN_HUMAN),
            ]
        )
        plan_chain = plan_prompt | self.llm.with_structured_output(CarouselPlan)
        plan: CarouselPlan = await plan_chain.ainvoke(
            {
                "script_content": script_content,
                "refined_context": refined_context,
                "verified_claims": claims_text,
                "platform": platform or "default",
                "correction_hint": correction_hint,
            }
        )
        plan_text = plan.model_dump_json(indent=2)
        logger.info("Carousel plan produced: %d slides", len(plan.slides))

        carousel_exec_system = (
            f"{hedge_block}{CAROUSEL_FORMATTER_SYSTEM}"
            if hedge_block
            else CAROUSEL_FORMATTER_SYSTEM
        )
        exec_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", carousel_exec_system),
                ("human", CAROUSEL_FORMATTER_HUMAN),
            ]
        )
        exec_chain = exec_prompt | self.llm.with_structured_output(
            CarouselFormatterOutput
        )
        result: CarouselFormatterOutput = await exec_chain.ainvoke(
            {
                "plan": plan_text,
                "script_content": script_content,
                "refined_context": refined_context,
                "verified_claims": claims_text,
                "platform": platform or "default",
                "correction_hint": correction_hint,
            }
        )

        payload = result.model_dump()
        payload["_format"] = "carousel"
        payload["_version"] = 1
        payload["char_limit_violations"] = []

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload=payload,
            reasoning="Carousel format generated via Plan→Execute from verified script and research context.",
            confidence_score=0.9,
            metadata={
                "model": self.model_name,
                "agent": "carousel_formatter",
                "planned_slides": len(plan.slides),
                "generated_slides": len(result.slides),
            },
        )


class VideoFormatterAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        script_content = context.get("script_content", "")
        refined_context = context.get("refined_context", "")
        verified_claims = context.get("verified_claims", [])
        hedge_index = context.get("hedge_index", [])
        correction_hint = context.get("correction_hint", "")

        if not script_content:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No script content provided for video formatting.",
                confidence_score=0.0,
            )

        if not refined_context:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No refined context provided for video formatting.",
                confidence_score=0.0,
            )

        claims_text = "\n".join(
            f"- {c.get('claim_text', '')} [{c.get('verdict', 'UNKNOWN')}]: {c.get('evidence_text', 'N/A')}"
            for c in verified_claims
        )

        hedge_block = _build_hedge_block(hedge_index)
        video_plan_system = (
            f"{hedge_block}{VIDEO_PLAN_SYSTEM}" if hedge_block else VIDEO_PLAN_SYSTEM
        )

        plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", video_plan_system),
                ("human", VIDEO_PLAN_HUMAN),
            ]
        )
        plan_chain = plan_prompt | self.llm.with_structured_output(VideoPlan)
        plan: VideoPlan = await plan_chain.ainvoke(
            {
                "script_content": script_content,
                "refined_context": refined_context,
                "verified_claims": claims_text,
                "correction_hint": correction_hint,
            }
        )
        plan_text = plan.model_dump_json(indent=2)
        logger.info("Video plan produced: %d scenes", len(plan.scene_outline))

        video_exec_system = (
            f"{hedge_block}{VIDEO_FORMATTER_SYSTEM}"
            if hedge_block
            else VIDEO_FORMATTER_SYSTEM
        )
        exec_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", video_exec_system),
                ("human", VIDEO_FORMATTER_HUMAN),
            ]
        )
        exec_chain = exec_prompt | self.llm.with_structured_output(VideoFormatterOutput)
        result: VideoFormatterOutput = await exec_chain.ainvoke(
            {
                "plan": plan_text,
                "script_content": script_content,
                "refined_context": refined_context,
                "verified_claims": claims_text,
                "correction_hint": correction_hint,
            }
        )

        payload = result.model_dump()
        payload["_format"] = "video"
        payload["_version"] = 1

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload=payload,
            reasoning="Video format generated via Plan→Execute from verified script and research context.",
            confidence_score=0.9,
            metadata={
                "model": self.model_name,
                "agent": "video_formatter",
                "planned_scenes": len(plan.scene_outline),
                "generated_scenes": len(result.scenes),
            },
        )

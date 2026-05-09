import logging
from typing import Any, Dict, List

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from app.workers.agents import AgentActionStatus, AgentResult, BaseAgent
from app.schemas.formats import BlogSection, SeoMeta, CarouselSlide

logger = logging.getLogger(__name__)


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
    sections: List[BlogSection]
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
    "10. Structure: 4-8 sections with clear headings."
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
    "4. Include a visual_prompt for each slide describing what to display.\n"
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
    "Follow the plan's slide sequence and purposes. Generate the full carousel with visual prompts."
)


class BlogFormatterAgent(BaseAgent):
    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        script_content = context.get("script_content", "")
        refined_context = context.get("refined_context", "")
        verified_claims = context.get("verified_claims", [])
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

        plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", BLOG_PLAN_SYSTEM),
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

        exec_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", BLOG_FORMATTER_SYSTEM),
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

        plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CAROUSEL_PLAN_SYSTEM),
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

        exec_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CAROUSEL_FORMATTER_SYSTEM),
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

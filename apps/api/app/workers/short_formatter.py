import json
import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Set, Type

from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from app.workers.agents import AgentActionStatus, AgentResult, LLMAgent
from app.workers.formatters import _build_hedge_block, _resolve_aspect_ratio
from app.schemas.formats import ShortScene
from app.services.short_config import DEFAULT_SUBTITLE_PRESET_MAP, DEFAULT_VOICE_MAP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------


class ShortSceneOutline(BaseModel):
    scene_number: int
    is_hook: bool = False
    purpose: str
    key_visual: str
    duration_estimate: float
    suggested_asset_type: str = Field(
        default="video_clip",
        description='One of: "video_clip" or "ken_burns"',
    )


class ShortPlan(BaseModel):
    proposed_title: str
    scene_outline: List[ShortSceneOutline]
    visual_style_direction: str
    audio_direction: str
    music_mood: str
    voice_id: str
    subtitle_preset: str
    loop_hook: Optional[str] = Field(
        None,
        description="When loopable=true, a phrase that hooks back to the opener",
    )


class ShortFormatterOutput(BaseModel):
    scenes: List[ShortScene] = Field(min_length=2, max_length=12)
    target_total_duration: float = Field(..., ge=15.0, le=90.0)
    visual_style: str = Field(..., min_length=5)
    audio_direction: str = Field(...)
    music_mood: str = Field(...)
    voice_id: str = Field(...)
    subtitle_preset: str = Field(default="CENTER_POP_YELLOW")
    loop_hook: Optional[str] = Field(None)


# ---------------------------------------------------------------------------
# Prompt constants
# ---------------------------------------------------------------------------

_SHORT_PLAN_SYSTEM = (
    "You are the Short Formatter Planner at the AI Content Factory.\n"
    "Your task is to PLAN a short-form video (30–50s) scene structure from a "
    "verified narrative script.\n"
    "You do NOT produce the final scenes — you produce a structured scene outline.\n\n"
    "## PLANNING RULES\n"
    "1. Break the script into 2-8 distinct scenes with a clear narrative arc.\n"
    "2. First scene: hook — grab attention with a striking visual or provocative opening.\n"
    "3. Last scene: closer — end with a call-to-action or thought-provoking statement.\n"
    "4. Each scene needs a purpose, key visual description, duration estimate, and suggested_asset_type.\n"
    "5. Suggested asset types: 'video_clip' (AI-generated motion) or 'ken_burns' (animated still).\n"
    "6. Total target duration must be 30-50 seconds (15-90 allowed).\n"
    "7. Each scene should be 3-15 seconds.\n"
    "8. Define visual_style_direction and audio_direction for the overall short.\n"
    "9. Suggest a music_mood tag (e.g., 'dark_lofi', 'synthwave_hype', 'calm_informative').\n"
    "10. If loopable=true, design a loop_hook phrase that circles the closer back to the hook.\n\n"
    "STOP CONDITIONS — evaluate as you read the input, before producing any output:\n"
    "- If the script_content is too thin to produce a coherent short plan (e.g., no\n"
    "  scene structure, no narration beats, no duration or pacing signals), set\n"
    "  status=ERROR and describe what is missing. Do not fabricate scene breaks or\n"
    "  timing from content that does not contain them."
)

_SHORT_PLAN_HUMAN = (
    "Plan a short-form video scene outline from the following verified narrative script.\n\n"
    "<script>\n{script_content}\n</script>\n\n"
    "<refined_context>\n{refined_context}\n</refined_context>\n\n"
    "<verified_claims>\n{verified_claims}\n</verified_claims>\n\n"
    "<platform>\n{platform}\n</platform>\n\n"
    "<platform_aspect_ratio>\n{platform_aspect_ratio}\n</platform_aspect_ratio>\n\n"
    "<voice_id>\n{voice_id}\n</voice_id>\n\n"
    "<visual_style_theme>\n{visual_style_theme}\n</visual_style_theme>\n\n"
    "<correction_hint>\n{correction_hint}\n</correction_hint>\n\n"
    "Produce a structured scene outline with scene number, purpose, key visual, "
    "duration estimate, and suggested asset type per scene. "
    "Do NOT write the full narration or visual prompts."
)

_SHORT_FORMATTER_SYSTEM = (
    "You are the Short Formatter Agent at the AI Content Factory.\n"
    "You receive a verified narrative script, user reference context with editorial "
    "directives, verified claims, platform constraints, and a PLAN.\n"
    "Your task is to execute the plan — generate complete short scenes with "
    "narration, visual direction, and asset type decisions.\n\n"
    "## RULES\n"
    "1. Every factual claim MUST trace to the verified claims or refined_context.\n"
    "2. Follow the plan's scene sequence, purposes, and duration estimates.\n"
    "3. Each scene must have:\n"
    "   - narration_text: the voiceover narration (min 10 chars)\n"
    "   - visual_prompt: a detailed visual description for AI generation (min 10 chars)\n"
    "   - asset_type: 'video_clip' or 'ken_burns'\n"
    "   - kb_motion: required when asset_type='ken_burns' (one of: pan_left, pan_right, zoom_in, zoom_out, static_zoom_in)\n"
    "   - target_duration_seconds: scene length (3-15 seconds)\n"
    "4. Total target_total_duration must be 15-90 seconds.\n"
    "5. Minimum 2 scenes, maximum 12 scenes.\n"
    "6. visual_style: describe the overall visual style of the short.\n"
    "7. audio_direction: describe the overall audio/music direction.\n"
    "8. music_mood: a mood tag matching the plan (e.g., dark_lofi, synthwave_hype).\n"
    "9. voice_id: the resolved voice identifier passed in the plan.\n"
    "10. subtitle_preset: one of CENTER_POP_YELLOW, CLEAN_WHITE_LOWER, NEON_BOXED.\n"
    "11. If loopable=true, include a loop_hook that circles the closer back to the hook.\n"
    "12. Scene transitions should feel natural and maintain narrative flow.\n"
    "13. Write narration in a conversational, engaging tone — short-form pacing.\n"
    "14. Visual prompts should describe what the scene shows (setting, subjects, action, mood, colors) — not how to film it.\n"
    "15. The user selected visual_style_theme: '{visual_style_theme}'. Let this guide your visual_style_direction, pacing, and visual choices.\n"
    "16. Mark scene 1 as `is_hook: true` — this scene is the visual hook for the first 3 seconds.\n\n"
    "STOP CONDITIONS — evaluate as you read the input, before producing any output:\n"
    "- If the plan omits scene duration estimates or total short length targets,\n"
    "  state the pacing assumptions you are applying (e.g., 5s average scene,\n"
    "  35s total) before writing. Do not apply timing constraints the plan\n"
    "  did not specify without making them explicit.\n"
    "- If correction_hint requests a change that directly contradicts a structural\n"
    "  or factual constraint in the plan (not merely adds to it), set status=ESCALATE\n"
    "  and describe the specific conflict. Do not silently prioritize one over the other.\n"
    "- If the per-scene visual_prompt fields contain contradictory settings that cannot be\n"
    "  coherently described, set status=ERROR and describe the specific contradiction."
)

_SHORT_FORMATTER_HUMAN = (
    "Execute the following short plan to produce complete scenes.\n\n"
    "<plan>\n{plan}\n</plan>\n\n"
    "<script>\n{script_content}\n</script>\n\n"
    "<refined_context>\n{refined_context}\n</refined_context>\n\n"
    "<verified_claims>\n{verified_claims}\n</verified_claims>\n\n"
    "<platform>\n{platform}\n</platform>\n\n"
    "<platform_aspect_ratio>\n{platform_aspect_ratio}\n</platform_aspect_ratio>\n\n"
    "<voice_id>\n{voice_id}\n</voice_id>\n\n"
    "<loopable>\n{loopable}\n</loopable>\n\n"
    "<correction_hint>\n{correction_hint}\n</correction_hint>\n\n"
    "Follow the plan's scene structure. Generate complete scenes with narration, "
    "visual prompts, asset types, kb_motion for ken_burns scenes, and timing. "
    "Include overall visual_style, audio_direction, music_mood, voice_id, subtitle_preset, "
    "and loop_hook when loopable."
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _resolve_voice_id(directives: dict, platform: str) -> str:
    """Return user-specified voice_id or fall back to platform default."""
    user_voice = (directives.get("voice_id") or "").strip()
    if user_voice and user_voice != "default":
        return user_voice
    return DEFAULT_VOICE_MAP.get(platform.lower().strip(), "")


def _resolve_subtitle_preset(payload: dict, platform: str) -> str:
    """Return payload subtitle_preset or fall back to platform default."""
    if payload.get("subtitle_preset"):
        return payload["subtitle_preset"]
    return DEFAULT_SUBTITLE_PRESET_MAP.get(
        platform.lower().strip(), "CENTER_POP_YELLOW"
    )


def _resolve_short_aspect_ratio(platform: str) -> str:
    """Return aspect ratio for SHORT format (9:16 for TikTok/YouTube, 4:5 for Instagram)."""
    short_map = {
        "tiktok": "9:16",
        "youtube": "9:16",
        "instagram": "4:5",
    }
    return short_map.get(platform.lower().strip(), _resolve_aspect_ratio(platform))


def _extract_json_fallback(text: str) -> dict:
    """Extract the first JSON object from a text string."""
    text = text.strip()
    # Try markdown code block first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Fall back to first brace-delimited block
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in text")
    return json.loads(text[start : end + 1])


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ShortFormatterAgent(LLMAgent):
    _required_di_tools: ClassVar[List[str]] = []
    _required_llm_tools: ClassVar[List[str]] = []
    _permissions: ClassVar[Set[str]] = {"ShortFormatterAgent"}
    input_schema: ClassVar[Optional[Type[BaseModel]]] = None

    async def _plan(self, context: Dict[str, Any]) -> ShortPlan:
        """Plan step: produce a structured scene outline."""
        script_content = context["script_content"]
        refined_context = context["refined_context"]
        verified_claims = context["verified_claims"]
        hedge_index = context["hedge_index"]
        epistemic_ledger = context["epistemic_ledger"]
        correction_hint = context["correction_hint"]
        platform = context["platform"]
        voice_id = context["voice_id"]
        visual_style_theme = context["visual_style_theme"]
        platform_aspect_ratio = context["platform_aspect_ratio"]

        claims_text = "\n".join(
            f"- {c.get('claim_text', '')} [{c.get('verdict', 'UNKNOWN')}]: {c.get('evidence_text', 'N/A')}"
            for c in verified_claims
        )

        hedge_block = _build_hedge_block(hedge_index, epistemic_ledger)
        short_plan_system = (
            f"{hedge_block}{_SHORT_PLAN_SYSTEM}" if hedge_block else _SHORT_PLAN_SYSTEM
        )

        plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", short_plan_system),
                ("human", _SHORT_PLAN_HUMAN),
            ]
        )
        plan_chain = plan_prompt | self.llm.with_structured_output(ShortPlan)
        return await plan_chain.ainvoke(
            {
                "script_content": script_content,
                "refined_context": refined_context,
                "verified_claims": claims_text,
                "platform": platform or "default",
                "platform_aspect_ratio": platform_aspect_ratio,
                "voice_id": voice_id,
                "loopable": str(context.get("loopable", True)),
                "visual_style_theme": visual_style_theme or "default",
                "correction_hint": correction_hint,
            }
        )

    async def _execute_formatter(
        self, context: Dict[str, Any], plan: ShortPlan
    ) -> ShortFormatterOutput:
        """Execute step: produce complete scenes from a plan."""
        script_content = context["script_content"]
        refined_context = context["refined_context"]
        verified_claims = context["verified_claims"]
        hedge_index = context["hedge_index"]
        epistemic_ledger = context["epistemic_ledger"]
        correction_hint = context["correction_hint"]
        platform = context["platform"]
        voice_id = context["voice_id"]
        visual_style_theme = context["visual_style_theme"]
        platform_aspect_ratio = context["platform_aspect_ratio"]
        loopable = context.get("loopable", True)

        claims_text = "\n".join(
            f"- {c.get('claim_text', '')} [{c.get('verdict', 'UNKNOWN')}]: {c.get('evidence_text', 'N/A')}"
            for c in verified_claims
        )

        hedge_block = _build_hedge_block(hedge_index, epistemic_ledger)
        short_exec_system = (
            f"{hedge_block}{_SHORT_FORMATTER_SYSTEM}"
            if hedge_block
            else _SHORT_FORMATTER_SYSTEM
        )
        exec_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", short_exec_system),
                ("human", _SHORT_FORMATTER_HUMAN),
            ]
        )
        exec_chain = exec_prompt | self.llm.with_structured_output(
            ShortFormatterOutput
        )
        return await exec_chain.ainvoke(
            {
                "plan": plan.model_dump_json(indent=2),
                "script_content": script_content,
                "refined_context": refined_context,
                "verified_claims": claims_text,
                "platform": platform or "default",
                "platform_aspect_ratio": platform_aspect_ratio,
                "voice_id": voice_id,
                "loopable": str(loopable),
                "visual_style_theme": visual_style_theme or "default",
                "correction_hint": correction_hint,
            }
        )

    async def _execute_formatter_fallback(
        self, context: Dict[str, Any], plan: ShortPlan
    ) -> ShortFormatterOutput:
        """Fallback: raw LLM invocation + manual JSON extraction."""
        script_content = context["script_content"]
        refined_context = context["refined_context"]
        verified_claims = context["verified_claims"]
        hedge_index = context["hedge_index"]
        epistemic_ledger = context["epistemic_ledger"]
        correction_hint = context["correction_hint"]
        platform = context["platform"]
        voice_id = context["voice_id"]
        visual_style_theme = context["visual_style_theme"]
        platform_aspect_ratio = context["platform_aspect_ratio"]
        loopable = context.get("loopable", True)

        claims_text = "\n".join(
            f"- {c.get('claim_text', '')} [{c.get('verdict', 'UNKNOWN')}]: {c.get('evidence_text', 'N/A')}"
            for c in verified_claims
        )

        hedge_block = _build_hedge_block(hedge_index, epistemic_ledger)
        short_exec_system = (
            f"{hedge_block}{_SHORT_FORMATTER_SYSTEM}"
            if hedge_block
            else _SHORT_FORMATTER_SYSTEM
        )
        exec_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", short_exec_system),
                ("human", _SHORT_FORMATTER_HUMAN),
            ]
        )
        raw_chain = exec_prompt | self.llm
        msg = await raw_chain.ainvoke(
            {
                "plan": plan.model_dump_json(indent=2),
                "script_content": script_content,
                "refined_context": refined_context,
                "verified_claims": claims_text,
                "platform": platform or "default",
                "platform_aspect_ratio": platform_aspect_ratio,
                "voice_id": voice_id,
                "loopable": str(loopable),
                "visual_style_theme": visual_style_theme or "default",
                "correction_hint": correction_hint,
            }
        )
        json_data = _extract_json_fallback(msg.content)
        return ShortFormatterOutput.model_validate(json_data)

    async def _execute(self, context: Dict[str, Any], **kwargs) -> AgentResult:
        script_content = context.get("script_content", "")
        refined_context = context.get("refined_context", "")
        platform = context.get("platform", "")
        loopable = context.get("loopable", True)
        voice_id = context.get("voice_id", "")
        visual_style_theme = context.get("visual_style_theme", "")
        platform_aspect_ratio = _resolve_short_aspect_ratio(platform)

        if not script_content:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No script content provided for short formatting.",
                confidence_score=0.0,
            )

        if not refined_context:
            return AgentResult(
                status=AgentActionStatus.ERROR,
                payload={},
                reasoning="No refined context provided for short formatting.",
                confidence_score=0.0,
            )

        # Normalize context keys for plan/execute helpers
        _ctx = {
            "script_content": script_content,
            "refined_context": refined_context,
            "verified_claims": context.get("verified_claims", []),
            "hedge_index": context.get("hedge_index", []),
            "epistemic_ledger": context.get("epistemic_ledger", {}),
            "correction_hint": context.get("correction_hint", ""),
            "platform": platform,
            "voice_id": voice_id,
            "visual_style_theme": visual_style_theme,
            "platform_aspect_ratio": platform_aspect_ratio,
            "loopable": loopable,
        }

        # Plan step (cached in context across harness retries)
        plan_key = "_short_plan_checkpoint"
        plan = context.get(plan_key)
        if plan is None:
            try:
                plan = await self._plan(_ctx)
            except (OutputParserException, ValidationError) as exc:
                logger.warning("Short plan structured output failed: %s", exc)
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=f"Short plan structured output failed: {exc}",
                    confidence_score=0.0,
                )
            context[plan_key] = plan

        logger.info("Short plan produced: %d scenes", len(plan.scene_outline))

        # Execute step with JSON fallback
        try:
            result = await self._execute_formatter(_ctx, plan)
        except (OutputParserException, ValidationError) as exc:
            logger.warning(
                "Short formatter structured output failed, "
                "attempting JSON fallback: %s",
                exc,
            )
            try:
                result = await self._execute_formatter_fallback(_ctx, plan)
            except Exception as fb_exc:
                logger.warning("JSON fallback also failed: %s", fb_exc)
                return AgentResult(
                    status=AgentActionStatus.ERROR,
                    payload={},
                    reasoning=f"Short formatter failed: {exc}; fallback also failed: {fb_exc}",
                    confidence_score=0.0,
                )

        payload = result.model_dump()
        payload["_format"] = "short"
        payload["_version"] = 1

        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload=payload,
            reasoning="Short format generated via Plan→Execute from verified script and research context.",
            confidence_score=0.9,
            metadata={
                "model": self.model_name,
                "agent": "short_formatter",
                "planned_scenes": len(plan.scene_outline),
                "generated_scenes": len(result.scenes),
                "loopable": loopable,
                "platform": platform,
            },
        )

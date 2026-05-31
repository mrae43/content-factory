from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from pydantic import BaseModel, Field

from app.services.tools import ToolRegistry
from app.services.vector_store import make_ingest_chunks_tool
from app.services.format_validator import (
    make_validate_format_tool,
    BlogValidator,
    CarouselValidator,
    VideoValidator,
)
from app.workers.agents import ServiceAgent, AgentActionStatus, AgentResult
from app.workers.harness import AgentHarness


# ── Helpers ────────────────────────────────────────────────────────────────────


class _FakeVectorStore:
    async def ingest_chunks(self, job_id, chunks, scope="LOCAL", meta=None):
        return len(chunks)


class _ConcreteService(ServiceAgent):
    _required_di_tools = []
    _permissions = {"*"}

    async def _execute(self, context, **kwargs):
        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={},
            reasoning="ok",
            confidence_score=1.0,
        )


class _InputSchema(BaseModel):
    job_id: str = Field(description="Job identifier")
    script_content: str = Field(description="Script body")


class _FilteredService(ServiceAgent):
    _required_di_tools = []
    _permissions = {"*"}
    input_schema = _InputSchema

    async def _execute(self, context, **kwargs):
        return AgentResult(
            status=AgentActionStatus.SUCCESS,
            payload={"received_keys": list(context.keys())},
            reasoning="ok",
            confidence_score=1.0,
        )


# ── D1: ingest_chunks tool ─────────────────────────────────────────────────────


class TestIngestChunksTool:
    def test_factory_creates_tool_with_correct_name(self):
        vs = _FakeVectorStore()
        tool = make_ingest_chunks_tool(vs)
        assert tool.name == "ingest_chunks"
        assert tool.llm_schema is None

    def test_tool_callable_delegates_to_vector_store(self):
        vs = _FakeVectorStore()
        tool = make_ingest_chunks_tool(vs)
        tool.callable = AsyncMock(wraps=tool.callable)

        job_id = uuid4()
        chunks = ["chunk one", "chunk two"]

        import asyncio

        result = asyncio.run(
            tool.callable(job_id=job_id, chunks=chunks, scope="RAW-CONTEXT")
        )

        assert result == 2

    def test_permissions_include_wildcard(self):
        vs = _FakeVectorStore()
        tool = make_ingest_chunks_tool(vs)
        assert "*" in tool.permissions


# ── D2: validate_format tool ───────────────────────────────────────────────────


class TestValidateFormatTool:
    def test_factory_creates_tool_with_correct_name(self):
        validator = BlogValidator()
        tool = make_validate_format_tool(validator)
        assert tool.name == "validate_format"

    def test_blog_validator_tool_rejects_empty_payload(self):
        validator = BlogValidator()
        tool = make_validate_format_tool(validator)

        import asyncio

        result = asyncio.run(tool.callable(payload={}))
        assert result["valid"] is False
        assert "error_message" in result

    def test_carousel_validator_tool_rejects_oversized_text(self):
        validator = CarouselValidator(platform="twitter")
        tool = make_validate_format_tool(validator)

        payload = {
            "_format": "carousel",
            "slides": [
                {
                    "slide_number": 1,
                    "text": "x" * 300,
                    "visual_description": "A description",
                    "hook_type": "question",
                    "sources_used": [],
                },
            ],
            "thread_title": "Test Title",
            "hashtags": ["#test"],
            "cta_slide": "Follow",
            "char_limit_violations": [],
        }

        import asyncio

        result = asyncio.run(tool.callable(payload=payload))
        assert result["valid"] is False
        assert "char limit" in result.get("error_message", "").lower()

    def test_video_validator_tool_rejects_empty_scene(self):
        validator = VideoValidator()
        tool = make_validate_format_tool(validator)

        payload = {
            "_format": "video",
            "scenes": [
                {
                    "scene_number": 1,
                    "narration_text": "Narration for scene one that is long enough.",
                    "visual_prompt": "A visual prompt that is long enough.",
                    "audio_cue": "Tension build",
                    "duration_seconds": 30.0,
                },
                {
                    "scene_number": 2,
                    "narration_text": " " * 15,
                    "visual_prompt": "Another visual prompt long enough.",
                    "audio_cue": "Silence",
                    "duration_seconds": 30.0,
                },
                {
                    "scene_number": 3,
                    "narration_text": "Narration for scene three that is long enough.",
                    "visual_prompt": "Yet another visual prompt long enough.",
                    "audio_cue": "Climax",
                    "duration_seconds": 30.0,
                },
            ],
            "total_duration_seconds": 90.0,
            "visual_style": "Cinematic documentary",
            "audio_direction": "Orchestral",
            "unified_visual_prompt": "A cinematic documentary visual style with tension-building and climax scenes.",
        }

        import asyncio

        result = asyncio.run(tool.callable(payload=payload))
        assert result["valid"] is False
        assert "empty" in result.get("error_message", "").lower()

    def test_tool_is_di_only(self):
        validator = BlogValidator()
        tool = make_validate_format_tool(validator)
        assert tool.llm_schema is None

    def test_permissions_include_harness(self):
        validator = BlogValidator()
        tool = make_validate_format_tool(validator)
        assert "AgentHarness" in tool.permissions
        assert "*" in tool.permissions


# ── D6: harness context filtering ──────────────────────────────────────────────


class TestHarnessContextFiltering:
    def test_no_schema_passes_full_context(self):
        agent = _ConcreteService()
        harness = AgentHarness(agent=agent)

        context = {"job_id": "abc", "script_content": "hello", "extra_key": 42}
        filtered = harness._filter_context(context)
        assert filtered == context

    def test_with_schema_filters_unknown_keys(self):
        agent = _FilteredService()
        harness = AgentHarness(agent=agent)

        context = {"job_id": "abc", "script_content": "hello", "extra_key": 42}
        filtered = harness._filter_context(context)
        assert "job_id" in filtered
        assert "script_content" in filtered
        assert "extra_key" not in filtered

    def test_with_schema_missing_required_field(self):
        agent = _FilteredService()
        harness = AgentHarness(agent=agent)

        context = {"script_content": "hello"}
        filtered = harness._filter_context(context)
        assert "job_id" not in filtered
        assert "script_content" in filtered


# ── D8: end-to-end tool wiring ─────────────────────────────────────────────────


class TestToolWiring:
    def setup_method(self):
        ToolRegistry().clear()

    def test_register_and_resolve_ingest_chunks(self):
        registry = ToolRegistry()
        vs = _FakeVectorStore()
        tool = make_ingest_chunks_tool(vs)
        registry.register(tool)

        assert "ingest_chunks" in registry
        resolved = registry.get("ingest_chunks")
        assert resolved.name == "ingest_chunks"
        assert resolved is tool

    def test_register_and_resolve_validate_format(self):
        registry = ToolRegistry()
        validator = BlogValidator()
        tool = make_validate_format_tool(validator)
        registry.register(tool)

        assert "validate_format" in registry
        resolved = registry.get("validate_format")
        assert resolved.name == "validate_format"

    def test_permission_resolution_for_ingest_chunks(self):
        registry = ToolRegistry()
        vs = _FakeVectorStore()
        registry.register(make_ingest_chunks_tool(vs))

        permitted = registry.get_permitted_tools("CopywriterAgent")
        names = {t.name for t in permitted}
        assert "ingest_chunks" in names

    def test_no_llm_schema_for_new_tools(self):
        registry = ToolRegistry()
        vs = _FakeVectorStore()
        validator = BlogValidator()
        registry.register(make_ingest_chunks_tool(vs))
        registry.register(make_validate_format_tool(validator))

        llm_tools = registry.get_llm_tools("RedTeamAgent")
        llm_names = {t.name for t in llm_tools}
        assert "ingest_chunks" not in llm_names
        assert "validate_format" not in llm_names

    def test_harness_injects_tools_for_service_agent(self):
        registry = ToolRegistry()
        vs = _FakeVectorStore()
        registry.register(make_ingest_chunks_tool(vs))

        agent = _ConcreteService()
        AgentHarness(agent=agent)

        assert hasattr(agent, "di_tools")
        assert "ingest_chunks" in agent.di_tools

    def test_duplicate_registration_raises(self):
        registry = ToolRegistry()
        vs = _FakeVectorStore()
        registry.register(make_ingest_chunks_tool(vs))

        with pytest.raises(ValueError, match="already registered"):
            registry.register(make_ingest_chunks_tool(vs))

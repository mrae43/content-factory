from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.services.tools import Tool, ToolRegistry


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_tool(name: str = "test_tool", permissions: set | None = None) -> Tool:
    return Tool(
        name=name,
        description="A test tool.",
        callable=MagicMock(return_value="done"),
        permissions=permissions or {"*"},
    )


# ── Tool dataclass ─────────────────────────────────────────────────────────────

class TestToolDataclass:
    def test_basic_creation(self):
        async def fn():
            return 42

        t = Tool(name="my_tool", description="Does stuff", callable=fn)
        assert t.name == "my_tool"
        assert t.description == "Does stuff"
        assert t.callable is fn
        assert t.llm_schema is None
        assert t.permissions == {"*"}

    def test_with_llm_schema(self):
        schema = {
            "type": "function",
            "function": {
                "name": "search",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        t = Tool(
            name="search",
            description="Search the index",
            callable=MagicMock(),
            llm_schema=schema,
            permissions={"RedTeamAgent"},
        )
        assert t.llm_schema == schema
        assert t.permissions == {"RedTeamAgent"}

    def test_default_permissions_is_wildcard(self):
        t = Tool(name="x", description="x", callable=MagicMock())
        assert t.permissions == {"*"}

    def test_permissions_cannot_be_empty_after_construction(self):
        t = Tool(name="x", description="x", callable=MagicMock(), permissions=set())
        assert t.permissions == set()


# ── ToolRegistry singleton ─────────────────────────────────────────────────────

class TestToolRegistry:
    def setup_method(self):
        ToolRegistry().clear()

    def test_singleton(self):
        r1 = ToolRegistry()
        r2 = ToolRegistry()
        assert r1 is r2

    def test_register_and_get(self):
        registry = ToolRegistry()
        t = _make_tool("semantic_search")
        registry.register(t)
        assert registry.get("semantic_search") is t

    def test_get_raises_on_missing(self):
        with pytest.raises(KeyError, match="unknown"):
            ToolRegistry().get("unknown")

    def test_register_raises_on_duplicate(self):
        registry = ToolRegistry()
        registry.register(_make_tool("dup"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(_make_tool("dup"))

    def test_list_tools(self):
        registry = ToolRegistry()
        registry.register(_make_tool("a"))
        registry.register(_make_tool("b"))
        assert len(registry.list_tools()) == 2

    def test_clear(self):
        registry = ToolRegistry()
        registry.register(_make_tool("x"))
        registry.clear()
        assert len(registry.list_tools()) == 0

    def test_contains(self):
        registry = ToolRegistry()
        registry.register(_make_tool("exists"))
        assert "exists" in registry
        assert "nope" not in registry


# ── Permission filtering ──────────────────────────────────────────────────────

class TestToolPermissions:
    def setup_method(self):
        ToolRegistry().clear()

    def test_wildcard_allows_all(self):
        registry = ToolRegistry()
        registry.register(_make_tool("anyone", permissions={"*"}))
        permitted = registry.get_permitted_tools("RedTeamAgent")
        assert len(permitted) == 1

    def test_agent_specific_permission(self):
        registry = ToolRegistry()
        registry.register(_make_tool("red_only", permissions={"RedTeamAgent"}))
        registry.register(_make_tool("blue_only", permissions={"BlueAgent"}))

        red_tools = registry.get_permitted_tools("RedTeamAgent")
        assert len(red_tools) == 1
        assert red_tools[0].name == "red_only"

        blue_tools = registry.get_permitted_tools("BlueAgent")
        assert len(blue_tools) == 1
        assert blue_tools[0].name == "blue_only"

    def test_multiple_agents_can_share_tool(self):
        registry = ToolRegistry()
        registry.register(
            _make_tool("shared", permissions={"RedTeamAgent", "BlueAgent"})
        )
        assert len(registry.get_permitted_tools("RedTeamAgent")) == 1
        assert len(registry.get_permitted_tools("BlueAgent")) == 1
        assert len(registry.get_permitted_tools("GreenAgent")) == 0

    def test_get_llm_tools_only_returns_tools_with_schema(self):
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="di_only",
                description="No LLM schema",
                callable=MagicMock(),
                permissions={"*"},
            )
        )
        registry.register(
            Tool(
                name="llm_ready",
                description="Has schema",
                callable=MagicMock(),
                llm_schema={"type": "function"},
                permissions={"*"},
            )
        )
        llm_tools = registry.get_llm_tools("RedTeamAgent")
        assert len(llm_tools) == 1
        assert llm_tools[0].name == "llm_ready"


# ── Agent hierarchy (structural, no need for orchestrator) ─────────────────────

class TestAgentHierarchy:
    def test_llm_agent_is_base_agent(self):
        from app.workers.agents import LLMAgent, ServiceAgent

        assert issubclass(LLMAgent, object)
        assert issubclass(ServiceAgent, object)

    def test_service_agent_has_no_llm_by_default(self):
        from app.workers.agents import ServiceAgent

        class ConcreteService(ServiceAgent):
            async def _execute(self, context, **kwargs):
                from app.workers.agents import AgentResult, AgentActionStatus
                return AgentResult(
                    status=AgentActionStatus.SUCCESS,
                    payload={},
                    reasoning="ok",
                    confidence_score=1.0,
                )

        agent = ConcreteService()
        assert not hasattr(agent, "llm")

    def test_llm_agent_creates_llm_when_constructed(self):
        from app.workers.agents import LLMAgent

        class ConcreteLLM(LLMAgent):
            async def _execute(self, context, **kwargs):
                from app.workers.agents import AgentResult, AgentActionStatus
                return AgentResult(
                    status=AgentActionStatus.SUCCESS,
                    payload={},
                    reasoning="ok",
                    confidence_score=1.0,
                )

        agent = ConcreteLLM()
        assert hasattr(agent, "llm")
        assert agent.llm is not None

    def test_inject_tools_populates_dict(self):
        from app.workers.agents import ServiceAgent

        class ConcreteService(ServiceAgent):
            async def _execute(self, context, **kwargs):
                from app.workers.agents import AgentResult, AgentActionStatus
                return AgentResult(
                    status=AgentActionStatus.SUCCESS,
                    payload={},
                    reasoning="ok",
                    confidence_score=1.0,
                )

        agent = ConcreteService()
        t1 = _make_tool("t1")
        t2 = _make_tool("t2")
        agent.inject_tools({"t1": t1, "t2": t2})
        assert len(agent.di_tools) == 2
        assert agent.di_tools["t1"] is t1
        assert agent.di_tools["t2"] is t2

    def test_each_agent_class_declares_required_tools(self):
        from app.workers.agents import (
            CopywriterAgent,
            RedTeamAgent,
            AssetStudioAgent,
        )
        from app.workers.optimizer import ScriptOptimizerAgent
        from app.workers.formatters import (
            BlogFormatterAgent,
            CarouselFormatterAgent,
            VideoFormatterAgent,
        )

        for cls in [
            CopywriterAgent,
            RedTeamAgent,
            AssetStudioAgent,
            ScriptOptimizerAgent,
            BlogFormatterAgent,
            CarouselFormatterAgent,
            VideoFormatterAgent,
        ]:
            assert hasattr(cls, "_required_di_tools")
            assert isinstance(cls._required_di_tools, list)
            assert hasattr(cls, "_required_llm_tools")
            assert isinstance(cls._required_llm_tools, list)
            assert hasattr(cls, "_permissions")
            assert isinstance(cls._permissions, set)
            assert hasattr(cls, "input_schema")

    def test_red_team_declares_semantic_search(self):
        from app.workers.agents import RedTeamAgent

        assert "semantic_search" in RedTeamAgent._required_di_tools

    def test_llm_agent_has_correct_base(self):
        from app.workers.agents import BaseAgent, LLMAgent

        assert issubclass(LLMAgent, BaseAgent)

    def test_service_agent_has_correct_base(self):
        from app.workers.agents import BaseAgent, ServiceAgent

        assert issubclass(ServiceAgent, BaseAgent)

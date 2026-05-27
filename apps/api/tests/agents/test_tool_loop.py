import pytest
from unittest.mock import AsyncMock, MagicMock

from app.workers.agents import LLMAgent, AgentActionStatus, RedTeamAgent
from app.services.tools import Tool


class _ConcreteToolAgent(LLMAgent):
    _required_di_tools = []
    _required_llm_tools = []
    _permissions = {"*"}
    input_schema = None

    async def _execute(self, context, **kwargs):
        return MagicMock()


def _make_agent():
    agent = _ConcreteToolAgent.__new__(_ConcreteToolAgent)
    agent.model_name = "gemini-1.5-pro"
    agent.temperature = 0.1
    agent.llm = MagicMock()
    agent.di_tools = {}
    return agent


# ── _run_tool_loop ─────────────────────────────────────────────────────────────


@pytest.mark.agent
async def test_tool_loop_falls_back_to_plain_llm_when_no_tools_bound():
    agent = _make_agent()

    class _FakeResponse:
        content = "Direct response"
        tool_calls = None

    agent.llm.ainvoke = AsyncMock(return_value=_FakeResponse())

    result = await agent._run_tool_loop(
        system_prompt="Be helpful",
        human_content="Hello",
    )

    assert result == "Direct response"
    agent.llm.ainvoke.assert_awaited_once()


@pytest.mark.agent
async def test_tool_loop_resolves_single_tool_call():
    agent = _make_agent()
    mock_tool = MagicMock()
    mock_tool.callable = AsyncMock(return_value={"results": [{"url": "https://example.com", "content": "test"}]})
    agent.llm_tools = {"execute_web_search": mock_tool}

    from langchain_core.messages import AIMessage

    first = AIMessage(
        content="",
        tool_calls=[{"name": "execute_web_search", "args": {"query": "BRICS GDP"}, "id": "call_1"}],
    )
    second = AIMessage(content="Found GDP data: 3.2%")

    agent.llm_with_tools = AsyncMock()
    agent.llm_with_tools.ainvoke = AsyncMock(side_effect=[first, second])

    result = await agent._run_tool_loop(
        system_prompt="Research assistant",
        human_content="Search for BRICS GDP",
        max_rounds=5,
    )

    assert result == "Found GDP data: 3.2%"
    assert agent.llm_with_tools.ainvoke.await_count == 2
    mock_tool.callable.assert_awaited_once_with(query="BRICS GDP")


@pytest.mark.agent
async def test_tool_loop_handles_multiple_tool_calls():
    agent = _make_agent()
    mock_tool = MagicMock()
    mock_tool.callable = AsyncMock(return_value={"results": []})

    agent.llm_tools = {"execute_web_search": mock_tool}

    from langchain_core.messages import AIMessage

    first = AIMessage(
        content="",
        tool_calls=[
            {"name": "execute_web_search", "args": {"query": "Q1"}, "id": "call_1"},
            {"name": "execute_web_search", "args": {"query": "Q2"}, "id": "call_2"},
        ],
    )
    second = AIMessage(content="Synthesized results from both queries")

    agent.llm_with_tools = AsyncMock()
    agent.llm_with_tools.ainvoke = AsyncMock(side_effect=[first, second])

    result = await agent._run_tool_loop(
        system_prompt="Research",
        human_content="Search Q1 and Q2",
        max_rounds=5,
    )

    assert result == "Synthesized results from both queries"
    assert agent.llm_with_tools.ainvoke.await_count == 2
    assert mock_tool.callable.await_count == 2


@pytest.mark.agent
async def test_tool_loop_handles_unknown_tool_gracefully():
    agent = _make_agent()
    agent.llm_tools = {}

    from langchain_core.messages import AIMessage

    first = AIMessage(
        content="",
        tool_calls=[{"name": "nonexistent_tool", "args": {"x": 1}, "id": "call_1"}],
    )
    second = AIMessage(content="Recovered from unknown tool")

    agent.llm_with_tools = AsyncMock()
    agent.llm_with_tools.ainvoke = AsyncMock(side_effect=[first, second])

    result = await agent._run_tool_loop(
        system_prompt="Research",
        human_content="Try a tool",
        max_rounds=5,
    )

    assert result == "Recovered from unknown tool"


@pytest.mark.agent
async def test_tool_loop_handles_tool_exception():
    agent = _make_agent()
    mock_tool = MagicMock()
    mock_tool.callable = AsyncMock(side_effect=ValueError("API failed"))

    agent.llm_tools = {"execute_web_search": mock_tool}

    from langchain_core.messages import AIMessage

    first = AIMessage(
        content="",
        tool_calls=[{"name": "execute_web_search", "args": {"query": "test"}, "id": "call_1"}],
    )
    second = AIMessage(content="Tool errored, continuing without data")

    agent.llm_with_tools = AsyncMock()
    agent.llm_with_tools.ainvoke = AsyncMock(side_effect=[first, second])

    result = await agent._run_tool_loop(
        system_prompt="Research",
        human_content="Search test",
        max_rounds=5,
    )

    assert result == "Tool errored, continuing without data"
    mock_tool.callable.assert_awaited_once_with(query="test")


@pytest.mark.agent
async def test_tool_loop_respects_max_rounds():
    agent = _make_agent()
    mock_tool = MagicMock()
    mock_tool.callable = AsyncMock(return_value={"data": "more info"})
    agent.llm_tools = {"execute_web_search": mock_tool}

    from langchain_core.messages import AIMessage

    # Always returns tool calls — should stop after max_rounds
    tool_response = AIMessage(
        content="",
        tool_calls=[{"name": "execute_web_search", "args": {"query": "again"}, "id": "call_x"}],
    )

    agent.llm_with_tools = AsyncMock()
    agent.llm_with_tools.ainvoke = AsyncMock(return_value=tool_response)

    await agent._run_tool_loop(
        system_prompt="Research",
        human_content="Search repeatedly",
        max_rounds=2,
    )

    assert agent.llm_with_tools.ainvoke.await_count == 2 + 1  # initial + 2 rounds
    assert mock_tool.callable.await_count == 2


# ── RedTeamAgent with tool-backed DI ───────────────────────────────────────────


def _inject_search_tool(agent, mock_vector_store):
    from uuid import UUID

    async def _search(
        query: str,
        job_id: str | None = None,
        scope: str | None = None,
        scopes: list[str] | None = None,
        top_k: int = 5,
        similarity_threshold: float | None = None,
    ):
        return await mock_vector_store.semantic_search(
            query=query,
            job_id=UUID(job_id) if job_id else None,
            scope=scope,
            scopes=scopes,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )

    agent.di_tools["semantic_search"] = Tool(
        name="semantic_search",
        description="Mock",
        callable=_search,
    )


@pytest.mark.agent
async def test_red_team_uses_di_tools_for_pass2(
    mock_vector_store,
    job_id,
    claim_extraction_single,
    red_team_verdict_supported,
    multi_chain_mock,
):
    mock_vector_store.semantic_search.return_value = [
        {"content": "IMF confirms BRICS GDP grew 3.2% in 2024.", "similarity_score": 0.95},
    ]
    agent = RedTeamAgent.__new__(RedTeamAgent)
    agent.model_name = "gemini-1.5-pro"
    agent.temperature = 0.1
    agent.llm = MagicMock()
    agent.di_tools = {}
    _inject_search_tool(agent, mock_vector_store)

    context = {
        "script_content": "BRICS GDP grew 3.2% in 2024.",
        "job_id": job_id,
    }

    with multi_chain_mock([claim_extraction_single, red_team_verdict_supported]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.SUCCESS
    mock_vector_store.semantic_search.assert_awaited_once()


@pytest.mark.agent
async def test_red_team_escalates_when_no_di_tool(
    job_id,
    claim_extraction_single,
    multi_chain_mock,
):
    agent = RedTeamAgent.__new__(RedTeamAgent)
    agent.model_name = "gemini-1.5-pro"
    agent.temperature = 0.1
    agent.llm = MagicMock()
    agent.di_tools = {}

    context = {
        "script_content": "BRICS GDP grew 3.2% in 2024.",
        "job_id": job_id,
    }

    with multi_chain_mock([claim_extraction_single]):
        result = await agent._execute(context)

    assert result.status == AgentActionStatus.ESCALATE
    assert "No research sources" in result.reasoning

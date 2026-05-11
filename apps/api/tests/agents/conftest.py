import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from app.workers.agents import (
    ResearchSchema,
    CopywriterSchema,
    RedTeamVerdict,
    ClaimItem,
    ClaimExtractionResult,
    ExtractedClaim,
    StudioPromptSchema,
)
from app.workers.optimizer import OptimizerOutput


@pytest.fixture
def research_schema_output():
    return ResearchSchema(
        chunks=[
            "BRICS GDP grew 3.2% in 2024.",
            "New payment system announced Q2.",
        ],
        refined_context="BRICS collective GDP grew 3.2% in 2024 according to IMF data. A new payment system was announced in Q2 2025 by the New Development Bank.",
        reasoning="Selected chunks with verified economic data.",
        confidence=0.85,
    )


@pytest.fixture
def copywriter_schema_output():
    return CopywriterSchema(
        script_content="Did you know BRICS is reshaping global finance?",
        reasoning="Opened with question, built through data-driven narrative.",
        confidence=0.8,
    )


@pytest.fixture
def red_team_verdict_supported():
    return RedTeamVerdict(
        claims=[
            ClaimItem(
                claim_text="BRICS GDP grew 3.2% in 2024.",
                verdict="SUPPORTED",
                confidence=0.9,
                evidence_text="IMF data confirms BRICS collective GDP growth of 3.2%.",
            ),
        ],
        overall_reasoning="All claims are supported by research sources.",
    )


@pytest.fixture
def red_team_verdict_unsupported():
    return RedTeamVerdict(
        claims=[
            ClaimItem(
                claim_text="BRICS GDP grew 15% last year.",
                verdict="UNSUPPORTED",
                confidence=0.3,
                evidence_text="No source confirms 15% GDP growth.",
            ),
            ClaimItem(
                claim_text="New payment system launched.",
                verdict="SUPPORTED",
                confidence=0.85,
                evidence_text="Announced Q2 2025.",
            ),
        ],
        overall_reasoning="1 of 2 claims unsupported. GDP figure is fabricated.",
    )


@pytest.fixture
def red_team_verdict_contested():
    return RedTeamVerdict(
        claims=[
            ClaimItem(
                claim_text="BRICS controls 40% of global trade.",
                verdict="CONTESTED",
                confidence=0.5,
                evidence_text="Sources disagree: IMF says 32%, BRICS says 38%.",
            ),
        ],
        overall_reasoning="Claim is contested \u2014 sources contradict.",
    )


@pytest.fixture
def claim_extraction_single():
    return ClaimExtractionResult(
        claims=[
            ExtractedClaim(
                claim_text="BRICS GDP grew 3.2% in 2024.",
                claim_category="statistic",
                search_query="BRICS GDP growth rate 2024",
            ),
        ]
    )


@pytest.fixture
def claim_extraction_double():
    return ClaimExtractionResult(
        claims=[
            ExtractedClaim(
                claim_text="BRICS GDP grew 15% last year.",
                claim_category="statistic",
                search_query="BRICS GDP growth 15 percent",
            ),
            ExtractedClaim(
                claim_text="New payment system launched.",
                claim_category="attribution",
                search_query="BRICS new payment system launch",
            ),
        ]
    )


@pytest.fixture
def claim_extraction_single_contested():
    return ClaimExtractionResult(
        claims=[
            ExtractedClaim(
                claim_text="BRICS controls 40% of global trade.",
                claim_category="statistic",
                search_query="BRICS share of global trade percentage",
            ),
        ]
    )


@pytest.fixture
def studio_prompt_schema_output():
    return StudioPromptSchema(
        visual_prompts=[
            "World map animation with BRICS highlight",
            "Currency overlay",
        ],
        audio_prompts="Tension-building orchestral with electronic undertones",
    )


@pytest.fixture
def optimizer_schema_output():
    return OptimizerOutput(
        patched_script_content="BRICS GDP grew 3.2% in 2024. This is a significant shift.",
        patch_summary="Replaced fabricated 15% GDP claim with verified 3.2% from IMF data.",
        reasoning="Claim 'BRICS GDP grew 15%' was UNSUPPORTED. Replaced with 3.2% from refined_context.",
        confidence=0.85,
    )


@pytest.fixture
def research_context(mock_vector_store, job_id):
    return {
        "topic": "BRICS De-dollarization 2025",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }


@pytest.fixture
def copywriter_context():
    return {
        "topic": "BRICS De-dollarization 2025",
        "feedback": "",
        "refined_context": "BRICS collective GDP grew 3.2% in 2024. A new payment system was announced in Q2.",
    }


@pytest.fixture
def red_team_context(mock_vector_store, job_id):
    return {
        "script_content": "BRICS GDP grew 15% last year. New payment system launched.",
        "vector_store": mock_vector_store,
        "job_id": job_id,
    }


@pytest.fixture
def asset_studio_context(job_id):
    return {
        "script_content": "Did you know BRICS is reshaping global finance?",
        "scenes": [
            {
                "scene_number": 1,
                "narration_text": "Did you know BRICS is reshaping global finance?",
                "visual_prompt": "World map with BRICS nations highlighted",
                "audio_cue": "Tension build",
                "duration_seconds": 15,
            }
        ],
        "visual_style": "Cinematic documentary, golden hour lighting",
        "job_id": job_id,
    }


@pytest.fixture
def make_mock_llm_chain():
    def _make(schema_instance):
        mock_llm = MagicMock()
        mock_chain = AsyncMock()
        mock_chain.ainvoke = AsyncMock(return_value=schema_instance)
        mock_llm.with_structured_output.return_value = mock_chain
        return mock_llm

    return _make


@pytest.fixture
def chain_mock():
    """
    Returns a context manager that patches RunnableSequence.ainvoke to return
    the given schema instance. Needed because prompt | llm creates a
    RunnableSequence whose ainvoke must be controlled.
    """

    def _make(schema_instance):
        return patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
            return_value=schema_instance,
        )

    return _make


@pytest.fixture
def multi_chain_mock():
    """
    Returns a context manager that patches RunnableSequence.ainvoke to return
    different values for sequential calls. Use for multi-pass LLM agents
    (e.g. Red Team: extraction then evaluation).
    """

    def _make(schema_instances):
        return patch(
            "langchain_core.runnables.base.RunnableSequence.ainvoke",
            new_callable=AsyncMock,
            side_effect=schema_instances,
        )

    return _make

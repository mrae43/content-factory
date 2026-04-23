"""
Research Agent Outcome Evals — LLM-as-Judge scored against RESEARCH_RUBRIC.

Case coverage (14): H-001..H-004, R-001..R-004, F-001, F-002, M-001..M-004

Special handling:
- F-002 (empty retrieval): expects FAILED — agent returns ERROR on empty search
- M-001 (empty pre_context): raw_text="" — agent returns ERROR
"""

import pytest

from app.workers.agents import AgentActionStatus
from tests.evals.assertions import (
    assert_must_avoid,
    assert_must_include,
    assert_word_count_in_range,
    build_case_aware_vector_store,
)
from tests.evals.judge import judge_score
from tests.evals.schemas import GoldenCase

RESEARCH_CASE_IDS = [
    "H-001",
    "H-002",
    "H-003",
    "H-004",
    "R-001",
    "R-002",
    "R-003",
    "R-004",
    "F-001",
    "F-002",
    "M-001",
    "M-002",
    "M-003",
    "M-004",
]

ERROR_CASE_IDS = {"F-002", "M-001"}


@pytest.mark.eval
@pytest.mark.parametrize("golden_case", RESEARCH_CASE_IDS, indirect=True)
async def test_research_outcome(
    golden_case: GoldenCase,
    eval_runner,
    judge_llm,
    score_aggregator,
    baseline_recorder,
):
    case_vs = build_case_aware_vector_store(golden_case)
    result = await eval_runner.run_research(golden_case, vector_store=case_vs)

    if golden_case.id in ERROR_CASE_IDS:
        assert result.status == AgentActionStatus.ERROR, (
            f"Expected ERROR for {golden_case.id}, got {result.status}"
        )
        return

    research_spec = golden_case.expected_outcomes.research
    if research_spec is None:
        pytest.skip(f"No research outcome spec for {golden_case.id}")

    assert result.status == AgentActionStatus.SUCCESS, (
        f"Expected SUCCESS, got {result.status}: {result.reasoning}"
    )

    refined_context = result.payload.get("refined_context", "")

    assert len(refined_context.split()) >= 50, (
        f"refined_context too short: {len(refined_context.split())} words"
    )

    if research_spec.must_include_facts:
        assert_must_include(refined_context, research_spec.must_include_facts)

    if research_spec.must_avoid:
        assert_must_avoid(refined_context, research_spec.must_avoid)

    word_range = research_spec.refined_context_word_range
    if word_range and word_range != (800, 1500):
        assert_word_count_in_range(refined_context, word_range)

    assert result.confidence_score >= research_spec.min_confidence, (
        f"Confidence {result.confidence_score} below minimum {research_spec.min_confidence}"
    )

    scores = await judge_score(
        judge_llm,
        "research",
        golden_case.input,
        result.payload,
        research_spec,
    )

    threshold = golden_case.scoring.pass_threshold if golden_case.scoring else 0.75
    assert scores.weighted_average >= threshold, (
        f"{golden_case.id}: judge score {scores.weighted_average:.3f} "
        f"below threshold {threshold}"
    )

    if golden_case.scoring and golden_case.scoring.dimension_thresholds:
        for dim_name, dim_threshold in golden_case.scoring.dimension_thresholds.items():
            dim_score = next(
                (d.score for d in scores.dimensions if d.dimension == dim_name),
                None,
            )
            if dim_score is not None:
                assert dim_score >= dim_threshold, (
                    f"{golden_case.id}/{dim_name}: score {dim_score} "
                    f"below threshold {dim_threshold}"
                )

    dim_scores = {d.dimension: d.score for d in scores.dimensions}
    score_aggregator.record("research", dim_scores)

    baseline_recorder.record_case_score(
        golden_case.id,
        "research",
        {
            "weighted_average": scores.weighted_average,
            "dimensions": dim_scores,
            "reasoning": scores.reasoning,
        },
    )

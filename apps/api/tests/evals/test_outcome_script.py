"""
Script (Copywriter) Agent Outcome Evals — scored against SCRIPT_RUBRIC.

Case coverage (6): H-001..H-004, R-003, M-004

Modes:
  - Golden mode (default): Uses reference_outputs.research for refined_context input,
    scores reference_outputs.script deterministically.
  - Live mode (--live): Chains ResearchAgent -> CopywriterAgent with real LLM calls.
"""

import pytest

from app.workers.agents import AgentActionStatus
from tests.evals.assertions import (
    assert_has_hook,
    assert_has_loop,
    assert_must_avoid,
    assert_must_include,
    assert_scene_count_in_range,
    assert_storyboard_fields,
    assert_word_count_in_range,
    build_case_aware_vector_store,
)
from tests.evals.judge import judge_score
from tests.evals.schemas import GoldenCase

SCRIPT_CASE_IDS = [
    "H-001",
    "H-002",
    "H-003",
    "H-004",
    "R-003",
    "M-004",
]


def _get_refined_context(case: GoldenCase, research_result) -> str:
    if (
        case.reference_outputs
        and case.reference_outputs.research
        and case.reference_outputs.research.refined_context
    ):
        return case.reference_outputs.research.refined_context
    return research_result.payload.get("refined_context", "")


@pytest.mark.eval
@pytest.mark.parametrize("golden_case", SCRIPT_CASE_IDS, indirect=True)
async def test_script_outcome(
    golden_case: GoldenCase,
    eval_runner,
    judge_llm,
    score_aggregator,
    baseline_recorder,
):
    case_vs = build_case_aware_vector_store(golden_case)
    research_result = await eval_runner.run_research(golden_case, vector_store=case_vs)

    assert research_result.status == AgentActionStatus.SUCCESS, (
        f"Research failed for {golden_case.id}: {research_result.reasoning}"
    )

    refined_context = _get_refined_context(golden_case, research_result)

    script_spec = golden_case.expected_outcomes.script
    if script_spec is None:
        pytest.skip(f"No script outcome spec for {golden_case.id}")

    feedback = ""
    script_result = await eval_runner.run_copywriter(
        golden_case, refined_context=refined_context, feedback=feedback
    )

    assert script_result.status == AgentActionStatus.SUCCESS, (
        f"Copywriter failed for {golden_case.id}: {script_result.reasoning}"
    )

    script_content = script_result.payload.get("script_content", "")
    storyboard = script_result.payload.get("storyboard", [])

    scores = None
    try:
        assert len(script_content) > 0, "script_content is empty"

        if script_spec.must_have_hook:
            assert_has_hook(script_content)

        if script_spec.must_have_loop:
            assert_has_loop(script_content)

        if script_spec.must_include_topics:
            assert_must_include(script_content, script_spec.must_include_topics)

        if script_spec.must_avoid:
            assert_must_avoid(script_content, script_spec.must_avoid)

        assert_word_count_in_range(script_content, script_spec.word_count_range)

        if storyboard:
            assert_scene_count_in_range(storyboard, script_spec.scene_count_range)
            assert_storyboard_fields(storyboard, script_spec.storyboard_fields)

        scores = await judge_score(
            judge_llm,
            "script",
            {
                "topic": golden_case.input.topic,
                "refined_context": refined_context,
            },
            script_result.payload,
            script_spec,
        )

        threshold = golden_case.scoring.pass_threshold if golden_case.scoring else 0.75
        assert scores.weighted_average >= threshold, (
            f"{golden_case.id}: judge score {scores.weighted_average:.3f} "
            f"below threshold {threshold}"
        )

        if golden_case.scoring and golden_case.scoring.dimension_thresholds:
            for (
                dim_name,
                dim_threshold,
            ) in golden_case.scoring.dimension_thresholds.items():
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
        score_aggregator.record("script", dim_scores)
    finally:
        if scores is not None:
            dim_scores = {d.dimension: d.score for d in scores.dimensions}
            baseline_recorder.record_case_score(
                golden_case.id,
                "script",
                {
                    "weighted_average": scores.weighted_average,
                    "dimensions": dim_scores,
                    "reasoning": scores.reasoning,
                },
            )

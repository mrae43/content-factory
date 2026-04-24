"""
Research Agent Outcome Evals — LLM-as-Judge scored against RESEARCH_RUBRIC.

Case coverage (14): H-001..H-004, R-001..R-004, F-001, F-002, M-001..M-004

Modes:
  - Golden mode (default): Scores pre-recorded reference_outputs deterministically.
  - Live mode (--live): Runs real ResearchAgent, scores live output.

Error cases (F-002, M-001) derived from golden case data automatically.
"""

import pytest

from app.workers.agents import AgentActionStatus
from tests.evals.assertions import (
    assert_must_avoid,
    assert_must_include,
    assert_word_count_in_range,
    build_case_aware_vector_store,
)
from tests.evals.assertions import _empty_vector_store
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


def _is_error_case(case: GoldenCase) -> bool:
    return (
        case.expected_outcomes.research is None
        or not (case.input.pre_context.raw_text or "").strip()
    )


@pytest.mark.eval
@pytest.mark.parametrize("golden_case", RESEARCH_CASE_IDS, indirect=True)
async def test_research_outcome(
    golden_case: GoldenCase,
    eval_runner,
    judge_llm,
    score_aggregator,
    baseline_recorder,
):
    is_error = _is_error_case(golden_case)

    if is_error:
        case_vs = _empty_vector_store()
    else:
        case_vs = build_case_aware_vector_store(golden_case)
    result = await eval_runner.run_research(golden_case, vector_store=case_vs)

    if is_error:
        assert result.status == AgentActionStatus.ERROR, (
            f"Expected ERROR for {golden_case.id}, got {result.status}: {result.reasoning}"
        )
        return

    research_spec = golden_case.expected_outcomes.research
    if research_spec is None:
        pytest.skip(f"No research outcome spec for {golden_case.id}")

    assert result.status == AgentActionStatus.SUCCESS, (
        f"Expected SUCCESS, got {result.status}: {result.reasoning}"
    )

    scores = None
    assertion_errors = []
    try:
        refined_context = result.payload.get("refined_context", "")

        if len(refined_context.split()) < 50:
            assertion_errors.append(
                f"refined_context too short: {len(refined_context.split())} words"
            )

        if research_spec.must_include_facts:
            try:
                assert_must_include(refined_context, research_spec.must_include_facts)
            except AssertionError as e:
                assertion_errors.append(str(e))

        if research_spec.must_avoid:
            try:
                assert_must_avoid(refined_context, research_spec.must_avoid)
            except AssertionError as e:
                assertion_errors.append(str(e))

        word_range = research_spec.refined_context_word_range
        if word_range:
            try:
                assert_word_count_in_range(refined_context, word_range)
            except AssertionError as e:
                assertion_errors.append(str(e))

        if result.confidence_score < research_spec.min_confidence:
            assertion_errors.append(
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
        if scores.weighted_average < threshold:
            assertion_errors.append(
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
                if dim_score is not None and dim_score < dim_threshold:
                    assertion_errors.append(
                        f"{golden_case.id}/{dim_name}: score {dim_score} "
                        f"below threshold {dim_threshold}"
                    )

        if scores:
            dim_scores = {d.dimension: d.score for d in scores.dimensions}
            score_aggregator.record("research", dim_scores)
    finally:
        if scores is not None:
            dim_scores = {d.dimension: d.score for d in scores.dimensions}
            baseline_recorder.record_case_score(
                golden_case.id,
                "research",
                {
                    "weighted_average": scores.weighted_average,
                    "dimensions": dim_scores,
                    "reasoning": scores.reasoning,
                    "assertion_warnings": assertion_errors,
                },
            )
        else:
            baseline_recorder.record_case_score(
                golden_case.id,
                "research",
                {
                    "weighted_average": 0.0,
                    "dimensions": {},
                    "assertion_errors": assertion_errors,
                    "status": str(result.status.value),
                    "word_count": len(refined_context.split())
                    if refined_context
                    else 0,
                },
            )

    if assertion_errors:
        pytest.fail(
            f"{golden_case.id}: {len(assertion_errors)} assertion(s) failed:\n"
            + "\n".join(f"  - {e}" for e in assertion_errors)
        )

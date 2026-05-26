"""
Red Team (Fact-Check) Agent Outcome Evals — scored against FACT_CHECK_RUBRIC.

Case coverage (10): H-001..H-004, R-001, R-002, R-004, E-001, F-003, F-004

Modes:
  - Golden mode (default): Uses reference_outputs for research/script/fact_check
    inputs, scores deterministically.
  - Live mode (--live): Chains CopywriterAgent -> RedTeamAgent.

N-* cases excluded (no expected_outcomes.fact_check — null).
E-002, E-003 excluded (expect ESCALATE/parse failure, no claims to score).
M-* excluded (minimal input, no meaningful fact-check).
"""

import pytest

from app.workers.agents import AgentActionStatus
from tests.evals.assertions import (
    assert_claim_count_ge,
    assert_verdict_counts,
    build_case_aware_vector_store,
)
from tests.evals.judge import judge_score
from tests.evals.schemas import GoldenCase

FACTCHECK_CASE_IDS = [
    "H-001",
    "H-002",
    "H-003",
    "H-004",
    "R-001",
    "R-002",
    "R-004",
    "E-001",
    "F-003",
    "F-004",
]

ESCALATE_CASE_IDS = {"F-003"}


def _get_refined_context(case: GoldenCase) -> str:
    if (
        case.reference_outputs
        and case.reference_outputs.research
        and case.reference_outputs.research.refined_context
    ):
        return case.reference_outputs.research.refined_context
    return ""


def _get_script_content(case: GoldenCase, script_result) -> str:
    if (
        case.reference_outputs
        and case.reference_outputs.script
        and case.reference_outputs.script.script_content
    ):
        return case.reference_outputs.script.script_content
    return script_result.payload.get("script_content", "")


@pytest.mark.eval
@pytest.mark.parametrize("golden_case", FACTCHECK_CASE_IDS, indirect=True)
async def test_factcheck_outcome(
    golden_case: GoldenCase,
    eval_runner,
    judge_llm,
    score_aggregator,
    baseline_recorder,
):
    case_vs = build_case_aware_vector_store(golden_case)
    refined_context = _get_refined_context(golden_case)
    script_result = await eval_runner.run_copywriter(
        golden_case, refined_context=refined_context
    )

    assert script_result.status == AgentActionStatus.SUCCESS, (
        f"Copywriter failed for {golden_case.id}: {script_result.reasoning}"
    )

    script_content = _get_script_content(golden_case, script_result)

    factcheck_spec = golden_case.expected_outcomes.fact_check
    if factcheck_spec is None:
        pytest.skip(f"No fact_check outcome spec for {golden_case.id}")

    factcheck_result = await eval_runner.run_red_team(
        script_content, vector_store=case_vs, case=golden_case
    )

    if golden_case.id in ESCALATE_CASE_IDS:
        assert factcheck_result.status in (
            AgentActionStatus.ESCALATE,
            AgentActionStatus.REVISION_NEEDED,
        ), (
            f"Expected ESCALATE or REVISION_NEEDED for {golden_case.id}, "
            f"got {factcheck_result.status}"
        )
        return

    assert factcheck_result.status in (
        AgentActionStatus.SUCCESS,
        AgentActionStatus.REVISION_NEEDED,
    ), f"Unexpected status for {golden_case.id}: {factcheck_result.status}"

    claims = factcheck_result.payload.get("claims", [])

    scores = None
    try:
        assert_claim_count_ge(claims, factcheck_spec.min_claim_count)
        assert_verdict_counts(claims, factcheck_spec.max_unsupported_claims)

        if factcheck_spec.claims_with_known_verdicts:
            for expected in factcheck_spec.claims_with_known_verdicts:
                matching = [
                    c
                    for c in claims
                    if expected.claim_text.lower() in c.get("claim_text", "").lower()
                ]
                if matching:
                    actual_verdict = matching[0].get("verdict", "").upper()
                    expected_verdict = expected.expected_verdict.value.upper()
                    assert actual_verdict == expected_verdict, (
                        f"{golden_case.id}: claim '{expected.claim_text[:50]}...' "
                        f"expected {expected_verdict}, got {actual_verdict}"
                    )

        scores = await judge_score(
            judge_llm,
            "fact_check",
            {
                "script_content": script_content,
                "refined_context": refined_context,
            },
            factcheck_result.payload,
            factcheck_spec,
        )

        threshold = golden_case.scoring.pass_threshold if golden_case.scoring else 0.70
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
        score_aggregator.record("fact_check", dim_scores)
    finally:
        if scores is not None:
            dim_scores = {d.dimension: d.score for d in scores.dimensions}
            baseline_recorder.record_case_score(
                golden_case.id,
                "fact_check",
                {
                    "weighted_average": scores.weighted_average,
                    "dimensions": dim_scores,
                    "reasoning": scores.reasoning,
                },
            )

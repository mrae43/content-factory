"""
Optimizer Agent Outcome Evals — scored against OPTIMIZER_RUBRIC.

Case coverage (4): R-001, R-002, R-004, F-004

Modes:
  - Golden mode (default): Uses reference_outputs for research/script/fact_check
    to get pre-recorded failed claims, then runs optimizer with those claims.
  - Live mode (--live): Chains all 4 agents with real LLM calls.

The optimizer test is NOT dependent on RedTeam producing REVISION_NEEDED in
golden mode — it uses pre-recorded failed claims from reference_outputs.
"""

import pytest

from app.workers.agents import AgentActionStatus
from tests.evals.assertions import (
    build_case_aware_vector_store,
)
from tests.evals.judge import judge_score
from tests.evals.schemas import GoldenCase

OPTIMIZER_CASE_IDS = [
    "R-001",
    "R-002",
    "R-004",
    "F-004",
]


def _extract_failed_claims(claims: list) -> list:
    return [
        c
        for c in claims
        if c.get("verdict", "").upper() in ("UNSUPPORTED", "CONTESTED")
    ]


def _get_refined_context(case: GoldenCase, research_result) -> str:
    if (
        case.reference_outputs
        and case.reference_outputs.research
        and case.reference_outputs.research.refined_context
    ):
        return case.reference_outputs.research.refined_context
    return research_result.payload.get("refined_context", "")


def _get_script_content(case: GoldenCase, script_result) -> str:
    if (
        case.reference_outputs
        and case.reference_outputs.script
        and case.reference_outputs.script.script_content
    ):
        return case.reference_outputs.script.script_content
    return script_result.payload.get("script_content", "")


def _get_failed_claims_from_reference(case: GoldenCase, factcheck_result) -> list:
    if (
        case.reference_outputs
        and case.reference_outputs.fact_check
        and case.reference_outputs.fact_check.claims
    ):
        return _extract_failed_claims(case.reference_outputs.fact_check.claims)
    return _extract_failed_claims(factcheck_result.payload.get("claims", []))


@pytest.mark.eval
@pytest.mark.parametrize("golden_case", OPTIMIZER_CASE_IDS, indirect=True)
async def test_optimizer_outcome(
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

    script_result = await eval_runner.run_copywriter(
        golden_case, refined_context=refined_context
    )

    assert script_result.status == AgentActionStatus.SUCCESS, (
        f"Copywriter failed for {golden_case.id}: {script_result.reasoning}"
    )

    script_content = _get_script_content(golden_case, script_result)

    factcheck_result = await eval_runner.run_red_team(
        script_content, vector_store=case_vs, case=golden_case
    )

    assert factcheck_result.status in (
        AgentActionStatus.REVISION_NEEDED,
        AgentActionStatus.SUCCESS,
    ), f"Unexpected status for {golden_case.id}: {factcheck_result.status}"

    failed_claims = _get_failed_claims_from_reference(golden_case, factcheck_result)

    assert len(failed_claims) > 0, (
        f"No failed claims found for {golden_case.id} — optimizer has nothing to patch"
    )

    optim_spec = golden_case.expected_outcomes.optimization
    if optim_spec is None:
        pytest.skip(f"No optimization outcome spec for {golden_case.id}")

    optimizer_result = await eval_runner.run_optimizer(
        script_content=script_content,
        refined_context=refined_context,
        failed_claims=failed_claims,
    )

    assert optimizer_result.status == AgentActionStatus.SUCCESS, (
        f"Optimizer failed for {golden_case.id}: {optimizer_result.reasoning}"
    )

    patched_script = optimizer_result.payload.get("script_content", "")

    assert len(patched_script) > 0, "Patched script is empty"

    scores = None
    try:
        if optim_spec.must_preserve_claims:
            for claim_text in optim_spec.must_preserve_claims:
                assert claim_text.lower() in patched_script.lower(), (
                    f"{golden_case.id}: preserved claim missing after patch: "
                    f"'{claim_text[:60]}...'"
                )

        if optim_spec.must_patch_claims:
            for claim_text in optim_spec.must_patch_claims:
                assert claim_text.lower() not in patched_script.lower(), (
                    f"{golden_case.id}: patched claim still present: '{claim_text[:60]}...'"
                )

        scores = await judge_score(
            judge_llm,
            "optimizer",
            {
                "original_script": script_content,
                "failed_claims": failed_claims,
                "refined_context": refined_context,
            },
            optimizer_result.payload,
            optim_spec,
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
        score_aggregator.record("optimizer", dim_scores)
    finally:
        if scores is not None:
            dim_scores = {d.dimension: d.score for d in scores.dimensions}
            baseline_recorder.record_case_score(
                golden_case.id,
                "optimizer",
                {
                    "weighted_average": scores.weighted_average,
                    "dimensions": dim_scores,
                    "reasoning": scores.reasoning,
                },
            )

"""
Eval 1.1 — Research Coverage.

Rule-based eval that checks 5 dimensions of research ingestion:
  - chunk count, domain diversity, no duplicates, scope, source_type.

Golden mode (default): Uses mock web results from JSON fixture,
deterministic assertions, no API calls.

Live mode (--live): Real Tavily search + ingestion capture,
softer thresholds (>=1), records baselines.
"""

import pytest

from tests.evals.assertions import (
    check_chunk_count,
    check_domain_diversity,
    check_no_duplicates,
    check_scope,
    check_source_type,
)
from tests.evals.schemas import ResearchingCase

COVERAGE_CASE_IDS = [
    "coverage-happy",
    "coverage-sparse",
    "coverage-duplicates",
    "coverage-single-domain",
    "coverage-wrong-metadata",
    "coverage-boundary",
]


@pytest.mark.eval
@pytest.mark.parametrize("researching_case", COVERAGE_CASE_IDS, indirect=True)
async def test_research_coverage(
    researching_case: ResearchingCase,
    researching_runner,
    baseline_recorder,
    request,
):
    case = researching_case
    live = request.config.getoption("live", default=False)

    if live:
        result = await researching_runner.run_researching_live(case)
        chunks = result["chunks"]
        errors = []
        err = check_chunk_count(chunks, min_count=1)
        if err:
            errors.append(err)
        err = check_domain_diversity(chunks, min_domains=1)
        if err:
            errors.append(err)
        err = check_scope(chunks, "LOCAL")
        if err:
            errors.append(err)
        err = check_source_type(chunks, "WEB_SEARCH")
        if err:
            errors.append(err)
        domain_count = len({c["url"] for c in chunks if c["url"]})
        baseline_recorder.record_case_score(
            case.id,
            "research_coverage",
            {
                "chunk_count": len(chunks),
                "domain_count": domain_count,
                "assertion_errors": errors,
            },
        )
        if errors:
            pytest.fail(
                f"{case.id}: {len(errors)} assertion(s) failed:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
    else:
        result = researching_runner.run_researching(case)
        chunks = result["chunks"]
        errors = []
        if case.expectations.min_chunks is not None:
            err = check_chunk_count(chunks, case.expectations.min_chunks)
            if err:
                errors.append(err)
        if case.expectations.min_domains is not None:
            err = check_domain_diversity(chunks, case.expectations.min_domains)
            if err:
                errors.append(err)
        if case.expectations.max_similarity is not None:
            err = check_no_duplicates(chunks, case.expectations.max_similarity)
            if err:
                errors.append(err)
        err = check_scope(chunks, "LOCAL")
        if err:
            errors.append(err)
        err = check_source_type(chunks, "WEB_SEARCH")
        if err:
            errors.append(err)

        if case.should_pass:
            assert not errors, (
                f"{case.id}: expected all checks to pass, got errors:\n"
                + "\n".join(f"  - {e}" for e in errors)
            )
        else:
            assert errors, (
                f"{case.id}: expected at least one check to fail, all passed"
            )

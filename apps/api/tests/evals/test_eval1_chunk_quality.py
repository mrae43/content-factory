"""
Eval 1.2 — Chunk Quality.

LLM-as-Judge eval that scores research chunks on three 1–5 dimensions:
  - Topical relevance
  - Information density
  - Coherence

Golden mode (default): Uses cached_responses from eval1_research.json fixture.
Live mode (--live): Calls ChunkQualityScorer with real LLM and detects drift.
"""

import pytest

from tests.evals.schemas import QualityCorpusEntry

QUALITY_ENTRY_IDS = [
    "quality-brics",
    "quality-sparse-FR1",
    "quality-boilerplate",
    "quality-fusion",
    "quality-ev-battery",
    "quality-space",
    "quality-ai-regulation",
]


@pytest.mark.eval
class TestChunkQuality:
    @pytest.mark.parametrize("quality_entry", QUALITY_ENTRY_IDS, indirect=True)
    async def test_chunk_dimension_means(
        self,
        quality_entry: QualityCorpusEntry,
        chunk_quality_scorer,
        baseline_recorder,
        request,
    ):
        entry = quality_entry

        if not entry.source_chunks:
            pytest.skip(
                f"{entry.id}: quality_corpus not populated "
                f"(run scripts/capture_corpus.py first)"
            )

        live = request.config.getoption("live", default=False)
        golden_mode = bool(entry.cached_responses) and not live

        if golden_mode:
            scores = entry.cached_responses
            drift: list[str] = []
        else:
            scores = await chunk_quality_scorer.score_chunks(
                entry.topic, entry.source_chunks
            )
            drift = chunk_quality_scorer.detect_drift(entry.cached_responses, scores)

        means = chunk_quality_scorer.compute_entry_means(scores)

        violations: list[str] = []
        for dim in ["relevance", "density", "coherence"]:
            key = f"{dim}_mean"
            if means[key] < 3.5:
                violations.append(f"{key} {means[key]:.2f} < 3.5")

        low_rel_count = sum(1 for s in scores if s.relevance < 3)
        low_rel_pct = low_rel_count / len(scores) * 100
        if low_rel_pct > 20:
            violations.append(f"WARNING: {low_rel_pct:.0f}% chunks relevance < 3")

        baseline_recorder.record_case_score(
            entry.id,
            "chunk_quality",
            {
                "dimension_means": means,
                "low_relevance_pct": low_rel_pct,
                "drift_flags": drift,
            },
        )

        if violations:
            pytest.fail(f"{entry.id}: {'; '.join(violations)}")

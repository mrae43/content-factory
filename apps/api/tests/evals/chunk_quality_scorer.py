"""
LLM-as-Judge scorer for Eval 1.2 — Chunk Quality.

Scores individual chunks on three 1-5 dimensions (relevance, density, coherence)
using a standalone prompt template separate from judge.py's 0.0/0.5/1.0 scale.
"""

from statistics import mean
from typing import Any

from tests.evals.schemas import CachedChunkScore, SourceChunk

CHUNK_QUALITY_PROMPT = """You are evaluating a research chunk extracted from a web search result about the topic: "{topic}".

Chunk text:
---
{chunk_text}
---

Score the chunk on each dimension from 1–5:
1. Topical relevance: Is the chunk substantively about the topic?
2. Information density: Does it contain specific, usable facts (not boilerplate)?
3. Coherence: Is it a complete, readable unit (not mid-sentence truncation)?

Return JSON: {{"relevance": N, "density": N, "coherence": N}}"""

MEAN_THRESHOLD = 3.5
LOW_RELEVANCE_WARN_PCT = 20
DRIFT_THRESHOLD = 0.5


class ChunkQualityScorer:
    """Scores individual chunks on relevance (1–5), density (1–5), coherence (1–5).
    Separate from judge.py — own 1–5 scale, own prompt template."""

    def __init__(self, judge_llm):
        self._llm = judge_llm
        self._chain = judge_llm.with_structured_output(CachedChunkScore)

    async def score_chunk(self, topic: str, chunk_text: str) -> CachedChunkScore:
        prompt = CHUNK_QUALITY_PROMPT.format(topic=topic, chunk_text=chunk_text)
        result: CachedChunkScore = await self._chain.ainvoke(prompt)
        return result

    async def score_chunks(
        self, topic: str, chunks: list[SourceChunk]
    ) -> list[CachedChunkScore]:
        results: list[CachedChunkScore] = []
        for chunk in chunks:
            score = await self.score_chunk(topic, chunk.content)
            results.append(score)
        return results

    def compute_entry_means(
        self, scores: list[CachedChunkScore]
    ) -> dict[str, float]:
        if not scores:
            return {"relevance_mean": 0.0, "density_mean": 0.0, "coherence_mean": 0.0}
        return {
            "relevance_mean": round(mean(s.relevance for s in scores), 4),
            "density_mean": round(mean(s.density for s in scores), 4),
            "coherence_mean": round(mean(s.coherence for s in scores), 4),
        }

    def check_thresholds(
        self, means: dict[str, float], scores: list[CachedChunkScore]
    ) -> list[str]:
        violations: list[str] = []
        for dim in ["relevance", "density", "coherence"]:
            key = f"{dim}_mean"
            if means.get(key, 0.0) < MEAN_THRESHOLD:
                violations.append(f"{key} {means[key]:.2f} < {MEAN_THRESHOLD}")
        if scores:
            low_rel = sum(1 for s in scores if s.relevance < 3)
            pct = low_rel / len(scores) * 100
            if pct > LOW_RELEVANCE_WARN_PCT:
                violations.append(
                    f"WARNING: {pct:.0f}% chunks relevance < 3"
                )
        return violations

    def detect_drift(
        self,
        cached: list[CachedChunkScore],
        live: list[CachedChunkScore],
    ) -> list[str]:
        if not cached or not live:
            return ["insufficient data for drift detection"]
        if len(cached) != len(live):
            return [
                f"chunk count mismatch: cached {len(cached)} vs live {len(live)}"
            ]
        cached_means = self.compute_entry_means(cached)
        live_means = self.compute_entry_means(live)
        flags: list[str] = []
        for dim in ["relevance", "density", "coherence"]:
            key = f"{dim}_mean"
            diff = abs(live_means[key] - cached_means[key])
            if diff > DRIFT_THRESHOLD:
                flags.append(
                    f"{key} drifted {diff:.2f} (cached {cached_means[key]:.2f} "
                    f"vs live {live_means[key]:.2f})"
                )
        return flags

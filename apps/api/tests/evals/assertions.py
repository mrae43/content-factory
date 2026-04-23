"""
Deterministic assertion helpers for outcome eval test files.

All functions raise AssertionError with descriptive messages on failure.
"""

import re
from typing import Any, Dict, List, Sequence, Tuple
from unittest.mock import AsyncMock
from uuid import uuid4

from tests.evals.schemas import GoldenCase


def assert_must_include(text: str, facts: List[str]) -> None:
    missing = []
    for fact in facts:
        if fact.lower() not in text.lower():
            missing.append(fact)
    assert not missing, f"Missing required facts: {missing}"


def assert_must_avoid(text: str, patterns: List[str]) -> None:
    found = []
    for pattern in patterns:
        if pattern.lower() in text.lower():
            found.append(pattern)
    assert not found, f"Found forbidden patterns: {found}"


def assert_word_count_in_range(text: str, word_range: Tuple[int, int]) -> None:
    word_count = len(text.split())
    lo, hi = word_range
    assert lo <= word_count <= hi, f"Word count {word_count} outside range [{lo}, {hi}]"


def assert_scene_count_in_range(
    storyboard: Sequence, scene_range: Tuple[int, int]
) -> None:
    count = len(storyboard)
    lo, hi = scene_range
    assert lo <= count <= hi, f"Scene count {count} outside range [{lo}, {hi}]"


def assert_storyboard_fields(
    storyboard: Sequence[Dict[str, Any]], required_fields: List[str]
) -> None:
    missing = []
    for i, scene in enumerate(storyboard):
        for field in required_fields:
            if field not in scene or not scene[field]:
                missing.append(f"scene[{i}].{field}")
    assert not missing, f"Missing storyboard fields: {missing}"


_HOOK_PATTERNS = [
    r"\?",
    r"did you know",
    r"!",
    r"imagine",
    r"what if",
    r"guess",
    r"believe",
    r"think",
]


def assert_has_hook(script_content: str) -> None:
    first_sentences = script_content[:300].lower()
    found = any(re.search(p, first_sentences) for p in _HOOK_PATTERNS)
    assert found, "Script missing a hook pattern in the first 300 characters"


_LOOP_PATTERNS = [
    r"\?",
    r"subscribe",
    r"follow",
    r"next time",
    r"stay tuned",
    r"what do you think",
    r"let me know",
    r"comment",
]


def assert_has_loop(script_content: str) -> None:
    last_sentences = script_content[-300:].lower()
    found = any(re.search(p, last_sentences) for p in _LOOP_PATTERNS)
    assert found, "Script missing a loop/CTA pattern in the last 300 characters"


def assert_claim_count_ge(claims: Sequence, min_count: int) -> None:
    count = len(claims)
    assert count >= min_count, f"Claim count {count} below minimum {min_count}"


def assert_verdict_counts(
    claims: Sequence[Dict[str, Any]], max_unsupported: int
) -> None:
    unsupported = sum(
        1 for c in claims if c.get("verdict", "").upper() == "UNSUPPORTED"
    )
    assert unsupported <= max_unsupported, (
        f"UNSUPPORTED claims ({unsupported}) exceed max ({max_unsupported})"
    )


def build_case_aware_vector_store(
    case: GoldenCase, similarity_score: float = 0.88
) -> AsyncMock:
    """
    Splits the golden case raw_text into chunks and returns an AsyncMock
    vector store whose semantic_search returns those chunks.
    """
    raw_text = case.input.pre_context.raw_text or ""
    if not raw_text.strip():
        store = AsyncMock()
        store.semantic_search.return_value = []
        store.ingest_chunks = AsyncMock(return_value=0)
        return store

    chunk_size = min(2000, max(200, len(raw_text) // 3))
    chunk_overlap = chunk_size // 5
    total_len = len(raw_text)

    if total_len <= chunk_size:
        chunks_text = [raw_text]
    else:
        chunks_text = []
        start = 0
        while start < total_len:
            end = min(start + chunk_size, total_len)
            chunks_text.append(raw_text[start:end])
            start += chunk_size - chunk_overlap
            if len(chunks_text) >= 10:
                break

    results = [
        {
            "id": str(uuid4()),
            "content": chunk,
            "meta": {"scope": "RAW-CONTEXT", "version": "1.0"},
            "job_id": str(uuid4()),
            "similarity_score": similarity_score,
        }
        for chunk in chunks_text
    ]

    store = AsyncMock()
    store.semantic_search.return_value = results
    store.ingest_chunks = AsyncMock(return_value=len(results))
    return store

"""
Deterministic assertion helpers for outcome eval test files.

All functions raise AssertionError with descriptive messages on failure.
"""

import re
from typing import Any, Dict, List, Sequence, Tuple
from unittest.mock import AsyncMock
from uuid import uuid4

from app.services.chunking import split_pre_context
from tests.evals.schemas import GoldenCase


_STOP_WORDS = frozenset(
    "the a an is are was were be been being have has had do does did "
    "will would shall should may might must can could of in on at to "
    "for with by from about into through during before after above below "
    "between out off over under again further then once and but or nor "
    "not so yet both either neither each every all any few more most "
    "other some such no only own same than too very it its this that "
    "these those he she we they what which who whom how when where why "
    "if as just also".split()
)


def _key_terms(text: str) -> List[str]:
    tokens = re.findall(r"\b\w+\b", text.lower())
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 2]


def assert_must_include(text: str, facts: List[str], min_overlap: float = 0.5) -> None:
    missing = []
    for fact in facts:
        terms = _key_terms(fact)
        if not terms:
            continue
        text_lower = text.lower()
        matched = sum(1 for t in terms if t in text_lower)
        ratio = matched / len(terms)
        if ratio < min_overlap:
            missing.append(
                f"{fact} (matched {matched}/{len(terms)} key terms = {ratio:.0%})"
            )
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
    case: GoldenCase,
    job_id: str | None = None,
    similarity_score: float = 0.88,
) -> AsyncMock:
    """
    Splits the golden case raw_text into chunks using the production
    MarkdownTextSplitter and returns an AsyncMock vector store whose
    semantic_search returns those chunks.

    Respects job_id and scopes parameters for realistic mock behavior.
    """
    raw_text = case.input.pre_context.raw_text or ""
    if not raw_text.strip():
        return _empty_vector_store()

    chunks_text = split_pre_context(raw_text)

    if not chunks_text:
        return _empty_vector_store()

    effective_job_id = job_id or str(uuid4())

    results = [
        {
            "id": str(uuid4()),
            "content": chunk,
            "meta": {"scope": "RAW-CONTEXT", "version": "1.0"},
            "job_id": effective_job_id,
            "similarity_score": similarity_score,
        }
        for chunk in chunks_text
    ]

    store = AsyncMock()

    async def _semantic_search(query, **kwargs):
        scopes = kwargs.get("scopes")
        if scopes and "RAW-CONTEXT" not in scopes and "LOCAL" not in scopes:
            return []
        return results

    store.semantic_search.side_effect = _semantic_search
    store.ingest_chunks = AsyncMock(return_value=len(results))
    return store


def _empty_vector_store() -> AsyncMock:
    store = AsyncMock()
    store.semantic_search.return_value = []
    store.ingest_chunks = AsyncMock(return_value=0)
    return store

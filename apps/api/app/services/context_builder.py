import logging
from typing import Any, Dict, List, Optional
from uuid import UUID

from app.services.vector_store import ContentFactoryVectorStore
from app.schemas.shorts import AssembledContext

logger = logging.getLogger("factory.context_builder")

TOPIC_RELEVANCE_THRESHOLDS = [
    (0.75, "HIGH"),
    (0.5, "MEDIUM"),
]


def _derive_topic_relevance(score: float) -> str:
    for threshold, label in TOPIC_RELEVANCE_THRESHOLDS:
        if score >= threshold:
            return label
    return "LOW"


def _compose_query(topic: str, story_directives: dict) -> str:
    parts = [topic]
    for key in ("tone", "angle", "target_audience"):
        val = story_directives.get(key, "")
        if val:
            parts.append(str(val))
    return " ".join(parts)


def _format_evidence_sections(chunks: List[Dict[str, Any]]) -> str:
    if not chunks:
        return ""

    sorted_chunks = sorted(
        chunks, key=lambda c: c.get("similarity_score", 0), reverse=True
    )

    lines = ["## Retrieved Evidence", ""]
    for i, chunk in enumerate(sorted_chunks, 1):
        score = chunk.get("similarity_score", 0)
        source_type = chunk.get("source_type", "UNKNOWN")
        relevance = chunk.get("topic_relevance", "UNKNOWN")
        content = chunk.get("content", "")
        lines.append(
            f"Chunk {i} (similarity: {score:.2f}, source: {source_type}, relevance: {relevance}):"
        )
        lines.append(content)
        lines.append("")

    return "\n".join(lines)


async def build(
    topic: str,
    story_directives: dict,
    refined_context: str,
    vector_store: ContentFactoryVectorStore,
    job_id: UUID,
    top_k: int = 10,
) -> AssembledContext:
    query = _compose_query(topic, story_directives)
    logger.info(
        f"ContextBuilder query for job {job_id}: {query!r} (top_k={top_k})"
    )

    retrieved = await vector_store.semantic_search(
        query=query,
        job_id=job_id,
        scopes=["RAW-CONTEXT", "LOCAL"],
        top_k=top_k,
    )

    enriched: List[Dict[str, Any]] = []
    for chunk in retrieved:
        score = chunk.get("similarity_score", 0)
        meta = chunk.get("meta", {})
        enriched.append(
            {
                "id": chunk.get("id"),
                "content": chunk.get("content", ""),
                "similarity_score": score,
                "topic_relevance": _derive_topic_relevance(score),
                "source_type": meta.get("source_type", "INFERRED"),
                "meta": meta,
            }
        )

    evidence_sections = _format_evidence_sections(enriched)

    log_count = len(enriched)
    if log_count == 0:
        logger.warning(
            f"ContextBuilder retrieved 0 chunks for job {job_id} — evidence_sections will be empty"
        )
    else:
        top_score = enriched[0]["similarity_score"]
        logger.info(
            f"ContextBuilder assembled {log_count} chunks for job {job_id} "
            f"(top score: {top_score:.3f})"
        )

    return AssembledContext(
        narrative_summary=refined_context,
        evidence_sections=evidence_sections,
        raw_chunks=enriched,
    )

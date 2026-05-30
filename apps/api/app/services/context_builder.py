import logging
from typing import Any, Dict, List
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


def _compose_diversified_queries(
    title: str, story_directives: dict, user_reference: str = ""
) -> tuple[str, str]:
    query1 = title
    parts = [title]
    angle = story_directives.get("angle", "")
    if angle:
        parts.append(str(angle))
    if user_reference:
        parts.append(user_reference[:500])
    query2 = " ".join(parts)
    return query1, query2


def _dedup_and_cap(
    results: List[Dict[str, Any]], max_chunks: int = 12
) -> List[Dict[str, Any]]:
    seen: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cid = r.get("id")
        score = r.get("similarity_score", 0)
        if cid not in seen or score > seen[cid].get("similarity_score", 0):
            seen[cid] = r
    sorted_results = sorted(
        seen.values(), key=lambda x: x.get("similarity_score", 0), reverse=True
    )
    return sorted_results[:max_chunks]


def _enrich_chunks(
    chunks: List[Dict[str, Any]], default_source: str
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for chunk in chunks:
        score = chunk.get("similarity_score", 0)
        meta = chunk.get("meta", {})
        enriched.append(
            {
                "id": chunk.get("id"),
                "content": chunk.get("content", ""),
                "similarity_score": score,
                "topic_relevance": _derive_topic_relevance(score),
                "source_type": meta.get("source_type", default_source),
                "meta": meta,
            }
        )
    return enriched


def _format_evidence_sections(
    local_chunks: List[Dict[str, Any]],
    global_chunks: List[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    chunk_num = 0

    if local_chunks:
        parts.append("=== CURRENT RUN RESEARCH ===")
        parts.append("")
        for chunk in local_chunks:
            chunk_num += 1
            score = chunk.get("similarity_score", 0)
            source_type = chunk.get("source_type", "UNKNOWN")
            relevance = chunk.get("topic_relevance", "UNKNOWN")
            content = chunk.get("content", "")
            parts.append(
                f"#### [Chunk {chunk_num:02d}] | Source: {source_type} "
                f"| Relevance: {relevance} (Match: {score:.2f})"
            )
            for line in content.split("\n"):
                parts.append(f"> {line}")
            parts.append("")

    if global_chunks:
        parts.append("=== SYSTEM INTEL ===")
        parts.append("")
        for chunk in global_chunks:
            chunk_num += 1
            score = chunk.get("similarity_score", 0)
            relevance = chunk.get("topic_relevance", "UNKNOWN")
            content = chunk.get("content", "")
            parts.append(
                f"#### [Chunk {chunk_num:02d}] | Relevance: {relevance} (Match: {score:.2f})"
            )
            for line in content.split("\n"):
                parts.append(f"> {line}")
            parts.append("")

    return "\n".join(parts)


async def build(
    title: str,
    story_directives: dict,
    refined_context: str,
    vector_store: ContentFactoryVectorStore,
    job_id: UUID,
    top_k: int = 10,
    user_reference: str = "",
) -> AssembledContext:
    query1, query2 = _compose_diversified_queries(
        title, story_directives, user_reference
    )
    logger.info(
        f"ContextBuilder diversified queries for job {job_id}: "
        f"q1={query1!r}, q2={query2!r}"
    )

    local_raw: List[Dict[str, Any]] = []
    for q in (query1, query2):
        results = await vector_store.semantic_search(
            query=q,
            job_id=job_id,
            scopes=["RAW-CONTEXT", "LOCAL"],
            top_k=7,
        )
        local_raw.extend(results)
    local_deduped = _dedup_and_cap(local_raw, top_k)

    global_raw: List[Dict[str, Any]] = []
    for q in (query1, query2):
        results = await vector_store.semantic_search(
            query=q,
            job_id=None,
            scopes=["GLOBAL"],
            top_k=7,
        )
        global_raw.extend(results)
    global_deduped = _dedup_and_cap(global_raw, top_k)

    enriched_local = _enrich_chunks(local_deduped, "INFERRED")
    enriched_global = _enrich_chunks(global_deduped, "SYSTEM_INTEL")

    evidence_sections = _format_evidence_sections(enriched_local, enriched_global)

    all_enriched = enriched_local + enriched_global
    log_count = len(all_enriched)
    if log_count == 0:
        logger.warning(
            f"ContextBuilder retrieved 0 chunks for job {job_id} — evidence_sections will be empty"
        )
    else:
        top_score = all_enriched[0]["similarity_score"]
        logger.info(
            f"ContextBuilder assembled {log_count} chunks for job {job_id} "
            f"(top score: {top_score:.3f}, "
            f"{len(enriched_global)} GLOBAL)"
        )

    return AssembledContext(
        narrative_summary=refined_context,
        evidence_sections=evidence_sections,
        raw_chunks=all_enriched,
    )

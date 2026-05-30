import logging
import re
from typing import Dict, Optional
from uuid import UUID

from app.services.tools import Tool
from app.services.vector_store import ContentFactoryVectorStore

logger = logging.getLogger("factory.optimizer_fallback")


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def find_evidence_for_claim(
    claim_text: str,
    red_team_evidence: Dict[str, dict],
) -> Optional[dict]:
    if claim_text in red_team_evidence:
        return red_team_evidence[claim_text]
    normalized = normalize_text(claim_text)
    for key, val in red_team_evidence.items():
        if normalize_text(key) == normalized:
            return val
    return None


def make_gated_search_tool(
    vector_store: ContentFactoryVectorStore,
    red_team_evidence: Dict[str, dict],
    job_id: UUID,
    top_k: int = 3,
    similarity_threshold: Optional[float] = None,
) -> Tool:
    fallback_count: list[int] = [0]

    async def _retrieve_evidence_for_claim(
        claim_text: str,
    ) -> str:
        evidence = find_evidence_for_claim(claim_text, red_team_evidence)
        if evidence and evidence.get("evidence_text"):
            correction = evidence["evidence_text"].strip()
            if correction.lower() not in ("", "wrong", "n/a", "none", "no evidence"):
                logger.info(
                    f"Gated tool PRIMARY hit for claim={claim_text[:80]!r} "
                    f"(verdict={evidence.get('verdict')})"
                )
                return correction

        fallback_count[0] += 1
        logger.info(
            f"Gated tool FALLBACK for claim={claim_text[:80]!r} "
            f"(fallback #{fallback_count[0]})"
        )

        results = await vector_store.semantic_search(
            query=claim_text,
            job_id=job_id,
            scopes=["RAW-CONTEXT", "LOCAL", "GLOBAL"],
            top_k=top_k,
            similarity_threshold=similarity_threshold,
        )
        if results:
            return "\n\n".join(r["content"] for r in results)

        return "No evidence found for this claim."

    _retrieve_evidence_for_claim.fallback_count = fallback_count

    return Tool(
        name="retrieve_evidence_for_claim",
        description="Search the knowledge base for evidence supporting or refuting a specific claim.",
        callable=_retrieve_evidence_for_claim,
        permissions={"ScriptOptimizerAgent"},
    )

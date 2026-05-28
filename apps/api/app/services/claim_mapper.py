import uuid
from typing import Dict, List, Optional

import numpy as np


def cosine_similarity_matrix(
    prev_vecs: List[List[float]],
    new_vecs: List[List[float]],
) -> np.ndarray:
    a = np.array(prev_vecs)
    b = np.array(new_vecs)
    return a @ b.T


def map_claims(
    prev_active: List[dict],
    new_claims: List[dict],
    embedder,
    threshold: float = 0.75,
) -> Dict[int, str]:
    prev_texts = [c["text"] for c in prev_active]
    new_texts = [c["claim_text"] for c in new_claims]

    prev_vecs = embedder.embed_documents(prev_texts)
    new_vecs = embedder.embed_documents(new_texts)

    matrix = cosine_similarity_matrix(prev_vecs, new_vecs)

    assigned_prev: set[int] = set()
    assigned_new: set[int] = set()
    mapping: Dict[int, str] = {}

    pairs = []
    for i in range(len(prev_active)):
        for j in range(len(new_claims)):
            pairs.append((matrix[i][j], i, j))
    pairs.sort(key=lambda x: -x[0])

    for score, prev_idx, new_idx in pairs:
        if prev_idx in assigned_prev or new_idx in assigned_new:
            continue
        if score < threshold:
            break
        mapping[new_idx] = prev_active[prev_idx]["uuid"]
        assigned_prev.add(prev_idx)
        assigned_new.add(new_idx)

    return mapping


def compute_verdict_delta(
    prev_active: List[dict],
    new_claims: List[dict],
    mapping: Dict[int, str],
) -> dict:
    prev_failed_ids: set[str] = {
        c["uuid"]
        for c in prev_active
        if c.get("latest_verdict", "") in ("UNSUPPORTED", "CONTESTED")
    }
    current_failed_ids: set[str] = set()
    for i, c in enumerate(new_claims):
        if c.get("verdict") in ("UNSUPPORTED", "CONTESTED"):
            uid = mapping.get(i)
            if uid:
                current_failed_ids.add(uid)

    return {
        "resolved": sorted(prev_failed_ids - current_failed_ids),
        "regressed": sorted(current_failed_ids - prev_failed_ids),
        "unchanged_failed": sorted(prev_failed_ids & current_failed_ids),
    }


def init_ledger(claims: List[dict]) -> dict:
    active_claims = []
    for c in claims:
        active_claims.append(
            {
                "uuid": str(uuid.uuid4()),
                "text": c.get("claim_text", ""),
                "first_seen_iteration": 0,
                "last_modified_iteration": 0,
                "latest_verdict": c.get("verdict", "UNCERTAIN"),
                "failure_reason": c.get("evidence_text", ""),
            }
        )
    return {
        "current_iteration": 0,
        "active_claims": active_claims,
        "historical_iterations": [],
    }


def update_ledger(
    ledger: dict,
    new_claims: List[dict],
    mapping: Dict[int, str],
    delta: dict,
    patches_applied: Optional[List[str]] = None,
) -> dict:
    iteration = ledger.get("current_iteration", 0) + 1

    uuid_to_entry: Dict[str, dict] = {
        c["uuid"]: c for c in ledger.get("active_claims", [])
    }

    seen_uuids: set[str] = set()

    for i, nc in enumerate(new_claims):
        claim_uuid = mapping.get(i)
        if claim_uuid and claim_uuid in uuid_to_entry:
            existing = uuid_to_entry[claim_uuid]
            existing["text"] = nc.get("claim_text", existing["text"])
            existing["last_modified_iteration"] = iteration
            existing["latest_verdict"] = nc.get("verdict", "UNCERTAIN")
            existing["failure_reason"] = nc.get("evidence_text", "")
            seen_uuids.add(claim_uuid)
        else:
            new_uuid = str(uuid.uuid4())
            uuid_to_entry[new_uuid] = {
                "uuid": new_uuid,
                "text": nc.get("claim_text", ""),
                "first_seen_iteration": iteration,
                "last_modified_iteration": iteration,
                "latest_verdict": nc.get("verdict", "UNCERTAIN"),
                "failure_reason": nc.get("evidence_text", ""),
            }
            seen_uuids.add(new_uuid)

    resolved_ids = set(delta.get("resolved", []))
    active_claims = [
        c
        for c in uuid_to_entry.values()
        if c["uuid"] in seen_uuids and c["uuid"] not in resolved_ids
    ]

    claims_snapshot = []
    for i, nc in enumerate(new_claims):
        claim_uuid = mapping.get(i)
        if not claim_uuid:
            text = nc.get("claim_text", "")
            for entry in uuid_to_entry.values():
                if entry["text"] == text:
                    claim_uuid = entry["uuid"]
                    break
        if not claim_uuid:
            claim_uuid = str(uuid.uuid4())
        claims_snapshot.append(
            {
                "uuid": claim_uuid,
                "text": nc.get("claim_text", ""),
                "verdict": nc.get("verdict", "UNCERTAIN"),
            }
        )

    historical = list(ledger.get("historical_iterations", []))
    historical.append(
        {
            "iteration": iteration,
            "patches_applied": patches_applied or [],
            "claims_snapshot": claims_snapshot,
        }
    )

    return {
        "current_iteration": iteration,
        "active_claims": active_claims,
        "historical_iterations": historical,
    }

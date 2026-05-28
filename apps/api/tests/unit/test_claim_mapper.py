import pytest
import uuid
from unittest.mock import MagicMock

import numpy as np

from app.services.claim_mapper import (
    cosine_similarity_matrix,
    map_claims,
    compute_verdict_delta,
    init_ledger,
    update_ledger,
)


@pytest.mark.unit
def test_cosine_similarity_matrix_identical():
    vecs = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    matrix = cosine_similarity_matrix(vecs, vecs)
    assert matrix.shape == (2, 2)
    assert np.isclose(matrix[0][0], 1.0)
    assert np.isclose(matrix[0][1], 0.0)
    assert np.isclose(matrix[1][0], 0.0)
    assert np.isclose(matrix[1][1], 1.0)


@pytest.mark.unit
def test_cosine_similarity_matrix_orthogonal():
    a = [[1.0, 0.0]]
    b = [[0.0, 1.0]]
    matrix = cosine_similarity_matrix(a, b)
    assert np.isclose(matrix[0][0], 0.0)


@pytest.mark.unit
def test_map_claims_all_match():
    prev_active = [
        {"uuid": str(uuid.uuid4()), "text": "GDP grew 5%"},
        {"uuid": str(uuid.uuid4()), "text": "Inflation at 3%"},
    ]
    new_claims = [
        {"claim_text": "GDP grew 5%", "verdict": "SUPPORTED"},
        {"claim_text": "Inflation at 3%", "verdict": "SUPPORTED"},
    ]
    embedder = MagicMock()
    embedder.embed_documents.side_effect = [
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.0, 1.0]],
    ]

    mapping = map_claims(prev_active, new_claims, embedder, threshold=0.5)

    assert len(mapping) == 2
    assert mapping[0] == prev_active[0]["uuid"]
    assert mapping[1] == prev_active[1]["uuid"]


@pytest.mark.unit
def test_map_claims_partial_match():
    prev_active = [
        {"uuid": str(uuid.uuid4()), "text": "GDP grew 5%"},
        {"uuid": str(uuid.uuid4()), "text": "Inflation at 3%"},
    ]
    new_claims = [
        {"claim_text": "GDP grew 5%", "verdict": "SUPPORTED"},
        {"claim_text": "New unrelated claim", "verdict": "UNSUPPORTED"},
    ]
    embedder = MagicMock()
    embedder.embed_documents.side_effect = [
        [[1.0, 0.0], [0.0, 1.0]],
        [[1.0, 0.0], [0.9, 0.1]],
    ]

    mapping = map_claims(prev_active, new_claims, embedder, threshold=0.8)

    assert len(mapping) == 1
    assert mapping[0] == prev_active[0]["uuid"]


@pytest.mark.unit
def test_map_claims_empty():
    mapping = map_claims([], [], MagicMock())
    assert mapping == {}


@pytest.mark.unit
def test_map_claims_below_threshold():
    prev_active = [
        {"uuid": str(uuid.uuid4()), "text": "GDP grew 5%"},
    ]
    new_claims = [
        {"claim_text": "Completely different text", "verdict": "UNSUPPORTED"},
    ]
    embedder = MagicMock()
    embedder.embed_documents.side_effect = [
        [[1.0]],
        [[0.0]],
    ]

    mapping = map_claims(prev_active, new_claims, embedder, threshold=0.8)

    assert mapping == {}


@pytest.mark.unit
def test_compute_verdict_delta():
    prev_active = [
        {"uuid": "a", "text": "Claim A", "latest_verdict": "UNSUPPORTED"},
        {"uuid": "b", "text": "Claim B", "latest_verdict": "SUPPORTED"},
        {"uuid": "c", "text": "Claim C", "latest_verdict": "CONTESTED"},
    ]
    new_claims = [
        {"claim_text": "Claim A", "verdict": "SUPPORTED"},
        {"claim_text": "Claim B", "verdict": "SUPPORTED"},
        {"claim_text": "Claim C", "verdict": "CONTESTED"},
    ]
    mapping = {0: "a", 1: "b", 2: "c"}

    delta = compute_verdict_delta(prev_active, new_claims, mapping)

    assert "a" in delta["resolved"]
    assert "c" in delta["unchanged_failed"]
    assert delta["regressed"] == []


@pytest.mark.unit
def test_compute_verdict_delta_new_failure():
    prev_active = [
        {"uuid": "a", "text": "Claim A", "latest_verdict": "SUPPORTED"},
    ]
    new_claims = [
        {"claim_text": "Claim A", "verdict": "UNSUPPORTED"},
    ]
    mapping = {0: "a"}

    delta = compute_verdict_delta(prev_active, new_claims, mapping)

    assert "a" in delta["regressed"]
    assert delta["resolved"] == []


@pytest.mark.unit
def test_init_ledger():
    claims = [
        {
            "claim_text": "GDP grew 5%",
            "verdict": "UNSUPPORTED",
            "evidence_text": "No source",
        },
        {
            "claim_text": "Inflation at 3%",
            "verdict": "SUPPORTED",
            "evidence_text": "Verified",
        },
    ]

    ledger = init_ledger(claims)

    assert ledger["current_iteration"] == 0
    assert len(ledger["active_claims"]) == 2
    assert ledger["historical_iterations"] == []
    for entry in ledger["active_claims"]:
        assert "uuid" in entry
        assert entry["first_seen_iteration"] == 0
        assert entry["last_modified_iteration"] == 0
    assert ledger["active_claims"][0]["text"] == "GDP grew 5%"
    assert ledger["active_claims"][0]["latest_verdict"] == "UNSUPPORTED"


@pytest.mark.unit
def test_update_ledger():
    ledger = {
        "current_iteration": 0,
        "active_claims": [
            {
                "uuid": "a",
                "text": "GDP grew 5%",
                "first_seen_iteration": 0,
                "last_modified_iteration": 0,
                "latest_verdict": "UNSUPPORTED",
                "failure_reason": "No source",
            },
        ],
        "historical_iterations": [],
    }
    new_claims = [
        {"claim_text": "GDP grew 3.2%", "verdict": "SUPPORTED"},
    ]
    mapping = {0: "a"}
    delta = {"resolved": ["a"], "regressed": [], "unchanged_failed": []}

    updated = update_ledger(ledger, new_claims, mapping, delta)

    assert updated["current_iteration"] == 1
    assert len(updated["historical_iterations"]) == 1
    assert updated["historical_iterations"][0]["iteration"] == 1
    assert updated["historical_iterations"][0]["claims_snapshot"][0]["uuid"] == "a"
    assert (
        updated["historical_iterations"][0]["claims_snapshot"][0]["verdict"]
        == "SUPPORTED"
    )


@pytest.mark.unit
def test_update_ledger_new_claim():
    ledger = {
        "current_iteration": 0,
        "active_claims": [
            {
                "uuid": "a",
                "text": "GDP grew 5%",
                "first_seen_iteration": 0,
                "last_modified_iteration": 0,
                "latest_verdict": "UNSUPPORTED",
                "failure_reason": "No source",
            },
        ],
        "historical_iterations": [],
    }
    new_claims = [
        {"claim_text": "GDP grew 5%", "verdict": "SUPPORTED"},
        {"claim_text": "New claim", "verdict": "UNSUPPORTED"},
    ]
    mapping = {0: "a"}
    delta = {"resolved": ["a"], "regressed": [], "unchanged_failed": []}

    updated = update_ledger(ledger, new_claims, mapping, delta)

    assert updated["current_iteration"] == 1
    assert len(updated["active_claims"]) == 1
    assert updated["active_claims"][0]["text"] == "New claim"
    assert updated["active_claims"][0]["first_seen_iteration"] == 1

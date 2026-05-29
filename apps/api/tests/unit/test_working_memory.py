import pytest
from unittest.mock import MagicMock

from app.workers.orchestrator import _update_epistemic_ledger
from app.workers.formatters import _build_hedge_block


@pytest.mark.unit
async def test_update_epistemic_ledger_derives_weak_passes():
    job = MagicMock()
    job.working_memory = {}
    claims_data = [
        {
            "claim_text": "Strong claim",
            "verdict": "SUPPORTED",
            "confidence": 0.95,
            "evidence_text": "Solid source",
        },
        {
            "claim_text": "Borderline claim",
            "verdict": "SUPPORTED",
            "confidence": 0.6,
            "evidence_text": "Weak source",
        },
        {
            "claim_text": "Uncertain claim",
            "verdict": "UNCERTAIN",
            "confidence": 0.4,
            "evidence_text": "No data",
        },
        {
            "claim_text": "Contested claim",
            "verdict": "CONTESTED",
            "confidence": 0.5,
            "evidence_text": "Disputed",
        },
        {
            "claim_text": "Unsupported claim",
            "verdict": "UNSUPPORTED",
            "confidence": 0.3,
            "evidence_text": "",
        },
    ]

    await _update_epistemic_ledger(job, claims_data)

    weak_passes = job.working_memory["epistemic_ledger"]["weak_passes"]
    weak_texts = {w["claim_text"] for w in weak_passes}
    assert "Borderline claim" in weak_texts
    assert "Uncertain claim" in weak_texts
    assert "Contested claim" in weak_texts
    assert "Strong claim" not in weak_texts
    assert "Unsupported claim" not in weak_texts


@pytest.mark.unit
async def test_update_epistemic_ledger_empty_input():
    job = MagicMock()
    job.working_memory = {}

    await _update_epistemic_ledger(job, [])

    assert job.working_memory == {}


@pytest.mark.unit
def test_build_hedge_block_appends_weakness_details():
    hedge_index = [
        {"claim_text": "GDP grew 3.2%", "verdict": "UNCERTAIN"},
    ]
    epistemic_ledger = {
        "weak_passes": [
            {
                "claim_text": "GDP grew 3.2%",
                "verdict": "UNCERTAIN",
                "confidence": 0.4,
                "weakness_reason": "Limited data available",
            }
        ],
    }

    result = _build_hedge_block(hedge_index, epistemic_ledger)

    assert "Weakness details:" in result
    assert "Limited data available" in result


@pytest.mark.unit
def test_build_hedge_block_no_epistemic_ledger():
    hedge_index = [
        {"claim_text": "GDP grew 3.2%", "verdict": "UNCERTAIN"},
    ]

    result = _build_hedge_block(hedge_index)

    assert "Weakness details:" not in result


@pytest.mark.unit
def test_build_hedge_block_empty_hedge_index():
    result = _build_hedge_block([], {"weak_passes": []})
    assert result == ""

    result = _build_hedge_block([])
    assert result == ""

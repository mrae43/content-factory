"""
Eval 1.3 — topic_relevance Calibration.

Reference-based eval comparing pipeline's auto-assigned topic_relevance
labels against human-labelled ground truth for a held-out set of 50 chunks.

Golden-only (no --live mode). Confusion matrix computed from
_derive_topic_relevance(similarity_score) vs human_label.
"""

import pytest

from app.services.context_builder import _derive_topic_relevance
from tests.evals.schemas import RelevancyCalibrationSet


@pytest.mark.eval
async def test_topic_relevance_calibration(
    calibration_set: RelevancyCalibrationSet,
    baseline_recorder,
):
    chunks = calibration_set.chunks

    # Build confusion matrix: predicted vs human
    tp_h = tn_h = fp_h = fn_h = 0  # HIGH positives
    tp_l = tn_l = fp_l = fn_l = 0  # LOW positives
    correct = 0

    for chunk in chunks:
        predicted = _derive_topic_relevance(chunk.similarity_score)
        actual = chunk.human_label

        if predicted == actual:
            correct += 1

        # HIGH confusion
        if predicted == "HIGH" and actual == "HIGH":
            tp_h += 1
        elif predicted == "HIGH" and actual != "HIGH":
            fp_h += 1
        elif predicted != "HIGH" and actual == "HIGH":
            fn_h += 1
        else:
            tn_h += 1

        # LOW confusion
        if predicted == "LOW" and actual == "LOW":
            tp_l += 1
        elif predicted == "LOW" and actual != "LOW":
            fp_l += 1
        elif predicted != "LOW" and actual == "LOW":
            fn_l += 1
        else:
            tn_l += 1

    n = len(chunks)
    accuracy = correct / n
    high_precision = tp_h / (tp_h + fp_h) if (tp_h + fp_h) > 0 else 0.0
    low_recall = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0

    violations = []
    if accuracy < 0.80:
        violations.append(f"accuracy {accuracy:.2f} < 0.80")
    if high_precision < 0.85:
        violations.append(f"HIGH precision {high_precision:.2f} < 0.85")
    if low_recall < 0.75:
        violations.append(f"LOW recall {low_recall:.2f} < 0.75")

    baseline_recorder.record_case_score(
        "topic_relevance_calibration",
        "calibration",
        {
            "accuracy": accuracy,
            "high_precision": high_precision,
            "low_recall": low_recall,
            "n": n,
            "confusion_matrix": {
                "high": {"tp": tp_h, "fp": fp_h, "fn": fn_h, "tn": tn_h},
                "low": {"tp": tp_l, "fp": fp_l, "fn": fn_l, "tn": tn_l},
            },
        },
    )

    if violations:
        pytest.fail(f"Calibration failed: {'; '.join(violations)}")

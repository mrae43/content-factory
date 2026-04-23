"""
Scoring rubrics for LLM-as-Judge evaluation.

Each rubric defines:
  - dimensions: ordered list of (name, description) tuples
  - weights: {dimension_name: float} — must sum to 1.0
  - score_levels: mapping of score → description for LLM prompt

Source: EVALS_TEST_POLICY.md §4, GOLDEN_DATASET_FOUNDATION.md §6.2
"""

from typing import Dict, List, Tuple


# ==========================================
# SCORE LEVELS (shared across all rubrics)
# ==========================================

SCORE_LEVELS: Dict[float, str] = {
    0.0: "Unacceptable — does not meet any criteria for this dimension",
    0.5: "Borderline — partially meets criteria, significant room for improvement",
    1.0: "Excellent — fully meets all criteria for this dimension",
}


# ==========================================
# HELPER: build a rubric dict
# ==========================================


def _build_rubric(
    name: str,
    dimensions: List[Tuple[str, str]],
    weights: Dict[str, float],
    score_levels: Dict[float, str] = SCORE_LEVELS,
) -> Dict:
    return {
        "name": name,
        "dimensions": dimensions,
        "weights": weights,
        "score_levels": score_levels,
    }


# ==========================================
# 1. RESEARCH QUALITY RUBRIC
# ==========================================

RESEARCH_RUBRIC = _build_rubric(
    name="research",
    dimensions=[
        (
            "completeness",
            "Are all key facts from the input captured in the output? "
            "Check against must_include_facts in the golden case.",
        ),
        (
            "accuracy",
            "Is the output free of fabrications and unsupported claims? "
            "Compare against source material — no invented statistics, dates, or attributions.",
        ),
        (
            "synthesis",
            "Does the output integrate information into a coherent narrative, "
            "or is it a raw dump of chunk text? Look for connecting reasoning and summaries.",
        ),
        (
            "confidence_calibration",
            "Is the confidence_score well-calibrated? "
            "High confidence should mean high accuracy; low confidence for uncertain areas.",
        ),
    ],
    weights={
        "completeness": 0.30,
        "accuracy": 0.30,
        "synthesis": 0.20,
        "confidence_calibration": 0.20,
    },
)


# ==========================================
# 2. SCRIPT QUALITY RUBRIC
# ==========================================

SCRIPT_RUBRIC = _build_rubric(
    name="script",
    dimensions=[
        (
            "factual_grounding",
            "Is the script fully grounded in refined_context? "
            "No hallucinations or claims that don't trace back to research output.",
        ),
        (
            "narrative_structure",
            "Does the script have a clear Hook-Value-Loop structure? "
            "Hook grabs attention in first 3 seconds, Value delivers content, Loop has CTA.",
        ),
        (
            "engagement",
            "Is the pacing high-retention? Short sentences, active voice, "
            "visual language. Not dry or academic.",
        ),
        (
            "storyboard_quality",
            "Are storyboard items precise? Each scene should have clear "
            "visual_prompt and audio_cue. Vague cues like 'some imagery' score low.",
        ),
        (
            "length",
            "Is the script in the target word count range (150-500 words)? "
            "Corresponds to ~60-180 seconds of narration. Outside range scores 0.",
        ),
    ],
    weights={
        "factual_grounding": 0.35,
        "narrative_structure": 0.20,
        "engagement": 0.15,
        "storyboard_quality": 0.15,
        "length": 0.15,
    },
)


# ==========================================
# 3. FACT-CHECK QUALITY RUBRIC
# ==========================================

FACT_CHECK_RUBRIC = _build_rubric(
    name="fact_check",
    dimensions=[
        (
            "claim_coverage",
            "Were all atomic factual claims extracted? "
            "Compare extracted claims against the full script — missing major claims scores 0.",
        ),
        (
            "verdict_accuracy",
            "Are the verdicts correct? SUPPORTED for well-grounded claims, "
            "UNSUPPORTED for fabrications, CONTESTED for conflicting evidence. "
            "Compare against claims_with_known_verdicts in the golden case.",
        ),
        (
            "evidence_quality",
            "Is the evidence specific and traceable? Strong evidence references "
            "actual source content. Weak evidence is vague or generic.",
        ),
        (
            "confidence_calibration",
            "Is confidence well-calibrated per claim? "
            "High confidence on wrong verdicts is especially penalized.",
        ),
    ],
    weights={
        "claim_coverage": 0.30,
        "verdict_accuracy": 0.40,
        "evidence_quality": 0.20,
        "confidence_calibration": 0.10,
    },
)


# ==========================================
# 4. OPTIMIZATION QUALITY RUBRIC
# ==========================================

OPTIMIZER_RUBRIC = _build_rubric(
    name="optimizer",
    dimensions=[
        (
            "patch_precision",
            "Were only the failed claims modified? The optimizer should not "
            "rewrite unrelated sections. Compare before/after — only failed claims should change.",
        ),
        (
            "narrative_preservation",
            "Does the script flow coherently after patching? "
            "No orphaned references, no jarring transitions, no broken narrative.",
        ),
        (
            "grounding",
            "Are the patched claims grounded in refined_context? "
            "No new hallucinations introduced. Patches must use available evidence.",
        ),
        (
            "claim_resolution",
            "Were the failed claims actually resolved? "
            "UNSUPPORTED claims should be fixed or removed. CONTESTED claims should be qualified.",
        ),
    ],
    weights={
        "patch_precision": 0.35,
        "narrative_preservation": 0.25,
        "grounding": 0.25,
        "claim_resolution": 0.15,
    },
)


# ==========================================
# RUBRIC REGISTRY
# ==========================================

RUBRICS = {
    "research": RESEARCH_RUBRIC,
    "script": SCRIPT_RUBRIC,
    "fact_check": FACT_CHECK_RUBRIC,
    "optimizer": OPTIMIZER_RUBRIC,
}


def get_rubric(name: str) -> Dict:
    return RUBRICS[name]


def compute_weighted_score(
    rubric_name: str, dimension_scores: Dict[str, float]
) -> float:
    rubric = RUBRICS[rubric_name]
    total = 0.0
    for dim, weight in rubric["weights"].items():
        total += dimension_scores.get(dim, 0.0) * weight
    return round(total, 4)


def format_rubric_for_prompt(rubric_name: str) -> str:
    rubric = RUBRICS[rubric_name]
    lines = [f"## {rubric['name'].upper()} RUBRIC\n"]
    for dim_name, dim_desc in rubric["dimensions"]:
        weight = rubric["weights"][dim_name]
        lines.append(f"### {dim_name} (weight: {weight:.0%})")
        lines.append(f"{dim_desc}\n")
        lines.append("Score levels:")
        for score, desc in sorted(rubric["score_levels"].items()):
            lines.append(f"  {score}: {desc}")
        lines.append("")
    return "\n".join(lines)

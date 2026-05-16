from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class GuardrailConfig:
    similarity_threshold: float = 0.75
    top_k_per_claim: int = 5
    uncertain_is_soft_fail: bool = False
    extract_categories: List[str] = field(
        default_factory=lambda: [
            "statistic",
            "attribution",
            "chronological",
            "causal",
            "comparative",
        ]
    )
    max_revision_cycles: int = 3


GUARDRAIL_PROFILES = {
    "Low": GuardrailConfig(
        similarity_threshold=0.65,
        top_k_per_claim=3,
        uncertain_is_soft_fail=False,
        extract_categories=["statistic", "attribution"],
        max_revision_cycles=2,
    ),
    "Medium": GuardrailConfig(
        similarity_threshold=0.72,
        top_k_per_claim=5,
        uncertain_is_soft_fail=False,
        extract_categories=[
            "statistic",
            "attribution",
            "chronological",
            "causal",
        ],
        max_revision_cycles=3,
    ),
    "High": GuardrailConfig(
        similarity_threshold=0.75,
        top_k_per_claim=5,
        uncertain_is_soft_fail=False,
        extract_categories=[
            "statistic",
            "attribution",
            "chronological",
            "causal",
            "comparative",
        ],
        max_revision_cycles=3,
    ),
}


def get_guardrail_config(
    strictness: str, strict_compliance_mode: bool
) -> GuardrailConfig:
    base = GUARDRAIL_PROFILES.get(strictness, GUARDRAIL_PROFILES["High"])
    if strict_compliance_mode:
        return GuardrailConfig(
            similarity_threshold=base.similarity_threshold,
            top_k_per_claim=base.top_k_per_claim,
            uncertain_is_soft_fail=True,
            extract_categories=base.extract_categories,
            max_revision_cycles=base.max_revision_cycles,
        )
    return base

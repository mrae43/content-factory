import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Tuple
import warnings


class GuardrailStrictness(str, Enum):
    Low = "Low"
    Medium = "Medium"
    High = "High"


@dataclass(frozen=True)
class GuardrailConfig:
    similarity_threshold: float = 0.75
    top_k_per_claim: int = 5
    uncertain_is_soft_fail: bool = False
    requires_human_review: bool = False
    claim_categories: Tuple[str, ...] = (
        "statistic",
        "attribution",
        "chronological",
        "causal",
        "comparative",
    )


GUARDRAIL_PROFILES: dict[GuardrailStrictness, GuardrailConfig] = {
    GuardrailStrictness.Low: GuardrailConfig(
        similarity_threshold=0.65,
        top_k_per_claim=3,
        uncertain_is_soft_fail=False,
        requires_human_review=False,
        claim_categories=("statistic", "attribution"),
    ),
    GuardrailStrictness.Medium: GuardrailConfig(
        similarity_threshold=0.72,
        top_k_per_claim=5,
        uncertain_is_soft_fail=False,
        requires_human_review=False,
        claim_categories=(
            "statistic",
            "attribution",
            "chronological",
            "causal",
        ),
    ),
    GuardrailStrictness.High: GuardrailConfig(
        similarity_threshold=0.75,
        top_k_per_claim=5,
        uncertain_is_soft_fail=False,
        requires_human_review=True,
        claim_categories=(
            "statistic",
            "attribution",
            "chronological",
            "causal",
            "comparative",
        ),
    ),
}


def get_guardrail_config(
    strictness: GuardrailStrictness, uncertain_pass_through: bool = False
) -> GuardrailConfig:
    if strictness not in GUARDRAIL_PROFILES:
        raise ValueError(f"Unknown guardrail strictness: {strictness}")

    base = GUARDRAIL_PROFILES[strictness]

    if strictness == GuardrailStrictness.High:
        if uncertain_pass_through:
            return base
        return _replace(base, uncertain_is_soft_fail=True)

    if uncertain_pass_through:
        warnings.warn(
            f"uncertain_pass_through=True has no effect for {strictness.value} profile. "
            f"Only High profile supports uncertain_pass_through.",
            stacklevel=2,
        )

    return base


def _replace(config: GuardrailConfig, **kwargs) -> GuardrailConfig:
    return dataclasses.replace(config, **kwargs)

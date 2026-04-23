"""
LLM-as-Judge scoring harness.

Wires judge_llm + rubric + agent output → structured dimension scores.
All outcome eval test files import judge_score() from this module.
"""

from typing import Any, List

from pydantic import BaseModel, Field

from tests.evals.rubrics import compute_weighted_score, format_rubric_for_prompt


JUDGE_PROMPT_TEMPLATE = """<role>
You are a senior content quality auditor specializing in AI-generated short-form video content.
You evaluate agent outputs against strict rubrics with evidence-based scoring.
You are impartial, methodical, and never inflate scores.
</role>

<task>
Evaluate the agent output below against the provided rubric. Score each dimension independently, then produce a structured verdict.
</task>

<rubric>
{rubric}
</rubric>

<input>
{input}
</input>

<output>
{output}
</output>

<reference>
{reference}
</reference>

<evaluation_protocol>
Follow these steps IN ORDER:

Step 1 — COMPREHENSION
Read the <input>, <output>, and <reference> sections carefully. Identify what the agent was asked to do and what it produced.

Step 2 — DIMENSION-BY-DIMENSION SCORING
For EACH dimension in the rubric:
  a) Re-read the dimension criteria and score level definitions (0.0, 0.5, 1.0).
  b) Locate concrete evidence in the <output> that supports or contradicts the criteria.
  c) Compare the <output> against the <reference> to calibrate your score. If <reference> is "N/A", rely on rubric criteria alone.
  d) Assign a score (0.0, 0.5, or 1.0) and write a 1-2 sentence justification quoting specific parts of the output.

Step 3 — REASONING SYNTHESIS
Write a brief overall assessment (2-4 sentences) summarizing strengths and the most critical weaknesses. Reference specific dimensions by name.

Step 4 — STRUCTURED OUTPUT
Return your evaluation as JSON matching the JudgeResult schema:
{{
  "dimensions": [
    {{"dimension": "<name>", "score": <0.0|0.5|1.0>, "evidence": "<justification with quotes>"}}
  ],
  "weighted_average": <computed>,
  "reasoning": "<overall assessment>"
}}
</evaluation_protocol>

<constraints>
- Score each dimension independently. Do not let one dimension bleed into another.
- Only use score values 0.0, 0.5, or 1.0. Intermediate values like 0.7 are invalid.
- Every evidence field must reference a specific quote or paraphrase from the output. Generic justifications such as "the output is good" are invalid.
- Do not inflate scores. An output that partially meets criteria earns 0.5, not 1.0.
</constraints>
"""


class JudgeDimensionScore(BaseModel):
    dimension: str
    score: float = Field(ge=0.0, le=1.0)
    evidence: str


class JudgeResult(BaseModel):
    dimensions: List[JudgeDimensionScore]
    weighted_average: float = Field(ge=0.0, le=1.0)
    reasoning: str


async def judge_score(
    judge_llm,
    rubric_name: str,
    agent_input: Any,
    agent_output: Any,
    reference: Any,
) -> JudgeResult:
    rubric_text = format_rubric_for_prompt(rubric_name)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        rubric=rubric_text,
        input=_serialize(agent_input),
        output=_serialize(agent_output),
        reference=_serialize(reference),
    )
    chain = judge_llm.with_structured_output(JudgeResult)
    result: JudgeResult = await chain.ainvoke(prompt)
    validated_average = compute_weighted_score(
        rubric_name,
        {d.dimension: d.score for d in result.dimensions},
    )
    result.weighted_average = validated_average
    return result


def _serialize(obj: Any) -> str:
    if obj is None:
        return "N/A"
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "model_dump_json"):
        return obj.model_dump_json(indent=2)
    if isinstance(obj, (dict, list)):
        import json

        return json.dumps(obj, indent=2, ensure_ascii=False, default=str)
    return str(obj)

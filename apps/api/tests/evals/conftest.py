"""
Eval infrastructure fixtures for the Golden Dataset evaluation framework.

Provides: judge_llm, golden_dataset, golden_case, eval_runner, score_aggregator,
          trace_capture, baseline_recorder, rubric_registry.

Depends on: tests/evals/schemas.py, tests/evals/rubrics.py,
            app/services/llm.py, app/workers/agents.py, app/workers/orchestrator.py
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import pytest

from app.workers.agents import (
    AgentResult,
    ResearchAgent,
    CopywriterAgent,
    RedTeamAgent,
)
from app.workers.optimizer import ScriptOptimizerAgent
from tests.evals.rubrics import (
    RUBRICS,
    compute_weighted_score,
)
from tests.evals.schemas import GoldenCase, GoldenDataset


_GOLDEN_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "golden" / "golden_dataset.json"
)
_BASELINES_PATH = Path(__file__).resolve().parent / "baselines.json"


def _load_golden_dataset() -> List[GoldenCase]:
    if not _GOLDEN_DATASET_PATH.exists():
        return []
    raw = json.loads(_GOLDEN_DATASET_PATH.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        dataset = GoldenDataset(cases=raw)
    elif isinstance(raw, dict) and "cases" in raw:
        dataset = GoldenDataset(**raw)
    else:
        raise ValueError(f"Unexpected golden_dataset.json structure: {type(raw)}")
    return dataset.cases


# ==========================================
# 1. JUDGE LLM
# ==========================================


@pytest.fixture(scope="session")
def judge_llm():
    """
    LLM-as-Judge instance: separate from pipeline LLMs.
    Uses gemini-2.5-flash (fast, cheap, different from evaluator models).
    """
    from app.services.llm import get_llm

    return get_llm(model_name="gemini-2.5-flash", temperature=0.0)


# ==========================================
# 2. GOLDEN DATASET
# ==========================================


@pytest.fixture(scope="session")
def golden_dataset() -> List[GoldenCase]:
    """Loads and validates golden_dataset.json into a list of GoldenCase models."""
    return _load_golden_dataset()


# ==========================================
# 3. GOLDEN CASE (parametrized by ID)
# ==========================================


@pytest.fixture
def golden_case(request, golden_dataset: List[GoldenCase]) -> GoldenCase:
    """
    Parametrized fixture yielding a single GoldenCase by ID.
    Use via indirect parametrization:
        @pytest.mark.parametrize("golden_case", ["H-001"], indirect=True)
    """
    case_id = getattr(request, "param", None)
    if case_id is None:
        raise ValueError(
            "golden_case fixture requires indirect parametrization with a case ID"
        )
    for case in golden_dataset:
        if case.id == case_id:
            return case
    available = [c.id for c in golden_dataset]
    raise ValueError(f"Golden case '{case_id}' not found. Available: {available}")


# ==========================================
# 4. EVAL RUNNER
# ==========================================


class EvalRunner:
    """
    Runs agents against golden case inputs and captures all intermediate outputs.

    In integration mode (real LLM calls): uses real agents with a real vector store.
    In unit/mocked mode: injects mock vector store returning golden case chunks.
    """

    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self.outputs: Dict[str, Any] = {}

    async def run_research(self, case: GoldenCase, vector_store=None) -> AgentResult:
        vs = vector_store or self.vector_store
        agent = ResearchAgent(model_name="gemini-2.5-flash")
        context = {
            "job_id": uuid4(),
            "topic": case.input.topic,
            "vector_store": vs,
        }
        result = await agent.run(context=context)
        self.outputs["research"] = result
        return result

    async def run_copywriter(
        self, case: GoldenCase, refined_context: str, feedback: str = ""
    ) -> AgentResult:
        agent = CopywriterAgent(model_name="gemini-1.5-pro", temperature=0.7)
        context = {
            "job_id": uuid4(),
            "topic": case.input.topic,
            "refined_context": refined_context,
            "feedback": feedback,
        }
        result = await agent.run(context=context)
        self.outputs["script"] = result
        return result

    async def run_optimizer(
        self,
        script_content: str,
        refined_context: str,
        failed_claims: List[dict],
    ) -> AgentResult:
        from app.core.config import settings

        agent = ScriptOptimizerAgent(
            model_name=settings.optimizer_model,
            temperature=settings.optimizer_temperature,
        )
        context = {
            "job_id": uuid4(),
            "script_content": script_content,
            "failed_claims": failed_claims,
            "refined_context": refined_context,
        }
        result = await agent.run(context=context)
        self.outputs["optimizer"] = result
        return result

    async def run_red_team(
        self,
        script_content: str,
        vector_store=None,
    ) -> AgentResult:
        vs = vector_store or self.vector_store
        agent = RedTeamAgent(model_name="gemini-1.5-pro", temperature=0.0)
        context = {
            "job_id": uuid4(),
            "script_content": script_content,
            "vector_store": vs,
        }
        result = await agent.run(context=context)
        self.outputs["fact_check"] = result
        return result


@pytest.fixture
def eval_runner(mock_vector_store) -> EvalRunner:
    """EvalRunner with a mock vector store for unit-level tests."""
    return EvalRunner(vector_store=mock_vector_store)


# ==========================================
# 5. SCORE AGGREGATOR
# ==========================================


class ScoreAggregator:
    """
    Collects deterministic + LLM-as-Judge scores and computes weighted averages.
    """

    def __init__(self):
        self.scores: Dict[str, Dict[str, float]] = {}

    def record(self, rubric_name: str, dimension_scores: Dict[str, float]) -> float:
        self.scores[rubric_name] = dimension_scores
        return compute_weighted_score(rubric_name, dimension_scores)

    def get_weighted_average(self, rubric_name: str) -> float:
        if rubric_name not in self.scores:
            return 0.0
        return compute_weighted_score(rubric_name, self.scores[rubric_name])

    def get_dimension(self, rubric_name: str, dimension: str) -> float:
        return self.scores.get(rubric_name, {}).get(dimension, 0.0)

    def all_averages(self) -> Dict[str, float]:
        return {
            name: compute_weighted_score(name, dims)
            for name, dims in self.scores.items()
        }

    def check_threshold(self, rubric_name: str, threshold: float) -> bool:
        return self.get_weighted_average(rubric_name) >= threshold

    def check_dimension_threshold(
        self, rubric_name: str, dimension: str, threshold: float
    ) -> bool:
        return self.get_dimension(rubric_name, dimension) >= threshold


@pytest.fixture
def score_aggregator() -> ScoreAggregator:
    return ScoreAggregator()


# ==========================================
# 6. TRACE CAPTURE
# ==========================================


class TraceCapture:
    """
    Wraps agent execution to record tool calls, state transitions,
    and intermediate results without modifying production code.

    Uses mock-based instrumentation (Foundation doc §8.1, Option 1).
    """

    def __init__(self):
        self.state_transitions: List[Dict[str, Any]] = []
        self.agent_calls: List[Dict[str, Any]] = []
        self.tool_calls: List[Dict[str, Any]] = []

    def wrap_agent(self, agent_class):
        """Returns an async wrapper that records inputs/outputs around the real agent.run()."""
        capture = self
        original_run = agent_class.run

        async def traced_run(context, **kwargs):
            capture.agent_calls.append(
                {
                    "agent": agent_class.__name__,
                    "input_keys": sorted(context.keys()),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            result = await original_run(context, **kwargs)
            capture.agent_calls[-1]["output_status"] = result.status.value
            capture.agent_calls[-1]["confidence_score"] = result.confidence_score
            return result

        return traced_run

    def wrap_vector_store(self, vector_store):
        """Wraps vector_store.semantic_search to record tool call arguments."""
        capture = self
        original_search = vector_store.semantic_search

        async def traced_search(query, **kwargs):
            capture.tool_calls.append(
                {
                    "tool": "semantic_search",
                    "query": query,
                    "kwargs": kwargs,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
            return await original_search(query, **kwargs)

        vector_store.semantic_search = traced_search
        return vector_store

    def record_state_transition(self, from_state: str, to_state: str):
        self.state_transitions.append(
            {
                "from": from_state,
                "to": to_state,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def get_state_sequence(self) -> List[str]:
        if not self.state_transitions:
            return []
        seq = [self.state_transitions[0]["from"]]
        for t in self.state_transitions:
            seq.append(t["to"])
        return seq

    def get_agent_call_sequence(self) -> List[Dict[str, Any]]:
        return [
            {
                "agent": call["agent"],
                "status": call["output_status"],
            }
            for call in self.agent_calls
        ]

    def get_tool_call_queries(self, tool_name: str = "semantic_search") -> List[str]:
        return [call["query"] for call in self.tool_calls if call["tool"] == tool_name]

    def reset(self):
        self.state_transitions.clear()
        self.agent_calls.clear()
        self.tool_calls.clear()


@pytest.fixture
def trace_capture() -> TraceCapture:
    return TraceCapture()


# ==========================================
# 7. BASELINE RECORDER
# ==========================================


class BaselineRecorder:
    """
    Reads/writes baselines.json, compares current scores vs. recorded baselines.
    Supports --update-baselines CLI flag.
    """

    REGRESSION_THRESHOLD = 0.05

    def __init__(self, path: Path = _BASELINES_PATH):
        self.path = path
        self._data: Optional[Dict[str, Any]] = None

    @property
    def data(self) -> Dict[str, Any]:
        if self._data is None:
            self._data = self._load()
        return self._data

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"last_updated": None, "cases": {}, "summary": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self):
        self.data["last_updated"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def get_case_baseline(self, case_id: str) -> Optional[Dict[str, Any]]:
        return self.data.get("cases", {}).get(case_id)

    def record_case_score(self, case_id: str, rubric_name: str, scores: Dict[str, Any]):
        cases = self.data.setdefault("cases", {})
        case_entry = cases.setdefault(case_id, {})
        case_entry[rubric_name] = scores

    def check_regression(
        self, case_id: str, rubric_name: str, current_score: float
    ) -> Optional[str]:
        baseline = self.get_case_baseline(case_id)
        if baseline is None or rubric_name not in baseline:
            return None
        baseline_score = baseline[rubric_name].get("weighted_average")
        if baseline_score is None:
            return None
        if current_score < baseline_score - self.REGRESSION_THRESHOLD:
            return (
                f"REGRESSION: {case_id}/{rubric_name} score {current_score:.4f} "
                f"dropped >{self.REGRESSION_THRESHOLD} from baseline {baseline_score:.4f}"
            )
        return None

    def update_baselines(self, scores: Dict[str, Dict[str, Dict[str, Any]]]):
        for case_id, rubric_scores in scores.items():
            for rubric_name, dims in rubric_scores.items():
                self.record_case_score(case_id, rubric_name, dims)
        self._compute_summary()
        self.save()

    def _compute_summary(self):
        cases = self.data.get("cases", {})
        if not cases:
            return
        all_averages = []
        by_category: Dict[str, List[float]] = {}
        for case_id, rubric_data in cases.items():
            for rubric_name, scores in rubric_data.items():
                avg = scores.get("weighted_average", 0.0)
                all_averages.append(avg)
                by_category.setdefault(rubric_name, []).append(avg)
        summary = self.data.setdefault("summary", {})
        summary["overall_avg"] = (
            round(sum(all_averages) / len(all_averages), 4) if all_averages else 0.0
        )
        summary["by_category"] = {
            cat: round(sum(vals) / len(vals), 4) for cat, vals in by_category.items()
        }
        summary["regression_threshold"] = self.REGRESSION_THRESHOLD


@pytest.fixture
def baseline_recorder() -> BaselineRecorder:
    return BaselineRecorder()


# ==========================================
# 8. RUBRIC REGISTRY
# ==========================================


@pytest.fixture(scope="session")
def rubric_registry() -> Dict[str, Dict]:
    """Maps rubric names to rubric definitions from rubrics.py."""
    return RUBRICS.copy()


# ==========================================
# HELPER: Negative golden PII checker
# ==========================================


def check_pii_propagation(outputs: Dict[str, Any], pii_entities: List[str]) -> bool:
    """
    Fuzzy-matches PII entities against all stage outputs.
    Returns True if NO PII found (pass), False if PII leaked (fail).
    """
    ssn_pattern = re.compile(r"\d{3}-\d{2}-\d{4}")
    for _stage_name, stage_output in outputs.items():
        output_text = str(stage_output)
        if ssn_pattern.search(output_text):
            return False
        for entity in pii_entities:
            if entity.lower() in output_text.lower():
                return False
    return True

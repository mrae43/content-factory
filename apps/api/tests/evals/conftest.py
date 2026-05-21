"""
Eval infrastructure fixtures for the Golden Dataset evaluation framework.

Provides: judge_llm, golden_dataset, golden_case, eval_runner, score_aggregator,
          trace_capture, baseline_recorder, rubric_registry.

Supports two modes:
  - "golden" (default): Uses pre-recorded reference_outputs from golden_dataset.json
    for deterministic, fast, free eval scoring. No API calls needed.
  - "live": Runs real agents with LLM calls. Use --live flag to refresh golden outputs.

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
    AgentActionStatus,
    AgentResult,
    ResearchAgent,
    CopywriterAgent,
    RedTeamAgent,
)
from app.workers.optimizer import ScriptOptimizerAgent
from app.core.config import settings
from app.services.web_search import TavilySearchService
from tests.evals.judge import judge_score as _judge_score
from tests.evals.rubrics import (
    RUBRICS,
    compute_weighted_score,
)
from tests.evals.schemas import (
    GoldenCase,
    GoldenDataset,
    QualityCorpus,
    QualityCorpusEntry,
    ResearchingCase,
)
from tests.evals.chunk_quality_scorer import ChunkQualityScorer


def pytest_addoption(parser):
    parser.addoption(
        "--update-baselines",
        action="store_true",
        default=False,
        help="Write eval scores to baselines.json after the session completes.",
    )
    parser.addoption(
        "--live",
        action="store_true",
        default=False,
        help="Run live agent calls instead of using golden reference outputs.",
    )
    parser.addoption(
        "--update-cache",
        action="store_true",
        default=False,
        help="Update cached_judge_response in fixture from baselines.json",
    )


_GOLDEN_DATASET_PATH = (
    Path(__file__).resolve().parent.parent / "golden" / "golden_dataset.json"
)
_BASELINES_PATH = Path(__file__).resolve().parent / "baselines.json"
_EVAL1_RESEARCH_FIXTURES_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "eval1_research.json"
)


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


def _load_eval1_research() -> dict:
    if not _EVAL1_RESEARCH_FIXTURES_PATH.exists():
        return {"coverage_cases": [], "quality_corpus": None, "relevance_cases": []}
    return json.loads(_EVAL1_RESEARCH_FIXTURES_PATH.read_text(encoding="utf-8"))


def _load_researching_cases() -> List[ResearchingCase]:
    data = _load_eval1_research()
    raw_cases = data.get("coverage_cases", [])
    return [ResearchingCase(**item) for item in raw_cases]


def _is_live_mode(request) -> bool:
    return request.config.getoption("live", default=False)


# ==========================================
# 1. JUDGE LLM
# ==========================================


@pytest.fixture(scope="session")
def judge_llm():
    """
    LLM-as-Judge instance: separate from pipeline LLMs.
    Uses Qwen3 235B-A22B via Together AI (strong reasoning, MoE keeps cost low).
    """
    from app.services.llm import get_llm

    return get_llm(
        model_name=settings.eval_judge_model,
        temperature=settings.eval_judge_temperature,
    )


# ==========================================
# 1b. JUDGE SCORER (wraps judge_llm)
# ==========================================


@pytest.fixture
def judge_scorer(judge_llm):
    """
    Callable that wraps judge_score() with the session-scoped judge_llm.
    Usage: result = await judge_scorer("research", input, output, reference)
    """

    async def _score(rubric_name, agent_input, agent_output, reference):
        return await _judge_score(
            judge_llm, rubric_name, agent_input, agent_output, reference
        )

    return _score


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

    In "golden" mode (default): returns pre-recorded reference_outputs as
    AgentResult objects — no API calls, deterministic, fast.

    In "live" mode (--live flag): runs real agents with actual LLM calls.
    Use live mode to refresh reference_outputs, not for CI.
    """

    def __init__(self, vector_store=None, live: bool = False):
        self.vector_store = vector_store
        self.live = live
        self.outputs: Dict[str, Any] = {}

    def _golden_result(self, stage: str, case: GoldenCase) -> Optional[AgentResult]:
        ref = case.reference_outputs
        if ref is None:
            return None

        stage_ref = getattr(ref, stage, None)
        if stage_ref is None:
            return None

        status_map = {
            "research": AgentActionStatus.SUCCESS,
            "script": AgentActionStatus.SUCCESS,
            "fact_check": None,
            "optimizer": AgentActionStatus.SUCCESS,
        }

        if stage == "fact_check":
            verdict = stage_ref.verdict
            status = (
                AgentActionStatus.SUCCESS
                if verdict == "SUPPORTED"
                else AgentActionStatus.REVISION_NEEDED
            )
        else:
            status = status_map.get(stage, AgentActionStatus.SUCCESS)

        payload = stage_ref.model_dump()
        return AgentResult(
            status=status,
            payload=payload,
            reasoning=stage_ref.overall_reasoning
            if hasattr(stage_ref, "overall_reasoning") and stage_ref.overall_reasoning
            else f"Golden reference output for {stage}",
            confidence_score=0.85,
            metadata={"source": "golden_reference"},
        )

    async def run_research(self, case: GoldenCase, vector_store=None) -> AgentResult:
        if not self.live:
            golden = self._golden_result("research", case)
            if golden is not None:
                self.outputs["research"] = golden
                return golden

        vs = vector_store or self.vector_store
        agent = ResearchAgent(
            model_name=settings.eval_research_model,
            temperature=settings.eval_research_temperature,
        )
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
        if not self.live:
            golden = self._golden_result("script", case)
            if golden is not None:
                self.outputs["script"] = golden
                return golden

        agent = CopywriterAgent(
            model_name=settings.eval_copywriter_model,
            temperature=settings.eval_copywriter_temperature,
        )
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
        agent = ScriptOptimizerAgent(
            model_name=settings.eval_optimizer_model,
            temperature=settings.eval_optimizer_temperature,
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
        case: GoldenCase = None,
    ) -> AgentResult:
        if not self.live and case is not None:
            golden = self._golden_result("fact_check", case)
            if golden is not None:
                self.outputs["fact_check"] = golden
                return golden

        vs = vector_store or self.vector_store
        agent = RedTeamAgent(
            model_name=settings.eval_red_team_model,
            temperature=settings.eval_red_team_temperature,
        )
        context = {
            "job_id": uuid4(),
            "script_content": script_content,
            "vector_store": vs,
        }
        result = await agent.run(context=context)
        self.outputs["fact_check"] = result
        return result

    def run_researching(self, case: ResearchingCase) -> dict:
        chunks = []
        for w in case.mock_web_results:
            if case.inject_metadata_errors:
                scope = "RAW-CONTEXT"
                source_type = "USER_UPLOAD"
            else:
                scope = "LOCAL"
                source_type = "WEB_SEARCH"
            chunks.append(
                {
                    "content": w.content,
                    "url": w.url,
                    "scope": scope,
                    "source_type": source_type,
                }
            )
        return {"chunks": chunks}

    async def run_researching_live(self, case: ResearchingCase) -> dict:
        service = TavilySearchService()
        results = await service.search(case.topic)
        chunks = []
        for r in results:
            content = r.get("content", "") or r.get("snippet", "")
            url = r.get("url", "")
            chunks.append(
                {
                    "content": content,
                    "url": url,
                    "scope": "LOCAL",
                    "source_type": "WEB_SEARCH",
                }
            )
        return {"chunks": chunks}


@pytest.fixture
def eval_runner(request, mock_vector_store) -> EvalRunner:
    """EvalRunner with mock vector store. Uses golden references unless --live."""
    live = _is_live_mode(request)
    return EvalRunner(vector_store=mock_vector_store, live=live)


# ==========================================
# 4b. RESEARCHING EVAL FIXTURES
# ==========================================


@pytest.fixture
def researching_runner(request) -> EvalRunner:
    """EvalRunner for research coverage eval (golden mode: mock chunks; live mode: real Tavily)."""
    live = _is_live_mode(request)
    return EvalRunner(live=live)


@pytest.fixture
def researching_case(request) -> ResearchingCase:
    """
    Parametrized fixture yielding a single ResearchingCase by ID.
    Use via indirect parametrization:
        @pytest.mark.parametrize("researching_case", ["coverage-happy"], indirect=True)
    """
    case_id = getattr(request, "param", None)
    if case_id is None:
        raise ValueError(
            "researching_case fixture requires indirect parametrization with a case ID"
        )
    cases = _load_researching_cases()
    for case in cases:
        if case.id == case_id:
            return case
    available = [c.id for c in cases]
    raise ValueError(f"Researching case '{case_id}' not found. Available: {available}")


# ==========================================
# 4c. CHUNK QUALITY FIXTURES (Eval 1.2)
# ==========================================


@pytest.fixture
def quality_corpus() -> QualityCorpus:
    data = _load_eval1_research()
    qc = data.get("quality_corpus")
    if qc is None:
        return QualityCorpus(description="", capture_run_id="", entries=[])
    return QualityCorpus(**qc)


@pytest.fixture
def quality_entry(request, quality_corpus: QualityCorpus) -> QualityCorpusEntry:
    entry_id = request.param
    for entry in quality_corpus.entries:
        slug = entry.topic.lower().replace(" ", "-")[:40]
        if slug == entry_id:
            return entry
    available = [e.topic.lower().replace(" ", "-")[:40] for e in quality_corpus.entries]
    raise ValueError(f"Quality entry '{entry_id}' not found. Available: {available}")


@pytest.fixture
def chunk_quality_scorer(judge_llm) -> ChunkQualityScorer:
    return ChunkQualityScorer(judge_llm=judge_llm)


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


@pytest.fixture(scope="session")
def baseline_recorder(request) -> BaselineRecorder:
    recorder = BaselineRecorder()
    yield recorder
    if request.config.getoption("update_baselines", default=False):
        recorder._compute_summary()
        recorder.save()


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

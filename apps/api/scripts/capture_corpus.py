"""
One-shot corpus construction for Eval 1.2 — Chunk Quality.

For each canonical topic:
  1. Call TavilySearchService.search(topic)
  2. Build source_chunks: [{content, source_url}, ...]
  3. Call LLM judge on each chunk -> cached_responses
  4. Write entry to quality_corpus section in eval1_research.json
  5. Append run record to capture_log.json

Usage:
  uv run python scripts/capture_corpus.py

Requires: TAVILY_API_KEY, GEMINI_API_KEY (or TOGETHER_API_KEY) in env.
Output: tests/evals/fixtures/eval1_research.json (updated), tests/evals/fixtures/capture_log.json (appended).
"""

import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

CANONICAL_TOPICS = [
    {
        "id": "quality-brics",
        "topic": "BRICS De-dollarization and the Shift Away from USD Dominance",
        "description": "Economics — high-density financial content",
    },
    {
        "id": "quality-sparse-FR1",
        "topic": "Quantum Computing Breakthroughs in 2025",
        "description": "Tech/Science — expected sparse/few results (tests density edge case)",
    },
    {
        "id": "quality-boilerplate",
        "topic": "US Federal Reserve Interest Rate Decision Impact on Markets 2025",
        "description": "Finance/Policy — expected boilerplate/low-info results (tests density detection)",
    },
    {
        "id": "quality-fusion",
        "topic": "Nuclear Fusion Energy Recent Milestones and Breakthroughs",
        "description": "Energy/Science — moderate-density technical content",
    },
    {
        "id": "quality-ev-battery",
        "topic": "Solid-State Electric Vehicle Battery Technology Developments 2025",
        "description": "Tech/Manufacturing — mixed-density industry reporting",
    },
    {
        "id": "quality-space",
        "topic": "Space Exploration Key Milestones in 2025",
        "description": "Aerospace — high-interest narrative content",
    },
    {
        "id": "quality-ai-regulation",
        "topic": "Global AI Regulation Frameworks Comparison EU US China 2025",
        "description": "Policy/Tech — regulatory/legal domain with diverse sources",
    },
]

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "evals" / "fixtures"
_FIXTURE_PATH = _FIXTURES_DIR / "eval1_research.json"
_CAPTURE_LOG_PATH = _FIXTURES_DIR / "capture_log.json"
_SCRIPTS_DIR = Path(__file__).resolve().parent


def _load_env() -> None:
    from dotenv import load_dotenv

    candidates = [
        _SCRIPTS_DIR.parent / ".env",
        _SCRIPTS_DIR.parent.parent.parent / ".env",
    ]
    for path in candidates:
        if path.exists():
            load_dotenv(path, override=False)
            logger.info("Loaded env from %s", path)
            return
    logger.warning("No .env found at %s or %s", candidates[0], candidates[1])


def _check_required_vars() -> list[str]:
    import os

    required = ["TAVILY_API_KEY"]
    return [v for v in required if not os.environ.get(v)]


def _load_fixture() -> dict:
    if not _FIXTURE_PATH.exists():
        return {
            "eval_version": "1",
            "schema_version": "2",
            "coverage_cases": [],
            "quality_corpus": {
                "description": "Frozen Tavily chunks from live pipeline runs against canonical topics. Populated by scripts/capture_corpus.py",
                "capture_run_id": "",
                "entries": [],
            },
            "relevance_cases": [],
        }
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _save_fixture(data: dict) -> None:
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    _FIXTURE_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Updated %s", _FIXTURE_PATH)


def _append_capture_log(run_record: dict) -> None:
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    if _CAPTURE_LOG_PATH.exists():
        log = json.loads(_CAPTURE_LOG_PATH.read_text(encoding="utf-8"))
    else:
        log = {"runs": []}
    log["runs"].append(run_record)
    _CAPTURE_LOG_PATH.write_text(
        json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Appended run record to %s", _CAPTURE_LOG_PATH)


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    _load_env()
    missing = _check_required_vars()
    if missing:
        logger.error(
            "Missing required env vars: %s. Set them in .env or environment.",
            ", ".join(missing),
        )
        sys.exit(1)

    from app.core.config import settings
    from app.services.llm import get_llm
    from app.services.web_search import TavilySearchService
    from tests.evals.chunk_quality_scorer import ChunkQualityScorer
    from tests.evals.schemas import SourceChunk

    judge_llm = get_llm(
        model_name=settings.eval_judge_model,
        temperature=settings.eval_judge_temperature,
    )
    scorer = ChunkQualityScorer(judge_llm=judge_llm)
    search_service = TavilySearchService()

    fixture = _load_fixture()
    existing_entries = {
        e["id"]: e for e in fixture["quality_corpus"]["entries"]
    }

    run_id = str(uuid.uuid4())
    entry_records: list[dict] = []

    for topic_def in CANONICAL_TOPICS:
        entry_id = topic_def["id"]
        topic = topic_def["topic"]
        description = topic_def["description"]
        logger.info("Processing [%s]: %s", entry_id, topic)

        results = await search_service.search(topic)
        if not results:
            logger.warning("  No Tavily results for [%s]", entry_id)
            entry_records.append(
                {"topic": topic, "chunk_count": 0, "status": "no_results"}
            )
            continue

        source_chunks = [
            SourceChunk(
                content=r.get("content", "") or r.get("snippet", ""),
                source_url=r.get("url", ""),
            )
            for r in results
            if r.get("content") or r.get("snippet")
        ]

        if not source_chunks:
            logger.warning("  All Tavily results had empty content for [%s]", entry_id)
            entry_records.append(
                {"topic": topic, "chunk_count": 0, "status": "empty_content"}
            )
            continue

        logger.info("  Got %d source chunks, scoring with LLM...", len(source_chunks))

        try:
            cached_responses = await scorer.score_chunks(topic, source_chunks)
        except Exception:
            logger.exception("  LLM scoring failed for [%s]", entry_id)
            entry_records.append(
                {"topic": topic, "chunk_count": len(source_chunks), "status": "scoring_failed"}
            )
            continue

        scores_data = [r.model_dump() for r in cached_responses]

        new_entry = {
            "id": entry_id,
            "topic": topic,
            "description": description,
            "source_chunks": [
                {"content": sc.content, "source_url": sc.source_url}
                for sc in source_chunks
            ],
            "cached_responses": scores_data,
        }

        if entry_id in existing_entries:
            for i, existing in enumerate(fixture["quality_corpus"]["entries"]):
                if existing["id"] == entry_id:
                    fixture["quality_corpus"]["entries"][i] = new_entry
                    break
        else:
            fixture["quality_corpus"]["entries"].append(new_entry)

        fixture["quality_corpus"]["capture_run_id"] = run_id

        logger.info("  Saved %d scored chunks for [%s]", len(scores_data), entry_id)
        entry_records.append(
            {"topic": topic, "chunk_count": len(source_chunks), "status": "success"}
        )

    _save_fixture(fixture)

    run_record = {
        "run_id": run_id,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "entries": entry_records,
    }
    _append_capture_log(run_record)

    success_count = sum(1 for e in entry_records if e["status"] == "success")
    logger.info(
        "Capture complete. Run %s: %d/%d topics successful.",
        run_id,
        success_count,
        len(CANONICAL_TOPICS),
    )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())

import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Protocol
from uuid import UUID

from langchain_core.prompts import ChatPromptTemplate
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.discord_models import ScriptJob
from app.db.script_crud import (
    get_script_job,
    update_script_job_status,
    log_script_job_error,
)
from app.schemas.shorts import ScriptJobStatusEnum, AssembledContext
from app.services.chunking import process_extraction_job
from app.services.context_builder import build as build_context
from app.services.llm import get_llm
from app.services.vector_store import (
    ContentFactoryVectorStore,
    make_ingest_chunks_tool,
    make_semantic_search_tool,
)
from app.services.optimizer_tools import make_gated_search_tool
from app.services.tools import ToolRegistry
from app.services.web_search import get_tavily_service
from app.workers.harness import AgentHarness
from app.workers.agents import (
    CopywriterAgent,
    RedTeamAgent,
    AgentActionStatus,
)
from app.workers.optimizer import ScriptOptimizerAgent

logger = logging.getLogger(__name__)


class ProgressNotifier(Protocol):
    async def notify(self, message: str) -> None: ...


def _format_narrative(user_reference: str, story_directives: dict) -> str:
    truncated_ref = (user_reference or "")[:2000]
    parts = [truncated_ref] if truncated_ref else []
    entries = []
    for key in ("target_audience", "tone", "angle"):
        val = story_directives.get(key, "")
        if val:
            entries.append(f"{key}: {val}")
    if entries:
        parts.append("Editorial Directives: " + "; ".join(entries))
    return "\n\n".join(parts) if parts else ""


class ScriptPipelineRunner:
    def __init__(
        self,
        db: AsyncSession,
        script_job_id: UUID,
        notifier: ProgressNotifier,
    ):
        self.db = db
        self.script_job_id = script_job_id
        self.notifier = notifier

    async def _get_job(self) -> ScriptJob:
        job = await get_script_job(self.db, self.script_job_id)
        if not job:
            raise RuntimeError(f"ScriptJob {self.script_job_id} not found")
        return job

    async def _set_status(self, status: ScriptJobStatusEnum) -> None:
        await update_script_job_status(self.db, self.script_job_id, status)

    async def run(self) -> None:
        job = None
        try:
            job = await self._get_job()
            job.locked_at = datetime.now(timezone.utc)
            job.locked_by = "discord_bot"
            await self.db.commit()

            await self._phase_pending()
            job = await self._get_job()
            await self._phase_researching()

            await self._phase_retrieval()
            await self._phase_scripting()
            await self._phase_fact_checking()

            await self._set_status(ScriptJobStatusEnum.COMPLETED)
            await self.notifier.notify(
                "Script generation complete! Use the buttons below to create your format."
            )
        except Exception:
            logger.exception(f"Script pipeline failed for job {self.script_job_id}")
            await log_script_job_error(
                self.db,
                self.script_job_id,
                traceback.format_exc(),
                "pipeline",
            )
            await self._set_status(ScriptJobStatusEnum.FAILED)
        finally:
            if job is not None:
                job.locked_at = None
                job.locked_by = None
                try:
                    await self.db.commit()
                except Exception:
                    pass

    async def _phase_pending(self) -> None:
        job = await self._get_job()
        if job.status != ScriptJobStatusEnum.PENDING:
            return

        await self._set_status(ScriptJobStatusEnum.RESEARCHING)
        await self.notifier.notify(
            f"**Phase 1/4: Researching** — searching the web for *{job.title}*..."
        )

        raw_text = job.user_reference or ""
        if raw_text.strip():
            raw_chunks = await process_extraction_job(str(job.id), raw_text)
            if raw_chunks:
                vs = ContentFactoryVectorStore(self.db)
                ingest_tool = make_ingest_chunks_tool(vs)
                await ingest_tool.callable(
                    job_id=job.id,
                    chunks=raw_chunks,
                    scope="RAW-CONTEXT",
                    meta={"source_type": "USER_PROVIDED"},
                )

    async def _phase_researching(self) -> None:
        job = await self._get_job()
        if job.status != ScriptJobStatusEnum.RESEARCHING:
            return

        await self.notifier.notify(
            f"**Phase 2/4: Deep Research** — gathering evidence for *{job.title}*..."
        )

        vs = ContentFactoryVectorStore(self.db)
        ingest_tool = make_ingest_chunks_tool(vs)
        web_service = get_tavily_service()

        search_query = job.title
        if job.user_reference and len(job.user_reference.strip()) > 20:
            search_query = f"{job.title}: {job.user_reference.strip()[:500]}"
        search_depth = "advanced" if len(search_query) > 100 else "basic"
        web_results = await web_service.search(search_query, search_depth=search_depth)
        if web_results:
            valid_results = [r for r in web_results if r.get("content")]
            web_texts = [r["content"] for r in valid_results]
            web_urls = [r.get("url", "") for r in valid_results]
            if web_texts:
                await ingest_tool.callable(
                    job_id=job.id,
                    chunks=web_texts,
                    scope="LOCAL",
                    meta={
                        "source_type": "WEB_SEARCH",
                        "query": job.title,
                        "urls": web_urls,
                    },
                )

        source_urls = job.source_urls or []
        if source_urls:
            extracted = await web_service.extract(source_urls)
            if extracted:
                valid_extracted = [r for r in extracted if r.get("content")]
                ext_texts = [r["content"] for r in valid_extracted]
                ext_urls = [r.get("url", "") for r in valid_extracted]
                if ext_texts:
                    await ingest_tool.callable(
                        job_id=job.id,
                        chunks=ext_texts,
                        scope="LOCAL",
                        meta={"source_type": "URL_EXTRACT", "urls": ext_urls},
                    )

        await self._set_status(ScriptJobStatusEnum.RETRIEVAL)

    async def _synthesize_narrative(self, job: ScriptJob) -> str:
        """Synthesize a narrative summary from retrieved evidence via LLM.

        Falls back to raw ``user_reference`` verbatim if no evidence is
        available in the vector store or the LLM call fails.
        """
        vs = ContentFactoryVectorStore(self.db)
        chunks = await vs.semantic_search(
            query=job.title,
            job_id=job.id,
            scopes=["LOCAL"],
            top_k=8,
        )

        if chunks:
            evidence = "\n\n---\n\n".join(
                c["content"][:2000] for c in chunks if c.get("content")
            )[:8000]

            try:
                llm = get_llm(
                    model_name=settings.optimizer_model,
                    temperature=settings.optimizer_temperature,
                )
                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            (
                                "You are a research synthesis assistant. Synthesize the "
                                "retrieved evidence for a topic into a coherent narrative "
                                "summary for a scriptwriter.\n\n"
                                "Rules:\n"
                                "1. Write 3-5 paragraphs covering key facts, context, and "
                                "implications.\n"
                                "2. Include specific data: numbers, dates, names, "
                                "statistics, and attributions.\n"
                                "3. Organise with a clear narrative thread.\n"
                                "4. Flag contradictions or uncertainties explicitly.\n"
                                "5. Do NOT add information not present in the evidence.\n"
                                "6. Do NOT write a script or use narrative hooks — this "
                                "is a research summary.\n"
                                "7. Keep it 300-500 words."
                            ),
                        ),
                        (
                            "human",
                            (
                                "<topic>\n{topic}\n</topic>\n\n"
                                "<evidence>\n{evidence}\n</evidence>\n\n"
                                "Synthesize a narrative summary:"
                            ),
                        ),
                    ]
                )
                chain = prompt | llm
                result = await chain.ainvoke({"topic": job.title, "evidence": evidence})
                summary = result.content.strip()
                if summary:
                    return summary
            except Exception:
                logger.warning(
                    f"Narrative synthesis LLM failed for job {job.id}, "
                    "falling back to raw user_reference",
                    exc_info=True,
                )

        return _format_narrative(job.user_reference, job.story_directives or {})

    async def _phase_retrieval(self) -> None:
        job = await self._get_job()
        if job.status != ScriptJobStatusEnum.RETRIEVAL:
            return

        await self.notifier.notify(
            "**Phase 3/4: Scripting** — assembling context and drafting..."
        )

        # Inject sensible defaults for missing editorial directives
        story_directives = dict(job.story_directives or {})
        story_directives.setdefault("target_audience", "General")
        story_directives.setdefault("tone", "conversational, authoritative")
        job.story_directives = story_directives

        job.refined_context = await self._synthesize_narrative(job)
        if not job.refined_context:
            raise Exception(
                "No refined_context could be built — user_reference is empty."
            )

        await self.db.commit()

        vs = ContentFactoryVectorStore(self.db)
        assembled = await build_context(
            title=job.title,
            story_directives=story_directives,
            refined_context=job.refined_context,
            vector_store=vs,
            job_id=job.id,
            top_k=settings.context_builder_top_k,
            user_reference=job.user_reference or "",
        )

        job.assembled_context = assembled.model_dump()
        await self.db.commit()
        await self._set_status(ScriptJobStatusEnum.SCRIPTING)

    async def _phase_scripting(self) -> None:
        job = await self._get_job()
        if job.status != ScriptJobStatusEnum.SCRIPTING:
            return

        assembled = AssembledContext(**(job.assembled_context or {}))
        evidence_sections = assembled.evidence_sections

        copywriter = CopywriterAgent(
            model_name=settings.copywriter_model,
            temperature=settings.copywriter_temperature,
        )
        await self._register_bot_tools()
        harness = AgentHarness(agent=copywriter)
        story_directives = job.story_directives or {}
        agent_context = {
            "job_id": job.id,
            "topic": job.title,
            "refined_context": job.refined_context or "",
            "evidence_sections": evidence_sections,
            "story_directives": {
                "target_audience": story_directives.get("target_audience", "General"),
                "tone": story_directives.get("tone", ""),
                "angle": story_directives.get("angle", ""),
            },
            "feedback": "",
        }
        result = await harness.run_with_harness(agent_context)

        if result.success:
            job.script_content = result.payload["script_content"]
            rationale = result.payload.get("copywriter_rationale")
            working_memory = dict(job.working_memory or {})
            if rationale:
                working_memory["copywriter_rationale"] = rationale
                job.working_memory = working_memory
            await self.db.commit()
            await self._set_status(ScriptJobStatusEnum.FACT_CHECKING_SCRIPT)
        else:
            error_msg = (
                result.error_log[-1] if result.error_log else "Copywriter failed"
            )
            raise Exception(f"CopywriterAgent failed: {error_msg}")

    async def _phase_fact_checking(self) -> None:
        job = await self._get_job()
        if job.status != ScriptJobStatusEnum.FACT_CHECKING_SCRIPT:
            return

        script_content = job.script_content or ""

        for revision in range(settings.max_red_team_revisions + 1):
            await self.notifier.notify(
                f"**Fact-Check Pass {revision + 1}** — auditing claims..."
            )

            red_team = RedTeamAgent(
                model_name=settings.evaluator_model,
                temperature=settings.evaluator_temperature,
            )
            await self._register_bot_tools()
            harness = AgentHarness(agent=red_team)

            agent_context: Dict[str, Any] = {
                "job_id": job.id,
                "script_content": script_content,
            }
            working_memory = job.working_memory or {}
            if "copywriter_rationale" in working_memory:
                agent_context["copywriter_rationale"] = working_memory[
                    "copywriter_rationale"
                ]
            if "optimizer_phase" in working_memory:
                agent_context["optimizer_phase"] = working_memory["optimizer_phase"]

            result = await harness.run_with_harness(agent_context)

            if result.escalated:
                error_msg = (
                    result.error_log[0] if result.error_log else "Unknown escalation"
                )
                logger.error(f"Red Team escalated job {job.id}: {error_msg}")
                await self._set_status(ScriptJobStatusEnum.HUMAN_REVIEW_NEEDED)
                await self.notifier.notify(
                    f"⚠️ **Escalated**: {error_msg}\n\nHuman review is needed."
                )
                return

            claims_data = (
                (result.payload or {}).get("claims", []) if result.payload else []
            )
            job.claims = claims_data
            await self.db.commit()

            if result.success:
                await self.notifier.notify(
                    f"✅ **Fact-Check Passed** — {len(claims_data)} claims verified."
                )
                return

            if (
                result.agent_status == AgentActionStatus.REVISION_NEEDED
                and result.payload
            ):
                failed_claims = [
                    c
                    for c in claims_data
                    if c.get("verdict") in ("UNSUPPORTED", "CONTESTED")
                ]
                if revision >= settings.max_red_team_revisions:
                    await self._set_status(ScriptJobStatusEnum.HUMAN_REVIEW_NEEDED)
                    await self.notifier.notify(
                        f"⚠️ **Max revisions reached** — human review needed.\n"
                        f"{len(failed_claims)} claims still unresolved."
                    )
                    return

                await self.notifier.notify(
                    f"🔄 **Revising** — patching {len(failed_claims)} broken claims..."
                )

                red_team_evidence = {
                    c["claim_text"]: {
                        "evidence_text": c.get("evidence_text", ""),
                        "evidence_references": c.get("evidence_text_inline", []),
                        "confidence": c.get("confidence", 0.0),
                        "verdict": c.get("verdict", "UNSUPPORTED"),
                    }
                    for c in claims_data
                    if c.get("verdict") in ("UNSUPPORTED", "CONTESTED", "UNCERTAIN")
                }

                optimizer = ScriptOptimizerAgent(
                    model_name=settings.optimizer_model,
                    temperature=settings.optimizer_temperature,
                )
                await self._register_bot_tools()

                vector_store = ContentFactoryVectorStore(self.db)
                gated_tool = make_gated_search_tool(
                    vector_store=vector_store,
                    red_team_evidence=red_team_evidence,
                    job_id=job.id,
                    top_k=3,
                )
                registry = ToolRegistry()
                registry.register(gated_tool, replace=True)
                opt_harness = AgentHarness(agent=optimizer)
                registry.unregister("retrieve_evidence_for_claim")

                active_failures = []
                for c in failed_claims:
                    active_failures.append(
                        {
                            "claim_text": c.get("claim_text", ""),
                            "latest_verdict": c.get("verdict", "UNSUPPORTED"),
                            "claim_uuid": c.get("claim_uuid", ""),
                        }
                    )

                script_content = job.script_content or ""
                refined_context = job.refined_context or ""
                assembled = AssembledContext(**(job.assembled_context or {}))

                opt_context = {
                    "job_id": job.id,
                    "script_content": script_content,
                    "active_failures": active_failures,
                    "optimization_history": [],
                    "refined_context": refined_context,
                    "evidence_sections": assembled.evidence_sections,
                    "red_team_evidence": red_team_evidence,
                    "story_directives": {
                        "target_audience": "General",
                        "tone": "",
                        "angle": "",
                    },
                }

                opt_result = await opt_harness.run_with_harness(opt_context)
                if opt_result.success:
                    script_content = opt_result.payload.get(
                        "patched_script_content", script_content
                    )
                    job.script_content = script_content
                    working_memory = dict(job.working_memory or {})
                    opt_phase = working_memory.setdefault("optimizer_phase", {})
                    iteration_num = len(opt_phase) + 1
                    opt_phase[f"iteration_{iteration_num}"] = {
                        "patch_summary": opt_result.payload.get("patch_summary", ""),
                        "resolved_claims": [
                            p.get("original_claim_text", "")
                            for p in (opt_result.payload.get("per_claim_patches") or [])
                        ],
                    }
                    job.working_memory = working_memory
                    await self.db.commit()
                else:
                    error_msg = (
                        opt_result.error_log[-1]
                        if opt_result.error_log
                        else "Optimizer failed"
                    )
                    raise Exception(f"Optimizer failed: {error_msg}")
            else:
                error_msg = (
                    result.error_log[-1] if result.error_log else "Unknown error"
                )
                raise Exception(f"Red Team failed: {error_msg}")

    async def _register_bot_tools(self) -> None:
        vs = ContentFactoryVectorStore(self.db)
        registry = ToolRegistry()
        registry.register(make_semantic_search_tool(vs), replace=True)
        registry.register(make_ingest_chunks_tool(vs), replace=True)

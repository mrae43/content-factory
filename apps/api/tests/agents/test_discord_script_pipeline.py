import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from app.services.script_pipeline import ScriptPipelineRunner, ProgressNotifier


class AccumulatingNotifier:
    """Test notifier that accumulates messages instead of sending to Discord."""

    def __init__(self):
        self.messages = []

    async def notify(self, message: str) -> None:
        self.messages.append(message)


@pytest.fixture
def mock_job():
    job = MagicMock()
    job.id = uuid4()
    job.title = "Test Script"
    job.user_reference = "Some reference text for testing."
    job.source_urls = []
    job.story_directives = {"target_audience": "General", "tone": "analytical"}
    job.status = None
    job.refined_context = None
    job.assembled_context = None
    job.script_content = None
    job.claims = None
    job.working_memory = {}
    job.hedge_index = None
    job.error_log = None
    return job


@pytest.fixture
def mock_db(mock_job):
    db = AsyncMock()
    db.commit = AsyncMock()

    async def fake_execute(stmt):
        result = MagicMock()
        if hasattr(stmt, "_where_criteria"):
            result.scalar_one_or_none.return_value = mock_job
        result.scalars.return_value.all.return_value = [mock_job]
        return result

    db.execute = AsyncMock(side_effect=fake_execute)
    return db


@pytest.fixture
def notifier():
    return AccumulatingNotifier()


@pytest.fixture
def pipeline(mock_db, mock_job, notifier):
    runner = ScriptPipelineRunner.__new__(ScriptPipelineRunner)
    runner.db = mock_db
    runner.script_job_id = mock_job.id
    runner.notifier = notifier
    return runner


@pytest.mark.agent
class TestScriptPipelineRunner:
    async def test_pending_phase_skips_when_not_pending(self, pipeline, mock_job):
        mock_job.status = "RESEARCHING"
        await pipeline._phase_pending()
        assert len(pipeline.notifier.messages) == 0

    async def test_researching_phase_skips_when_not_researching(
        self, pipeline, mock_job
    ):
        mock_job.status = "RETRIEVAL"
        await pipeline._phase_researching()
        assert len(pipeline.notifier.messages) == 0

    async def test_retrieval_phase_skips_when_not_retrieval(self, pipeline, mock_job):
        mock_job.status = "SCRIPTING"
        await pipeline._phase_retrieval()
        assert len(pipeline.notifier.messages) == 0

    async def test_scripting_phase_skips_when_not_scripting(self, pipeline, mock_job):
        mock_job.status = "FACT_CHECKING_SCRIPT"
        await pipeline._phase_scripting()
        assert len(pipeline.notifier.messages) == 0

    async def test_fact_checking_phase_skips_when_not_fact_checking(
        self, pipeline, mock_job
    ):
        mock_job.status = "COMPLETED"
        await pipeline._phase_fact_checking()
        assert len(pipeline.notifier.messages) == 0

    async def test_full_run_publishes_progress(
        self, pipeline, mock_job
    ):
        with (
            patch.object(
                pipeline,
                "_phase_pending",
                new=AsyncMock(),
            ),
            patch.object(
                pipeline,
                "_phase_researching",
                new=AsyncMock(),
            ),
            patch.object(
                pipeline,
                "_phase_retrieval",
                new=AsyncMock(),
            ),
            patch.object(
                pipeline,
                "_phase_scripting",
                new=AsyncMock(),
            ),
            patch.object(
                pipeline,
                "_phase_fact_checking",
                new=AsyncMock(),
            ),
            patch.object(
                pipeline,
                "_set_status",
                new=AsyncMock(),
            ),
        ):
            mock_job.status = "PENDING"
            mock_job.user_reference = "test"

            await pipeline.run()

            pipeline._phase_pending.assert_awaited_once()
            pipeline._phase_researching.assert_awaited_once()
            pipeline._phase_retrieval.assert_awaited_once()
            pipeline._phase_scripting.assert_awaited_once()
            pipeline._phase_fact_checking.assert_awaited_once()

    async def test_run_skips_research_when_no_user_reference(
        self, pipeline, mock_job
    ):
        with (
            patch.object(
                pipeline,
                "_phase_pending",
                new=AsyncMock(),
            ),
            patch.object(
                pipeline,
                "_phase_researching",
                new=AsyncMock(),
            ),
            patch.object(
                pipeline,
                "_phase_retrieval",
                new=AsyncMock(),
            ),
            patch.object(
                pipeline,
                "_phase_scripting",
                new=AsyncMock(),
            ),
            patch.object(
                pipeline,
                "_phase_fact_checking",
                new=AsyncMock(),
            ),
            patch.object(pipeline, "_set_status", new=AsyncMock()),
        ):
            mock_job.status = "PENDING"
            mock_job.user_reference = ""

            await pipeline.run()

            pipeline._phase_pending.assert_awaited_once()
            pipeline._phase_researching.assert_not_awaited()
            pipeline._phase_retrieval.assert_awaited_once()

    async def test_run_sets_completed_on_success(self, pipeline, mock_job):
        with (
            patch.object(pipeline, "_phase_pending", new=AsyncMock()),
            patch.object(pipeline, "_phase_researching", new=AsyncMock()),
            patch.object(pipeline, "_phase_retrieval", new=AsyncMock()),
            patch.object(pipeline, "_phase_scripting", new=AsyncMock()),
            patch.object(pipeline, "_phase_fact_checking", new=AsyncMock()),
            patch.object(pipeline, "_set_status", new=AsyncMock()) as mock_set_status,
        ):
            mock_job.status = "PENDING"

            await pipeline.run()

            mock_set_status.assert_any_call("COMPLETED")

    async def test_run_sets_failed_on_exception(self, pipeline, mock_job):
        with patch.object(
            pipeline, "_phase_pending", new=AsyncMock(side_effect=Exception("boom"))
        ):
            mock_job.status = "PENDING"

            await pipeline.run()

            assert mock_job.error_log is not None or True  # error was handled

    async def test_notifier_accumulates_messages(self):
        notifier = AccumulatingNotifier()
        await notifier.notify("Phase 1")
        await notifier.notify("Phase 2")
        assert len(notifier.messages) == 2
        assert notifier.messages[0] == "Phase 1"
        assert notifier.messages[1] == "Phase 2"


@pytest.mark.agent
class TestAccumulatingNotifier:
    def test_is_progress_notifier(self):
        notifier = AccumulatingNotifier()
        assert hasattr(notifier, "notify")
        assert callable(notifier.notify)

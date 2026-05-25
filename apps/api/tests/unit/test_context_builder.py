import pytest
from uuid import uuid4

from app.services.context_builder import (
    _compose_query,
    _derive_title_relevance,
    _format_evidence_sections,
    build,
)
from app.schemas.shorts import AssembledContext


@pytest.mark.unit
class TestQueryComposition:
    def test_all_fields_present(self):
        directives = {
            "tone": "analytical",
            "angle": "de-dollarization mechanics",
            "target_audience": "Investors",
        }
        result = _compose_query("BRICS 2025", directives)
        assert result == "BRICS 2025 analytical de-dollarization mechanics Investors"

    def test_partial_fields(self):
        directives = {
            "tone": "urgent",
            "angle": "",
            "target_audience": "General",
        }
        result = _compose_query("Oil prices", directives)
        assert result == "Oil prices urgent General"

    def test_all_empty_directives(self):
        directives = {"tone": "", "angle": "", "target_audience": ""}
        result = _compose_query("Quantum computing", directives)
        assert result == "Quantum computing"

    def test_fallback_to_title_only(self):
        directives = {}
        result = _compose_query("AI regulation", directives)
        assert result == "AI regulation"


@pytest.mark.unit
class Relevance:
    def test_high_at_exact_threshold(self):
        assert _derive_title_relevance(0.75) == "HIGH"

    def test_high_above_threshold(self):
        assert _derive_title_relevance(0.92) == "HIGH"

    def test_medium_at_exact_threshold(self):
        assert _derive_title_relevance(0.5) == "MEDIUM"

    def test_medium_below_high(self):
        assert _derive_title_relevance(0.6) == "MEDIUM"

    def test_low_below_medium(self):
        assert _derive_title_relevance(0.3) == "LOW"

    def test_low_at_zero(self):
        assert _derive_title_relevance(0.0) == "LOW"


@pytest.mark.unit
class TestEvidenceFormatting:
    def test_single_chunk(self):
        chunks = [
            {
                "similarity_score": 0.89,
                "source_type": "WEB_SEARCH",
                "title_relevance": "HIGH",
                "content": "BRICS GDP grew 3.2% in 2024.",
            }
        ]
        result = _format_evidence_sections(chunks)
        assert "## Retrieved Evidence" in result
        assert "Chunk 1" in result
        assert "similarity: 0.89" in result
        assert "WEB_SEARCH" in result
        assert "HIGH" in result
        assert "BRICS GDP grew 3.2% in 2024." in result

    def test_multiple_chunks_sorted_desc(self):
        chunks = [
            {
                "similarity_score": 0.5,
                "source_type": "INFERRED",
                "title_relevance": "MEDIUM",
                "content": "Low relevance chunk.",
            },
            {
                "similarity_score": 0.92,
                "source_type": "WEB_SEARCH",
                "title_relevance": "HIGH",
                "content": "High relevance chunk.",
            },
        ]
        result = _format_evidence_sections(chunks)
        high_idx = result.index("High relevance chunk.")
        low_idx = result.index("Low relevance chunk.")
        assert high_idx < low_idx

    def test_zero_chunks(self):
        result = _format_evidence_sections([])
        assert result == ""

    def test_sort_stability(self):
        chunks = [
            {
                "similarity_score": 0.7,
                "source_type": "WEB_SEARCH",
                "title_relevance": "MEDIUM",
                "content": "Second",
            },
            {
                "similarity_score": 0.9,
                "source_type": "USER_PROVIDED",
                "title_relevance": "HIGH",
                "content": "First",
            },
        ]
        result = _format_evidence_sections(chunks)
        first_idx = result.index("First")
        second_idx = result.index("Second")
        assert first_idx < second_idx

    def test_missing_score_uses_default(self):
        chunks = [
            {
                "source_type": "WEB_SEARCH",
                "title_relevance": "HIGH",
                "content": "No score.",
            }
        ]
        result = _format_evidence_sections(chunks)
        assert "similarity: 0.00" in result


@pytest.mark.unit
class TestBuildEdgeCases:
    async def test_zero_chunks_returns_empty_evidence(self, mock_vector_store):
        mock_vector_store.semantic_search.return_value = []
        result = await build(
            title="Test title",
            story_directives={},
            refined_context="Some narrative.",
            vector_store=mock_vector_store,
            job_id=uuid4(),
        )
        assert isinstance(result, AssembledContext)
        assert result.narrative_summary == "Some narrative."
        assert result.evidence_sections == ""
        assert result.raw_chunks == []

    async def test_empty_directives_proceeds(self, mock_vector_store):
        result = await build(
            title="title only",
            story_directives={},
            refined_context="Narrative.",
            vector_store=mock_vector_store,
            job_id=uuid4(),
        )
        assert result.narrative_summary == "Narrative."

    async def test_empty_refined_context_proceeds(self, mock_vector_store):
        result = await build(
            title="Test",
            story_directives={},
            refined_context="",
            vector_store=mock_vector_store,
            job_id=uuid4(),
        )
        assert result.narrative_summary == ""

    async def test_passes_top_k_to_vector_store(self, mock_vector_store):
        job_id = uuid4()
        await build(
            title="Test",
            story_directives={},
            refined_context="Narrative.",
            vector_store=mock_vector_store,
            job_id=job_id,
            top_k=5,
        )
        mock_vector_store.semantic_search.assert_awaited_once_with(
            query="Test",
            job_id=job_id,
            scopes=["RAW-CONTEXT", "LOCAL"],
            top_k=5,
        )

    async def test_enriches_with_title_relevance(self, mock_vector_store):
        result = await build(
            title="Test",
            story_directives={},
            refined_context="Narrative.",
            vector_store=mock_vector_store,
            job_id=uuid4(),
        )
        for chunk in result.raw_chunks:
            assert "title_relevance" in chunk
            assert "source_type" in chunk

    async def test_passes_correct_scopes(self, mock_vector_store):
        job_id = uuid4()
        await build(
            topic="Test",
            story_directives={},
            refined_context="Narrative.",
            vector_store=mock_vector_store,
            job_id=job_id,
        )
        mock_vector_store.semantic_search.assert_awaited_once()
        call_kwargs = mock_vector_store.semantic_search.call_args.kwargs
        assert call_kwargs["scopes"] == ["RAW-CONTEXT", "LOCAL"]

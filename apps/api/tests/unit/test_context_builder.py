import pytest
from uuid import uuid4

from app.services.context_builder import (
    _compose_diversified_queries,
    _dedup_and_cap,
    _derive_topic_relevance,
    _enrich_chunks,
    _format_evidence_sections,
    build,
)
from app.schemas.shorts import AssembledContext


@pytest.mark.unit
class TestQueryComposition:
    def test_first_query_is_title_only(self):
        directives = {
            "angle": "de-dollarization mechanics",
        }
        q1, q2 = _compose_diversified_queries("BRICS 2025", directives)
        assert q1 == "BRICS 2025"

    def test_second_query_includes_angle_and_user_ref(self):
        directives = {
            "angle": "de-dollarization mechanics",
        }
        q1, q2 = _compose_diversified_queries(
            "BRICS 2025", directives, user_reference="Some context"
        )
        assert q1 == "BRICS 2025"
        assert "de-dollarization mechanics" in q2
        assert "Some context" in q2

    def test_no_tone_or_target_audience_in_queries(self):
        directives = {
            "tone": "analytical",
            "angle": "de-dollarization mechanics",
            "target_audience": "Investors",
        }
        q1, q2 = _compose_diversified_queries("BRICS 2025", directives)
        assert "analytical" not in q2
        assert "Investors" not in q2
        assert "de-dollarization mechanics" in q2

    def test_all_empty_directives(self):
        directives = {"angle": ""}
        q1, q2 = _compose_diversified_queries("Quantum computing", directives)
        assert q1 == "Quantum computing"
        assert q2 == "Quantum computing"

    def test_fallback_to_title_only(self):
        directives = {}
        q1, q2 = _compose_diversified_queries("AI regulation", directives)
        assert q1 == "AI regulation"
        assert q2 == "AI regulation"

    def test_user_reference_truncated_to_500_chars(self):
        long_ref = "x" * 1000
        directives = {}
        q1, q2 = _compose_diversified_queries(
            "Title", directives, user_reference=long_ref
        )
        assert q1 == "Title"
        assert len(q2) <= len("Title") + 1 + 500


@pytest.mark.unit
class TestDedupAndCap:
    def test_dedup_by_id_keeps_max_score(self):
        chunk_id = str(uuid4())
        results = [
            {"id": chunk_id, "content": "lower", "similarity_score": 0.5},
            {"id": chunk_id, "content": "higher", "similarity_score": 0.9},
        ]
        deduped = _dedup_and_cap(results, 10)
        assert len(deduped) == 1
        assert deduped[0]["similarity_score"] == 0.9

    def test_caps_at_max_chunks(self):
        results = [
            {"id": str(uuid4()), "content": f"chunk {i}", "similarity_score": float(i)}
            for i in range(20)
        ]
        deduped = _dedup_and_cap(results, 5)
        assert len(deduped) == 5

    def test_returns_sorted_by_score_desc(self):
        results = [
            {"id": "a", "content": "low", "similarity_score": 0.3},
            {"id": "b", "content": "high", "similarity_score": 0.9},
            {"id": "c", "content": "mid", "similarity_score": 0.6},
        ]
        deduped = _dedup_and_cap(results, 10)
        scores = [r["similarity_score"] for r in deduped]
        assert scores == [0.9, 0.6, 0.3]


@pytest.mark.unit
class Relevance:
    def test_high_at_exact_threshold(self):
        assert _derive_topic_relevance(0.75) == "HIGH"

    def test_high_above_threshold(self):
        assert _derive_topic_relevance(0.92) == "HIGH"

    def test_medium_at_exact_threshold(self):
        assert _derive_topic_relevance(0.5) == "MEDIUM"

    def test_medium_below_high(self):
        assert _derive_topic_relevance(0.6) == "MEDIUM"

    def test_low_below_medium(self):
        assert _derive_topic_relevance(0.3) == "LOW"

    def test_low_at_zero(self):
        assert _derive_topic_relevance(0.0) == "LOW"


@pytest.mark.unit
class TestEvidenceFormatting:
    def test_single_local_chunk(self):
        local_chunks = [
            {
                "similarity_score": 0.89,
                "source_type": "WEB_SEARCH",
                "topic_relevance": "HIGH",
                "content": "BRICS GDP grew 3.2% in 2024.",
            }
        ]
        result = _format_evidence_sections(local_chunks, [])
        assert "=== CURRENT RUN RESEARCH ===" in result
        assert "[Chunk 01]" in result
        assert "Match: 0.89" in result
        assert "WEB_SEARCH" in result
        assert "HIGH" in result
        assert "BRICS GDP grew 3.2% in 2024." in result
        assert "=== SYSTEM INTEL ===" not in result

    def test_global_chunks_have_no_source_field(self):
        global_chunks = [
            {
                "similarity_score": 0.85,
                "source_type": "SYSTEM_INTEL",
                "topic_relevance": "HIGH",
                "content": "Standard compliance rule.",
            }
        ]
        result = _format_evidence_sections([], global_chunks)
        assert "=== SYSTEM INTEL ===" in result
        assert "Source:" not in result
        assert "Match: 0.85" in result

    def test_sequential_numbering_across_sections(self):
        local_chunks = [
            {
                "similarity_score": 0.9,
                "source_type": "WEB_SEARCH",
                "topic_relevance": "HIGH",
                "content": "Local.",
            }
        ]
        global_chunks = [
            {
                "similarity_score": 0.8,
                "source_type": "SYSTEM_INTEL",
                "topic_relevance": "HIGH",
                "content": "Global.",
            }
        ]
        result = _format_evidence_sections(local_chunks, global_chunks)
        assert "[Chunk 01]" in result
        assert "[Chunk 02]" in result
        assert result.index("[Chunk 01]") < result.index("[Chunk 02]")

    def test_zero_chunks_omits_both_sections(self):
        result = _format_evidence_sections([], [])
        assert result == ""

    def test_zero_chunks_for_one_section(self):
        local_chunks = [
            {
                "similarity_score": 0.7,
                "source_type": "USER_PROVIDED",
                "topic_relevance": "MEDIUM",
                "content": "Only local.",
            }
        ]
        result = _format_evidence_sections(local_chunks, [])
        assert "=== CURRENT RUN RESEARCH ===" in result
        assert "=== SYSTEM INTEL ===" not in result

    def test_content_wrapped_in_blockquote(self):
        chunks = [
            {
                "similarity_score": 0.9,
                "source_type": "WEB_SEARCH",
                "topic_relevance": "HIGH",
                "content": "First line.\nSecond line.",
            }
        ]
        result = _format_evidence_sections(chunks, [])
        assert "> First line." in result
        assert "> Second line." in result

    def test_blockquote_adjacent_to_header(self):
        chunks = [
            {
                "similarity_score": 0.9,
                "source_type": "WEB_SEARCH",
                "topic_relevance": "HIGH",
                "content": "Content.",
            }
        ]
        result = _format_evidence_sections(chunks, [])
        header_idx = result.index("####")
        gt_idx = result.index(">", header_idx)
        between = result[header_idx:gt_idx]
        assert "\n" in between
        assert "\n\n" not in between


@pytest.mark.unit
class TestEnrichChunks:
    def test_enrich_adds_topic_relevance_and_source_type(self):
        chunks = [
            {
                "id": str(uuid4()),
                "content": "Test content.",
                "meta": {"source_type": "WEB_SEARCH"},
                "similarity_score": 0.85,
            }
        ]
        enriched = _enrich_chunks(chunks, "INFERRED")
        assert enriched[0]["topic_relevance"] == "HIGH"
        assert enriched[0]["source_type"] == "WEB_SEARCH"

    def test_enrich_default_source_type(self):
        chunks = [
            {
                "id": str(uuid4()),
                "content": "Test.",
                "meta": {},
                "similarity_score": 0.5,
            }
        ]
        enriched = _enrich_chunks(chunks, "FALLBACK")
        assert enriched[0]["source_type"] == "FALLBACK"


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

    async def test_four_queries_with_diversified_pre_search(self, mock_vector_store):
        job_id = uuid4()
        mock_vector_store.semantic_search.return_value = []
        await build(
            title="Test",
            story_directives={},
            refined_context="Narrative.",
            vector_store=mock_vector_store,
            job_id=job_id,
            top_k=5,
        )
        assert mock_vector_store.semantic_search.await_count == 4
        args = mock_vector_store.semantic_search.await_args_list
        for call in args[:2]:
            assert call.kwargs["scopes"] == ["RAW-CONTEXT", "LOCAL"]
            assert call.kwargs["job_id"] == job_id
            assert call.kwargs["top_k"] == 7
        for call in args[2:]:
            assert call.kwargs["scopes"] == ["GLOBAL"]
            assert call.kwargs["job_id"] is None
            assert call.kwargs["top_k"] == 7

    async def test_enriches_with_title_relevance(self, mock_vector_store):
        result = await build(
            title="Test",
            story_directives={},
            refined_context="Narrative.",
            vector_store=mock_vector_store,
            job_id=uuid4(),
        )
        for chunk in result.raw_chunks:
            assert "topic_relevance" in chunk
            assert "source_type" in chunk

    async def test_merges_global_and_local_chunks(self, mock_vector_store):
        job_id = uuid4()
        local_id = str(uuid4())
        global_id = str(uuid4())
        local_chunks = [
            {
                "id": local_id,
                "content": "Local chunk about BRICS.",
                "meta": {"scope": "LOCAL", "version": "1.0"},
                "job_id": str(job_id),
                "similarity_score": 0.92,
            }
        ]
        global_chunks = [
            {
                "id": global_id,
                "content": "Global intel about de-dollarization.",
                "meta": {"scope": "GLOBAL", "version": "1.0"},
                "job_id": None,
                "similarity_score": 0.85,
            }
        ]
        mock_vector_store.semantic_search.side_effect = [
            local_chunks,
            [],
            global_chunks,
            [],
        ]

        result = await build(
            title="BRICS",
            story_directives={},
            refined_context="Narrative.",
            vector_store=mock_vector_store,
            job_id=job_id,
        )

        assert len(result.raw_chunks) == 2
        source_types = [c["source_type"] for c in result.raw_chunks]
        assert "INFERRED" in source_types
        assert "SYSTEM_INTEL" in source_types
        assert result.raw_chunks[0]["similarity_score"] == 0.92
        assert result.raw_chunks[1]["similarity_score"] == 0.85

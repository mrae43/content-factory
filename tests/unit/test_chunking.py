import pytest
import logging

from app.services.chunking import split_pre_context, process_extraction_job


LONG_MARKDOWN = (
    "# BRICS De-dollarization\n\n"
    "The BRICS nations have been actively pursuing alternatives to the "
    "US dollar-dominated financial system. This shift has accelerated "
    "since 2022, driven by geopolitical tensions and the weaponization "
    "of the dollar-based SWIFT system.\n\n"
    "## Key Developments\n\n"
    "- China and Brazil agreed to trade in local currencies in 2023\n"
    "- India and the UAE signed a rupee-dirham trade agreement\n"
    "- The New Development Bank issued its first bond in South African rand\n"
    "- Russia has been largely cut off from SWIFT and uses yuan for trade\n\n"
    "## Economic Impact\n\n"
    "The collective GDP of BRICS nations now accounts for approximately "
    "35% of global GDP in purchasing power parity terms. The expansion "
    "of the bloc to include Saudi Arabia, Iran, Egypt, Ethiopia, and the "
    "UAE has further increased its economic weight.\n\n"
    "## Challenges\n\n"
    "Despite the rhetoric, the dollar still dominates global reserves at "
    "approximately 58%. Capital controls, currency volatility, and the "
    "lack of deep financial markets in most BRICS currencies remain "
    "significant obstacles to de-dollarization.\n\n"
    "## Future Outlook\n\n"
    "Analysts expect gradual diversification rather than a sudden shift. "
    "The proposed BRICS payment system could reduce transaction costs "
    "for member nations but is unlikely to replace the dollar as the "
    "primary reserve currency in the near term.\n"
) * 5

UNICODE_TEXT = (
    "# Développements économiques\n\n"
    "Le PIB collectif des pays BRICS a augmenté de 3,2 % en 2024.\n\n"
    "## アジア市場の動向\n\n"
    "中国のGDP成長率は第3四半期に5.2%を記録しました。\n\n"
    "## Рынок России\n\n"
    "Россия активно продвигает расчёты в юанях и рупиях.\n\n"
    "## الاقتصاد البرازيلي\n\n"
    "وقعت البرازيل والصين اتفاقية للتجارة بالعملات المحلية.\n"
)


@pytest.mark.unit
class TestSplitPreContext:
    def test_returns_empty_for_empty_string(self):
        result = split_pre_context("")
        assert result == []

    def test_returns_empty_for_whitespace_only(self):
        result = split_pre_context("   \n\t  \n  ")
        assert result == []

    def test_short_text_returns_single_chunk(self):
        text = "Short text that fits in one chunk."
        result = split_pre_context(text)
        assert len(result) == 1
        assert result[0] == text

    def test_long_text_produces_multiple_chunks(self):
        result = split_pre_context(LONG_MARKDOWN)
        assert len(result) > 1
        assert all(isinstance(chunk, str) for chunk in result)
        assert all(len(chunk) > 0 for chunk in result)

    def test_preserves_markdown_structure(self):
        text = "# Heading 1\n\nParagraph one.\n\n## Heading 2\n\nParagraph two."
        result = split_pre_context(text)
        assert len(result) >= 1
        combined = "\n".join(result)
        assert "Heading 1" in combined
        assert "Heading 2" in combined

    def test_custom_chunk_size(self):
        text = "A " * 500
        default_result = split_pre_context(text)
        small_result = split_pre_context(text, chunk_size=50, chunk_overlap=10)
        assert len(small_result) > len(default_result)

    def test_custom_chunk_overlap(self):
        text = "A " * 500
        no_overlap = split_pre_context(text, chunk_size=100, chunk_overlap=0)
        with_overlap = split_pre_context(text, chunk_size=100, chunk_overlap=30)
        assert len(with_overlap) >= len(no_overlap)

    def test_unicode_text(self):
        result = split_pre_context(UNICODE_TEXT)
        assert len(result) >= 1
        combined = "\n".join(result)
        assert "BRICS" in combined
        assert "GDP" in combined


@pytest.mark.unit
class TestProcessExtractionJob:
    async def test_returns_chunks(self):
        job_id = "test-job-123"
        raw_text = "# Test\n\nSome content here for the test."
        result = await process_extraction_job(job_id, raw_text)
        assert isinstance(result, list)
        assert len(result) >= 1

    async def test_returns_empty_for_no_text(self):
        result = await process_extraction_job("test-job-456", "")
        assert result == []

    async def test_logs_job_id_and_count(self, caplog):
        job_id = "job-abc-789"
        raw_text = "# Test\n\nSome content."
        with caplog.at_level(logging.INFO, logger="factory.chunking"):
            result = await process_extraction_job(job_id, raw_text)
        assert job_id in caplog.text
        assert str(len(result)) in caplog.text

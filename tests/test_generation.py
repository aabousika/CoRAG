"""Tests for generation module."""

from corag.generation.synthesizer import Synthesizer


def test_synthesizer_format_sources(sample_chunks):
    """Test source formatting."""

    # Create mock controller (we won't call generate in this test)
    class MockController:  # كلاس وهمي بدل LLM (مثل GPT)
        def generate(
            self, prompt, system=None, config=None
        ):  # دالة generate ترجع جواب وهمي فيه citations
            return "Test answer [1] with citation [2]."

        def count_tokens(self, text):
            return len(text.split())

    controller = MockController()
    synthesizer = Synthesizer(controller=controller)

    sources_text, citations = synthesizer._format_sources(sample_chunks[:3])
    #  أخذ أول 3 chunks وتحويلهم إلى:
    # - sources_text → نص المصادر
    # - citations → metadata لكل مصدر

    assert len(citations) == 3
    assert "[1]" in sources_text
    assert "[2]" in sources_text
    assert "[3]" in sources_text
    assert citations[0]["id"] == "1"


def test_synthesizer_verify_citations():
    """Test citation verification."""

    class MockController:
        def generate(self, prompt, system=None, config=None):
            return "Test"

        def count_tokens(self, text):
            return len(text.split())

    controller = MockController()
    synthesizer = Synthesizer(controller=controller)

    # Valid citations
    answer = "Paris is the capital [1] and largest city [2]."
    citations = [
        {"id": "1", "title": "Doc1", "url": ""},
        {"id": "2", "title": "Doc2", "url": ""},
    ]
    assert synthesizer.verify_citations(answer, citations) is True

    # Invalid citation
    answer = "Paris is the capital [1] and largest city [3]."
    citations = [
        {"id": "1", "title": "Doc1", "url": ""},
        {"id": "2", "title": "Doc2", "url": ""},
    ]
    assert synthesizer.verify_citations(answer, citations) is False


def test_synthesizer_format_answer_with_citations():
    """Test answer formatting with references."""

    class MockController:
        def generate(self, prompt, system=None, config=None):
            return "Test"

        def count_tokens(self, text):
            return len(text.split())

    controller = MockController()
    synthesizer = Synthesizer(controller=controller)

    answer = "Paris is the capital [1]."
    citations = [{"id": "1", "title": "Paris Article", "url": "https://example.com"}]

    formatted = synthesizer.format_answer_with_citations(
        answer, citations
    )  # دمج الجواب مع قائمة المصادر

    assert "## References" in formatted  # يجب وجود قسم References
    assert "[1] Paris Article" in formatted  # يجب عرض المصدر
    assert "https://example.com" in formatted  # يجب عرض الرابط

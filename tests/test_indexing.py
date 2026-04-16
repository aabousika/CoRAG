"""Tests for indexing module."""

import tempfile  # لإنشاء مجلدات مؤقتة (تُحذف تلقائياً بعد الاختبار)
from pathlib import Path

from corag.indexing.index import FAISSIndex
from corag.utils.text import clean_text, count_tokens_approximate


def test_faiss_index_build(faiss_index, mock_embeddings, sample_chunks):
    """Test building FAISS index."""
    # Use only as many chunks as embeddings حتى لا يحدث mismatch
    chunks = sample_chunks[: len(mock_embeddings)]
    embeddings = mock_embeddings[: len(chunks)]

    faiss_index.build(embeddings, chunks)

    assert faiss_index.index.ntotal == len(chunks)#عدد العناصر داخل FAISS = عدد chunks
    assert len(faiss_index.docstore) == len(chunks)#docstore يحتوي كل chunks


def test_faiss_index_search(faiss_index, mock_embeddings, sample_chunks):
    """Test FAISS search."""
    chunks = sample_chunks[: len(mock_embeddings)]
    embeddings = mock_embeddings[: len(chunks)]

    faiss_index.build(embeddings, chunks)#بناء الفهرس قبل البحث

    # Search with first embedding
    query = mock_embeddings[0]
    results, scores = faiss_index.search(query, k=3)
   # k=3 → نريد أفضل 3 نتائج
#results → chunks
#scores → similarity scores

    assert len(results) <= 3
    assert len(results) == len(scores)#كل نتيجة يجب أن يكون لها score
    assert all(isinstance(s, float) for s in scores)#كل score يجب أن يكون float


def test_faiss_index_save_load(faiss_index, mock_embeddings, sample_chunks):
    """Test saving and loading index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        index_dir = Path(tmpdir) / "index"#تحديد مكان حفظ الفهرس

        chunks = sample_chunks[: len(mock_embeddings)]
        embeddings = mock_embeddings[: len(chunks)]

        # Build and save
        faiss_index.build(embeddings, chunks)
        faiss_index.save(index_dir)

        # Load
        loaded_index = FAISSIndex.load(index_dir)

        assert loaded_index.index.ntotal == faiss_index.index.ntotal#التأكد أن عدد العناصر لم يتغير
        assert len(loaded_index.docstore) == len(faiss_index.docstore)#التأكد أن البيانات المرتبطة لم تضيع


def test_clean_text():
    """Test text cleaning."""
    text = "  This   has   extra   spaces.  "
    cleaned = clean_text(text)
    assert cleaned == "This has extra spaces."


def test_count_tokens_approximate():
    """Test token counting."""
    text = "This is a test sentence."
    count = count_tokens_approximate(text)
    assert count > 0
    assert isinstance(count, int)#يجب أن يكون عدد صحيح

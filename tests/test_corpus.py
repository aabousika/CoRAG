"""Tests for corpus module."""

import json
import tempfile  # لإنشاء ملفات ومجلدات مؤقتة أثناء الاختبار
from pathlib import Path

from corag.corpus.document import Chunk, Document
from corag.corpus.ingest import (
    CorpusIngestor,  # كلاس مسؤول عن قراءة البيانات (مثلاً من JSONL)
)


def test_document_to_dict(sample_document):
    """Test document serialization."""  # اختبار تحويل Document → dict
    doc_dict = sample_document.to_dict()  # تحويل المستند إلى dictionary
    assert doc_dict["id"] == "doc1"  # التأكد أن id صحيح
    assert doc_dict["title"] == "Test Document"
    assert "test document" in doc_dict["text"]  # التأكد أن النص موجود (جزئياً)


def test_document_from_dict(sample_document):
    """Test document deserialization."""
    doc_dict = sample_document.to_dict()  # تحويل إلى dict
    doc = Document.from_dict(doc_dict)  # إعادة تحويله إلى Document
    assert doc.id == sample_document.id  # التأكد أن id لم يتغير
    assert doc.title == sample_document.title


def test_chunk_to_dict(sample_chunk):
    """Test chunk serialization."""  # اختبار تحويل Chunk → dict
    chunk_dict = sample_chunk.to_dict()
    assert chunk_dict["chunk_id"] == "chunk1"
    assert chunk_dict["doc_id"] == "doc1"  # التحقق من ارتباطه بالمستند


def test_chunker_basic(chunker, sample_document):
    """Test basic chunking."""
    chunks = chunker.chunk_document(sample_document)
    assert len(chunks) > 0
    assert all(isinstance(c, Chunk) for c in chunks)  # كل عنصر يجب أن يكون من نوع Chunk
    assert all(
        c.doc_id == sample_document.id for c in chunks
    )  # كل chunk يجب أن ينتمي لنفس المستند


def test_chunker_multiple_documents(chunker, sample_documents):
    """Test chunking multiple documents."""
    chunks = chunker.chunk_documents(sample_documents)  # تقسيم كل المستندات
    assert len(chunks) > 0
    doc_ids = {c.doc_id for c in chunks}  # استخراج كل doc_id من chunks
    assert "doc1" in doc_ids
    assert "doc2" in doc_ids


def test_corpus_ingestor_jsonl():
    """Test JSONL ingestion."""  # اختبار قراءة ملف JSONL
    with tempfile.TemporaryDirectory() as tmpdir:  # إنشاء مجلد مؤقت
        # Create test JSONL file
        jsonl_path = Path(tmpdir) / "test.jsonl"
        with open(jsonl_path, "w") as f:
            f.write(json.dumps({"id": "1", "title": "Doc 1", "text": "Text 1"}) + "\n")
            f.write(json.dumps({"id": "2", "title": "Doc 2", "text": "Text 2"}) + "\n")

        ingestor = CorpusIngestor()
        docs = list(ingestor.ingest_jsonl(jsonl_path))

        assert len(docs) == 2
        assert docs[0].id == "1"
        assert docs[1].id == "2"


def test_corpus_ingestor_max_docs():
    """Test max docs limit."""
    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "test.jsonl"
        with open(jsonl_path, "w") as f:
            for i in range(10):  # إنشاء 10 مستندات
                f.write(
                    json.dumps({"id": str(i), "title": f"Doc {i}", "text": f"Text {i}"})
                    + "\n"
                )

        ingestor = CorpusIngestor()
        docs = list(
            ingestor.ingest_jsonl(jsonl_path, max_docs=3)
        )  # قراءة فقط أول 3 مستندات

        assert len(docs) == 3

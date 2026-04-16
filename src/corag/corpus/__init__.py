"""Corpus management and document processing."""

from corag.corpus.chunker import (
    Chunker,  # استيراد كلاس Chunker المسؤول عن تقسيم النصوص إلى أجزاء صغيرة (Chunks)
)
from corag.corpus.document import Chunk, Document

# Document: يمثل المستند الكامل
# Chunk: يمثل جزء صغير من المستند بعد التقسيم
from corag.corpus.ingest import (
    CorpusIngestor,  # استيراد كلاس مسؤول عن إدخال البيانات (قراءة الملفات وتحويلها إلى Documents وChunks)
)

__all__ = ["Document", "Chunk", "Chunker", "CorpusIngestor"]
# تحديد العناصر التي سيتم تصديرها عند استخدام import *
# هذا يجعل الملف يعمل كواجهة (Facade) ويوفر فقط هذه الكلاسات للمستخدم

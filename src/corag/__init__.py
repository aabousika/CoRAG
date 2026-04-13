"""CoRAG: Adaptive Multi-Step Retrieval for Complex Queries."""

__version__ = "0.1.0"
__author__ = "Angel(aabousika)"
__email__ = "aabousika@gmail.com"

from corag.controller.base import Controller
# المتحكم الرئيسي بالنظام (Brain)
# يدير تدفق العمل بين كل المكونات
from corag.corpus.chunker import Chunker# مسؤول عن تقسيم النصوص إلى أجزاء (Chunks)
from corag.corpus.document import Document# يمثل المستند (Document) كوحدة بيانات
from corag.generation.synthesizer import Synthesizer# مسؤول عن توليد الإجابة النهائية باستخدام LLM
from corag.indexing.embedder import Embedder# يحول النصوص إلى Embeddings (متجهات رقمية)
from corag.indexing.index import FAISSIndex# قاعدة بيانات للبحث (Vector DB) باستخدام FAISS
from corag.retrieval.pipeline import RetrievalPipeline# pipeline كامل لعملية الاسترجاع (Retrieval)

__all__ = [
    "Document",
    "Chunker",
    "Embedder",
    "FAISSIndex",
    "Controller",
    "RetrievalPipeline",
    "Synthesizer",
]
# تحديد المكونات الأساسية التي سيتم تصديرها للمستخدم
# عند استخدام:
# from corag import *
#
# الهدف:
# - إخفاء التفاصيل الداخلية
# - تقديم واجهة نظيفة للمشروع
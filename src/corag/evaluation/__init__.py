"""Evaluation framework for multi-hop QA."""

from corag.evaluation.datasets import DatasetLoader, EvalExample

# استيراد الكلاسات الخاصة بتحميل الداتا الخاصة بالتقييم (Datasets)
# DatasetLoader: مسؤول عن قراءة وتحميل بيانات التقييم (مثل HotpotQA)
# EvalExample: يمثل مثال واحد للتقييم (سؤال + إجابة صحيحة)
from corag.evaluation.evaluator import EvaluationReport, EvaluationResult, Evaluator

#استيراد الكلاسات المسؤولة عن عملية التقييم نفسها
# EvaluationReport: تقرير شامل يحتوي النتائج النهائية
# EvaluationResult: نتيجة تقييم واحدة (لسؤال واحد)
# Evaluator: الكلاس الأساسي الذي ينفذ عملية التقييم
from corag.evaluation.metrics import (  # استيراد دوال القياس (metrics) المستخدمة لتقييم أداء النموذج
    exact_match,  #تتحقق هل الإجابة مطابقة 100% للإجابة الصحيحة
    f1_score,  #تقيس التشابه بين الإجابة المتوقعة والصحيحة (precision + recall)
    max_em_over_ground_truths,  # في حال وجود عدة إجابات صحيحة، يأخذ أفضل exact match بينهم
    max_f1_over_ground_truths,  # في حال وجود عدة إجابات صحيحة، يأخذ أفضل F1 score بينهم
)

# تحديد ما الذي سيتم تصديره عند استخدام:
# from module import *
# هذا يساعد على تنظيم واجهة الموديول (API)
__all__ = [
    "DatasetLoader", # تصدير كلاس تحميل البيانات
    "EvalExample", # تصدير تمثيل المثال الواحد
    "Evaluator",# تصدير كلاس التقييم الرئيسي
    "EvaluationReport",# تصدير تقرير التقييم النهائي
    "EvaluationResult",# تصدير نتيجة كل مثال
    # تصدير دوال القياس
    "exact_match",
    "f1_score",
    "max_em_over_ground_truths",
    "max_f1_over_ground_truths",
]

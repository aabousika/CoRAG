"""LLM controller for multi-step retrieval orchestration."""

from corag.controller.base import Controller, GenerationConfig
# استيراد:
# - Controller: الكلاس الأساسي الذي يحتوي منطق التحكم العام
# - GenerationConfig: إعدادات توليد الإجابة (مثل temperature, max tokens...)
from corag.controller.openai_controller import OpenAIController
# استيراد كلاس خاص للتعامل مع OpenAI
# هذا الكلاس يطبّق Controller باستخدام API الخاص بـ OpenAI

__all__ = ["Controller", "GenerationConfig", "OpenAIController"]
# هذا السطر يحدد ما الذي سيتم تصديره عند استخدام:
# from module import *
# أي أنه يسمح فقط باستخدام هذه العناصر الثلاثة خارج الملف:
# - Controller
# - GenerationConfig
# - OpenAIController
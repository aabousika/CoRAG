"""OpenAI-compatible LLM controller."""

import logging
import time  #time → لاستخدام sleep في retry
from typing import Any, cast  #cast → لتحويل النوع (Type Casting)

import tiktoken
from openai import OpenAI  #استيراد client الرسمي للتعامل مع OpenAI API
from openai.types.chat import (
    ChatCompletionMessageParam,  #نوع البيانات المطلوب للـ messages في API
)

from corag.controller.base import Controller, GenerationConfig

#Controller (الواجهة الأساسية)
#- GenerationConfig (إعدادات التوليد)

logger = logging.getLogger(__name__)


class OpenAIController(Controller):
    """OpenAI-compatible LLM controller with rate limiting."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        api_base: str | None = None,#يمكن تحديد API مختلف (مثلاً proxy أو local endpoint)
        max_retries: int = 3, #عدد المحاولات عند الفشل
        retry_delay: float = 1.0, #الزمن بين المحاولات
    ):
        """Initialize OpenAI controller.

        Args:
            api_key: OpenAI API key
            model: Model name
            api_base: Custom API base URL
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if api_base:
            self.client = OpenAI(api_key=api_key, base_url=api_base)
        else:
            self.client = OpenAI(api_key=api_key)

        # Initialize tokenizer
        try:
            self.tokenizer = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fallback to cl100k_base for unknown models
            self.tokenizer = tiktoken.get_encoding("cl100k_base")

        logger.info(f"Initialized OpenAI controller with model {model}")

    def generate(
        self,
        prompt: str,
        system: str | None = None,
        config: GenerationConfig | None = None,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: User prompt
            system: System prompt
            config: Generation configuration

        Returns:
            Generated text
        """
        if config is None: #إذا لم يتم تمرير config → استخدم القيم الافتراضية
            config = GenerationConfig()

        messages: list[dict[str, Any]] = []#إنشاء قائمة messages (كما يطلب OpenAI API)
        if system:#إضافة system prompt (اختياري)
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})#إضافة prompt المستخدم

        # Cast messages to the expected type for OpenAI API
        typed_messages = cast(list[ChatCompletionMessageParam], messages)

        # Retry loop
        for attempt in range(self.max_retries):
            try: #محاولة تنفيذ الطلب
                if config.stop: #إذا هناك stop sequences
                    response = self.client.chat.completions.create(#إرسال الطلب إلى OpenAI
                        model=self.model,
                        messages=typed_messages,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        top_p=config.top_p,
                        stop=config.stop,
                        seed=config.seed,
                    )
                else:
                    response = self.client.chat.completions.create(#نفس الطلب بدون stop
                        model=self.model,
                        messages=typed_messages,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        top_p=config.top_p,
                        seed=config.seed,
                    )

                text: str = response.choices[0].message.content or ""#استخراج النص من الاستجابة
                if not text:
                    raise ValueError("Response content is None or empty")

                logger.debug(
                    f"Generated {len(text)} chars, "
                    f"tokens: {response.usage.total_tokens if response.usage else 0}"
                )#تسجيل معلومات (عدد الأحرف والتوكنز)

                return text

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retries - 1:#الانتظار قبل إعادة المحاولة (زيادة تدريجية)
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise#إذا انتهت المحاولات → يرمي الخطأ

        raise RuntimeError("Failed to generate after all retries")#في حال فشل كامل

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens

        Returns:
            Number of tokens
        """
        return len(self.tokenizer.encode(text))#تحويل النص إلى tokens ثم حساب العدد

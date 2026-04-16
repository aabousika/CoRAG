"""Base controller interface for LLM providers."""

from abc import ABC, abstractmethod

# ABC = Abstract Base Class → لإنشاء كلاس مجرد (لا يمكن استخدامه مباشرة)
# abstractmethod = لإجبار الكلاسات التي ترث منه على تنفيذ دوال معينة
from dataclasses import dataclass

# dataclass = لتسهيل إنشاء كلاس مخصص لتخزين البيانات (بدون كتابة constructor يدوي)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

    temperature: float = 0.2
    max_tokens: int = 2048
    top_p: float = 1.0  # أسلوب sampling (nucleus sampling)
    stop: list[str] | None = None  # قائمة كلمات إذا ظهرت يتوقف التوليد عندها
    seed: int | None = None  # seed لتثبيت النتائج (reproducibility)


class Controller(ABC):
    """Abstract base class for LLM controllers."""

    @abstractmethod
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
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens

        Returns:
            Number of tokens
        """
        pass

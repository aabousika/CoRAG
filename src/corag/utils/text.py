"""Text processing utilities."""

import re
# استيراد مكتبة re الخاصة بـ Regular Expressions (التعابير النمطية)
# تُستخدم للبحث والتعديل داخل النصوص بشكل متقدم



def clean_text(text: str) -> str:
    """Clean and normalize text."""
    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)
    # \s+ تعني: أي عدد من الفراغات (space, tab, newline)
    # يتم استبدالها بفراغ واحد فقط
    # مثال: "hello    world" → "hello world"-----
    # Remove leading/trailing whitespace
    text = text.strip()
    # حذف الفراغات من بداية ونهاية النص
    # مثال: "  hello  " → "hello"---------------
    # Normalize unicode quotes
    text = text.replace("\u201c", '"').replace("\u201d", '"') # استبدال علامات الاقتباس الذكية “ ” بعلامات اقتباس عادية "
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text


def count_tokens_approximate(text: str) -> int:
    """Approximate token count (words + punctuation)."""
    # Simple approximation: split on whitespace and count
    # More accurate would use tiktoken but this is faster
    tokens = text.split()
    # تقسيم النص إلى كلمات بناءً على الفراغات
    # مثال: "hello world" → ["hello", "world"]------
    # Add some overhead for punctuation
    return int(len(tokens) * 1.3)
     # ضرب عدد الكلمات بـ 1.3 لتعويض علامات الترقيم
    # لأنه split() لا يحسبها بشكل دقيق


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # تقسيم النص عند:
    # نقطة . أو ! أو ؟
    # ثم فراغ بعدها
    #
    # (?<=...) = lookbehind (تأكد أن قبل الفراغ يوجد علامة نهاية جملة)
    #
    # مثال:
    # "Hi. How are you?" →
    # ["Hi.", "How are you?"]
    return [s.strip() for s in sentences if s.strip()]
    # إزالة الفراغات من كل جملة
    # حذف الجمل الفارغة


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate text to max characters at word boundary."""
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars].rsplit(" ", 1)[0]
    # يأخذ أول max_chars حرف
    # ثم يقص إلى آخر كلمة كاملة (بدون قطعها)
    # rsplit(" ", 1) = تقسيم من اليمين مرة واحدة

    return truncated + "..."
    # إضافة "..." للدلالة أن النص تم قطعه


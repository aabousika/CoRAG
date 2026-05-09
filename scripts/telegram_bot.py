"""
CoRAG Legal Search - Telegram Bot
Connects CoRAG legal search system to Telegram
"""
import logging
import subprocess
import json
import os
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import os
from dotenv import load_dotenv

load_dotenv()
# ===================== إعدادات =====================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
INDEX_DIR = "data/index_trained"
MAX_STEPS = 6
K = 8

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# ===================== أوامر الـ Bot =====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚖️ *CoRAG Legal Search Bot*\n\n"
        "مرحباً! أنا نظام بحث قانوني ذكي.\n\n"
        "أرسل سؤالك القانوني وسأجيبك مع المراجع!\n\n"
        "مثال:\n"
        "_What are the principles for awarding indemnity costs?_\n\n"
        "الأوامر:\n"
        "/start — بدء المحادثة\n"
        "/help — المساعدة\n"
        "/about — معلومات عن النظام",
        parse_mode='Markdown'
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📚 *كيفية الاستخدام*\n\n"
        "1. أرسل سؤالك القانوني بالإنجليزية\n"
        "2. انتظر ~30 ثانية\n"
        "3. ستصلك إجابة مفصّلة مع مراجع قانونية\n\n"
        "⚠️ النظام متخصص في القانون الأسترالي فقط",
        parse_mode='Markdown'
    )

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *CoRAG — Adaptive Multi-Step Retrieval*\n\n"
        "📊 *إحصائيات النظام:*\n"
        "• Hit Rate: 90% بعد Fine-tuning\n"
        "• F1 Score: 56.4% على HotpotQA\n"
        "• Exact Match: 34.0%\n"
        "• 37,393 متجه قانوني\n\n"
        "🛠 *التقنيات:*\n"
        "• FAISS + Sentence Transformers\n"
        "• GPT-4o-mini\n"
        "• FastAPI\n\n"
        "👨‍💻 *إعداد:* Angel Abou Sika\n"
        "🎓 *إشراف:* Dr. Tarek Barhom",
        parse_mode='Markdown'
    )

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text

    # رسالة انتظار
    waiting_msg = await update.message.reply_text(
        "⏳ جاري البحث في قاعدة البيانات القانونية...\n"
        "انتظر ~30 ثانية"
    )

    try:
        # خطوة 1 — البحث
        search_result = subprocess.run([
            'corag-search',
            '--question', question,
            '--index-dir', INDEX_DIR,
            '--max-steps', str(MAX_STEPS),
            '--k', str(K),
            '--output', 'data/telegram_trace.json'
        ], capture_output=True, text=True, cwd='/Users/angel/Desktop/CoRAG')

        # خطوة 2 — توليد الإجابة
        answer_result = subprocess.run([
            'corag-answer',
            '--trace', 'data/telegram_trace.json',
            '--output', 'data/telegram_answer.md'
        ], capture_output=True, text=True, cwd='/Users/angel/Desktop/CoRAG')

        # قراءة الإجابة
        with open('/Users/angel/Desktop/CoRAG/data/telegram_answer.md', 'r') as f:
            answer = f.read()

        # قراءة المراجع من الـ trace
        with open('/Users/angel/Desktop/CoRAG/data/telegram_trace.json', 'r') as f:
            trace = json.load(f)

        # تقصير الإجابة لـ Telegram (حد 4096 حرف)
        if len(answer) > 3000:
            answer = answer[:3000] + "\n\n..."

        # حذف رسالة الانتظار
        await waiting_msg.delete()

        # إرسال الإجابة
        await update.message.reply_text(
            f"⚖️ *الإجابة:*\n\n{answer}",
            parse_mode='Markdown'
        )

        # إرسال الإحصائيات
        num_steps = len(trace.get('steps', []))
        num_chunks = trace.get('total_chunks_retrieved', 0)
        await update.message.reply_text(
            f"📊 *إحصائيات البحث:*\n"
            f"• خطوات البحث: {num_steps}\n"
            f"• القطع المسترجعة: {num_chunks}\n"
            f"• الـ Index: Legal (Fine-tuned)",
            parse_mode='Markdown'
        )

    except Exception as e:
        await waiting_msg.delete()
        await update.message.reply_text(
            "❌ حدث خطأ أثناء البحث\n"
            "تأكد من أن السيرفر يعمل وأن السؤال بالإنجليزية"
        )
        logging.error(f"Error: {e}")

# ===================== تشغيل الـ Bot =====================
def main():
    print("🚀 CoRAG Telegram Bot يعمل...")
    print("افتح Telegram وابحث عن @corag_legal_bot")
    print("اضغط Ctrl+C لإيقاف الـ Bot")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()

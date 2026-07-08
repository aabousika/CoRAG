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
# settings
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
INDEX_DIR = "data/index_trained"
MAX_STEPS = 6
K = 8

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Bot Commands
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "⚖️ *CoRAG Legal Search Bot*\n\n"
        "Hello! I am an intelligent legal search system.\n\n"
        "Send your legal question and I will answer it with references!\n\n"
        "Example:\n"
        "_What are the principles for awarding indemnity costs?_\n\n"
        "Commands:\n"
        "/start — Start the conversation\n"
        "/help — Help\n"
        "/about — Information about the system",
        parse_mode='Markdown'#يعني أن تيليجرام سيقوم بتنسيق النص باستخدام Markdown بدل عرضه كنص عادي.
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "📚 How to Use\n\n"
        "1. Send your legal question in English\n"
        "2. Wait about ~30 seconds\n"
        "3. You will receive a detailed answer with legal references\n\n"
        "⚠️ The system specializes in Australian law only",
        parse_mode='Markdown'
    )

async def about(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🤖 *CoRAG — Adaptive Multi-Step Retrieval*\n\n"
        "📊 System Statistics:\n"
        "• Hit Rate: 96%. after Fine-tuning\n"
        "• F1 Score: 56.4%. on HotpotQA\n"
        "• Exact Match: 34.0%\n"
        "• 37,393 legal vectors\n\n"
        "🛠 Technologies:\n"
        "• FAISS + Sentence Transformers\n"
        "• GPT-4o-mini\n"
        "• FastAPI\n\n"
        "👨‍💻 Developed by: Angel Abou Sika\n"
        "🎓 Supervised by: Dr. Tarek Barhom",
        parse_mode='Markdown'
    )

async def handle_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    question = update.message.text

    # wait message
    waiting_msg = await update.message.reply_text(
        "⏳ Searching the legal database...\n"
        "Please wait about ~30 seconds",
    )

    try:
        # search - step1
        search_result = subprocess.run([
            'corag-search',
            '--question', question,
            '--index-dir', INDEX_DIR,
            '--max-steps', str(MAX_STEPS),
            '--k', str(K),
            '--output', 'data/telegram_trace.json'
        ], capture_output=True, text=True, cwd='/Users/angel/Desktop/CoRAG')

        # generate answer - step2
        answer_result = subprocess.run([
            'corag-answer',
            '--trace', 'data/telegram_trace.json',
            '--output', 'data/telegram_answer.md'
        ], capture_output=True, text=True, cwd='/Users/angel/Desktop/CoRAG')

        # read answer
        with open('/Users/angel/Desktop/CoRAG/data/telegram_answer.md', 'r') as f:
            answer = f.read()

        # read reference from trace
        with open('/Users/angel/Desktop/CoRAG/data/telegram_trace.json', 'r') as f:
            trace = json.load(f)

        # Trimming the answer for Telegram (4096 character limit)
        if len(answer) > 3000:
            answer = answer[:3000] + "\n\n..."

        # delet wait message
        await waiting_msg.delete()

        # send answer
        await update.message.reply_text(
            f"⚖️ *answer:*\n\n{answer}",
            parse_mode='Markdown'
        )

        # Sending statistics
        num_steps = len(trace.get('steps', []))
        num_chunks = trace.get('total_chunks_retrieved', 0)
        await update.message.reply_text(
            f"📊 *Search Statistics:*\n"
            f"• Search steps: {num_steps}\n"
            f"• Retrieved chunks: {num_chunks}\n"
            f"• Index: Legal (Fine-tuned)",
            parse_mode='Markdown'
        )

    except Exception as e:
        await waiting_msg.delete()
        await update.message.reply_text(
            "An error occurred during search\n"
            "Make sure the server is running and the question is in English"
        )
        logging.error(f"Error: {e}")

# run bot
def main():
    print(" CoRAG Telegram Bot is running...")
    print("Open Telegram and search for @corag_legal_bot")
    print("Press Ctrl+C to stop the bot")

    app = Application.builder().token(TELEGRAM_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("about", about))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_question))

    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()

"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           BOJXONA PASSPORT SCANNER - TELEGRAM BOT                            ║
║                                                                              ║
║  ✅ To'g'ridan-to'g'ri rasm skanerlash (Mini App shart emas)                ║
║  ✅ DeepSeek OCR - DeepSeek-OCR-Latest-BF16.I64 model                        ║
║  ✅ Tesseract OCR - Fallback engine                                          ║
║  ✅ ICAO 9303 TD3 - Xalqaro standart                                         ║
║                                                                              ║
║  Versiya: 3.0.0                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import io
import asyncio
import logging
from telegram import Update, WebAppInfo, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
import uvicorn
from threading import Thread

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Bot configuration
BOT_TOKEN = "8319403923:AAH8LkaqsickGL980lJKEOk51tVyhI6onZA"
WEBAPP_URL = os.environ.get("RAILWAY_STATIC_URL", "http://localhost:8000")

# Ensure HTTPS for production Railway deployments
if "railway.app" in WEBAPP_URL:
    if not WEBAPP_URL.startswith(("http://", "https://")):
        WEBAPP_URL = f"https://{WEBAPP_URL}"
    elif WEBAPP_URL.startswith("http://"):
        WEBAPP_URL = WEBAPP_URL.replace("http://", "https://")


# ============================================
# LOKAL SCANNER IMPORT
# ============================================

def get_local_scanner():
    """Lokal scanner instance olish"""
    try:
        from main import LocalPassportScanner
        return LocalPassportScanner()
    except Exception as e:
        logger.error(f"Scanner yuklanmadi: {e}")
        return None


# Global scanner instance (lazy loading)
_scanner = None

def get_scanner():
    """Scanner instance olish (singleton)"""
    global _scanner
    if _scanner is None:
        _scanner = get_local_scanner()
    return _scanner


# ============================================
# BOT COMMAND HANDLERS
# ============================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command - Show Mini App button and direct scan option"""
    user = update.effective_user

    welcome_message = (
        f"🛂 <b>Bojxona Passport Scanner</b>\n\n"
        f"Salom, {user.first_name}! 👋\n\n"
        f"Bu bot O'zbekiston Respublikasi Bojxona qo'mitasi uchun "
        f"biometrik pasportlarni skanerlash tizimi.\n\n"
        f"<b>Imkoniyatlar:</b>\n"
        f"✅ ICAO 9303 TD3 standartiga mos MRZ skanerlash\n"
        f"✅ Pasport raqami, JSHSHIR/PNFL ekstraktsiyasi\n"
        f"✅ Avtomatik tekshiruv va validatsiya\n"
        f"✅ DeepSeek OCR - Yuqori aniqlikdagi skanerlash\n\n"
        f"<b>Foydalanish:</b>\n"
        f"📷 <i>Pasport rasmini to'g'ridan-to'g'ri yuboring</i>\n"
        f"<i>yoki Mini App tugmasini bosing</i>"
    )

    # Create Web App button
    keyboard = [
        [InlineKeyboardButton(
            "📷 Mini App orqali skanerlash",
            web_app=WebAppInfo(url=WEBAPP_URL)
        )]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        welcome_message,
        parse_mode='HTML',
        reply_markup=reply_markup
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    help_text = (
        "📖 <b>Yordam</b>\n\n"
        "<b>Skanerlash usullari:</b>\n\n"
        "<b>1. To'g'ridan-to'g'ri rasm yuborish:</b>\n"
        "   • Pasport rasmini botga yuboring\n"
        "   • Bot avtomatik skanerlaydi\n"
        "   • Natija darhol ko'rsatiladi\n\n"
        "<b>2. Mini App orqali:</b>\n"
        "   • /start buyrug'ini bering\n"
        "   • Mini App tugmasini bosing\n"
        "   • Kamerani yoqing va rasmga oling\n\n"
        "<b>Muhim:</b>\n"
        "• MRZ zonasi (pastki qism) aniq ko'rinishi kerak\n"
        "• Yaxshi yoritilgan rasm yuboring\n"
        "• Pasport to'g'ri joylashgan bo'lsin\n\n"
        "<b>Buyruqlar:</b>\n"
        "/start - Boshlash\n"
        "/help - Yordam\n"
        "/info - Tizim haqida\n\n"
        "<b>Versiya:</b> 3.0.0 (DeepSeek OCR)\n"
        "<b>Standard:</b> ICAO 9303 TD3"
    )

    await update.message.reply_text(help_text, parse_mode='HTML')


async def info_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /info command - System information"""
    info_text = (
        "ℹ️ <b>Tizim Ma'lumotlari</b>\n\n"
        "<b>Nomi:</b> Bojxona Passport Scanner\n"
        "<b>Versiya:</b> 3.0.0 (DeepSeek OCR)\n"
        "<b>Standard:</b> ICAO 9303 TD3\n"
        "<b>OCR Engine:</b> DeepSeek OCR (DeepSeek-OCR-Latest-BF16.I64)\n"
        "<b>Fallback:</b> Tesseract OCR\n"
        "<b>Backend:</b> FastAPI + Python\n"
        "<b>Frontend:</b> Telegram Mini App\n\n"
        "<b>Xavfsizlik:</b>\n"
        "🔒 DeepSeek OCR - Xavfsiz API orqali\n"
        "🔒 Tesseract fallback - Lokal qayta ishlash\n\n"
        "<b>Qo'llab-quvvatlanadigan hujjatlar:</b>\n"
        "• O'zbekiston Respublikasi Biometrik Pasporti\n\n"
        "<b>Ekstraktsiya qilinadigan ma'lumotlar:</b>\n"
        "• Pasport raqami\n"
        "• Familiya va Ism\n"
        "• Tug'ilgan sana\n"
        "• Jins (M/F)\n"
        "• Amal qilish muddati\n"
        "• JSHSHIR/PNFL (Shaxsiy raqam)\n"
        "• Millat\n\n"
        "<b>Validatsiya:</b>\n"
        "✓ ICAO 9303 Checksum (Modulo 10)\n"
        "✓ Sana formatlari\n"
        "✓ Composite checkdigit"
    )

    await update.message.reply_text(info_text, parse_mode='HTML')


# ============================================
# PHOTO HANDLER - TO'G'RIDAN-TO'G'RI SKANERLASH
# ============================================

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handle photos sent directly to the bot
    DeepSeek OCR ishlatib skanerlash (Tesseract fallback)
    """
    user = update.effective_user
    logger.info(f"📸 Rasm qabul qilindi: {user.first_name} ({user.id})")

    # "Skanerlash..." xabari
    processing_msg = await update.message.reply_text(
        "🔍 <b>Skanerlash...</b>\n\n"
        "⏳ MRZ zonasi qidirilmoqda...",
        parse_mode='HTML'
    )

    try:
        # Rasmni yuklab olish (eng katta versiyasini)
        photo = update.message.photo[-1]  # Eng katta o'lcham
        file = await context.bot.get_file(photo.file_id)

        # Rasmni bytes ga o'qish
        image_bytes = await file.download_as_bytearray()
        image_bytes = bytes(image_bytes)

        logger.info(f"📥 Rasm yuklandi: {len(image_bytes)} bytes")

        # Scanner olish
        scanner = get_scanner()
        if scanner is None:
            await processing_msg.edit_text(
                "❌ <b>Xato</b>\n\n"
                "Scanner yuklanmadi. Iltimos, qaytadan urinib ko'ring.",
                parse_mode='HTML'
            )
            return

        # Skanerlash
        result = scanner.scan(image_bytes, f"telegram_{user.id}.jpg")

        logger.info(f"✅ Skanerlash muvaffaqiyatli: {result.get('passport_number')}")

        # Natijani formatlash
        validation_status = result.get('validation_status', 'UNKNOWN')
        status_emoji = "✅" if validation_status == "PASS" else "⚠️"

        response_text = (
            f"🛂 <b>Pasport Ma'lumotlari</b>\n\n"
            f"<b>Pasport raqami:</b> <code>{result.get('passport_number', 'N/A')}</code>\n"
            f"<b>Familiya:</b> {result.get('surname', 'N/A')}\n"
            f"<b>Ism:</b> {result.get('given_names', 'N/A')}\n"
            f"<b>Tug'ilgan sana:</b> {result.get('birth_date', 'N/A')}\n"
            f"<b>Jinsi:</b> {result.get('sex', 'N/A')}\n"
            f"<b>Amal qilish:</b> {result.get('expiry_date', 'N/A')}\n"
            f"<b>JSHSHIR:</b> <code>{result.get('pinfl', 'N/A')}</code>\n"
            f"<b>Fuqarolik:</b> {result.get('nationality', 'N/A')}\n\n"
            f"{status_emoji} <b>Validatsiya:</b> {validation_status}\n\n"
            f"<i>🤖 DeepSeek OCR - Yuqori aniqlik bilan skanerlandi</i>"
        )

        # Validatsiya tafsilotlarini qo'shish
        validations = result.get('validations', {})
        if validations:
            validation_details = []
            for key, value in validations.items():
                if key != 'all_valid':
                    emoji = "✓" if value else "✗"
                    name = key.replace('_valid', '').replace('_', ' ').title()
                    validation_details.append(f"{emoji} {name}")

            if validation_details:
                response_text += f"\n\n<b>Tekshiruvlar:</b>\n" + "\n".join(validation_details)

        await processing_msg.edit_text(response_text, parse_mode='HTML')

    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Skanerlash xatosi: {error_msg}")

        # Foydalanuvchiga xato xabarini ko'rsatish
        if "MRZ topilmadi" in error_msg:
            await processing_msg.edit_text(
                "❌ <b>MRZ topilmadi</b>\n\n"
                "Pasportning MRZ zonasi (pastki qismdagi 2 qator) "
                "rasmda aniq ko'rinishi kerak.\n\n"
                "<b>Maslahatlar:</b>\n"
                "• Pasportni tekis yuzaga qo'ying\n"
                "• Yaxshi yoritilgan joyda rasmga oling\n"
                "• MRZ zonasi to'liq ko'rinsin\n"
                "• Rasmni aniqroq oling",
                parse_mode='HTML'
            )
        else:
            await processing_msg.edit_text(
                f"❌ <b>Xato</b>\n\n"
                f"Skanerlashda xato yuz berdi:\n"
                f"<code>{error_msg[:200]}</code>\n\n"
                f"Iltimos, boshqa rasm bilan qaytadan urinib ko'ring.",
                parse_mode='HTML'
            )


# ============================================
# DOCUMENT HANDLER - Fayl sifatida yuborilgan rasmlar
# ============================================

async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle documents (images sent as files)"""
    document = update.message.document

    # Rasm formatlarini tekshirish
    if document.mime_type and document.mime_type.startswith('image/'):
        user = update.effective_user
        logger.info(f"📎 Fayl qabul qilindi: {document.file_name} ({user.first_name})")

        # "Skanerlash..." xabari
        processing_msg = await update.message.reply_text(
            "🔍 <b>Skanerlash...</b>\n\n"
            "⏳ MRZ zonasi qidirilmoqda...",
            parse_mode='HTML'
        )

        try:
            # Faylni yuklab olish
            file = await context.bot.get_file(document.file_id)
            image_bytes = await file.download_as_bytearray()
            image_bytes = bytes(image_bytes)

            logger.info(f"📥 Fayl yuklandi: {len(image_bytes)} bytes")

            # Scanner olish
            scanner = get_scanner()
            if scanner is None:
                await processing_msg.edit_text(
                    "❌ <b>Xato</b>\n\n"
                    "Scanner yuklanmadi. Iltimos, qaytadan urinib ko'ring.",
                    parse_mode='HTML'
                )
                return

            # Skanerlash
            result = scanner.scan(image_bytes, document.file_name or "document.jpg")

            logger.info(f"✅ Skanerlash muvaffaqiyatli: {result.get('passport_number')}")

            # Natijani formatlash
            validation_status = result.get('validation_status', 'UNKNOWN')
            status_emoji = "✅" if validation_status == "PASS" else "⚠️"

            response_text = (
                f"🛂 <b>Pasport Ma'lumotlari</b>\n\n"
                f"<b>Pasport raqami:</b> <code>{result.get('passport_number', 'N/A')}</code>\n"
                f"<b>Familiya:</b> {result.get('surname', 'N/A')}\n"
                f"<b>Ism:</b> {result.get('given_names', 'N/A')}\n"
                f"<b>Tug'ilgan sana:</b> {result.get('birth_date', 'N/A')}\n"
                f"<b>Jinsi:</b> {result.get('sex', 'N/A')}\n"
                f"<b>Amal qilish:</b> {result.get('expiry_date', 'N/A')}\n"
                f"<b>JSHSHIR:</b> <code>{result.get('pinfl', 'N/A')}</code>\n"
                f"<b>Fuqarolik:</b> {result.get('nationality', 'N/A')}\n\n"
                f"{status_emoji} <b>Validatsiya:</b> {validation_status}\n\n"
                f"<i>🤖 DeepSeek OCR - Yuqori aniqlik bilan skanerlandi</i>"
            )

            await processing_msg.edit_text(response_text, parse_mode='HTML')

        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Skanerlash xatosi: {error_msg}")

            if "MRZ topilmadi" in error_msg:
                await processing_msg.edit_text(
                    "❌ <b>MRZ topilmadi</b>\n\n"
                    "Pasportning MRZ zonasi rasmda aniq ko'rinishi kerak.\n\n"
                    "<b>Maslahatlar:</b>\n"
                    "• Pasportni tekis yuzaga qo'ying\n"
                    "• Yaxshi yoritilgan joyda rasmga oling\n"
                    "• MRZ zonasi to'liq ko'rinsin",
                    parse_mode='HTML'
                )
            else:
                await processing_msg.edit_text(
                    f"❌ <b>Xato</b>\n\n"
                    f"<code>{error_msg[:200]}</code>\n\n"
                    f"Iltimos, qaytadan urinib ko'ring.",
                    parse_mode='HTML'
                )
    else:
        # Rasm emas
        await update.message.reply_text(
            "⚠️ Iltimos, pasport <b>rasmini</b> yuboring.\n\n"
            "Qo'llab-quvvatlanadigan formatlar: JPG, PNG, WEBP, BMP",
            parse_mode='HTML'
        )


# ============================================
# ERROR HANDLER
# ============================================

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors including Telegram conflicts"""
    error_msg = str(context.error)

    # Handle Telegram Conflict (multiple bot instances)
    if "Conflict" in error_msg or "terminated by other getUpdates" in error_msg:
        logger.warning("⚠️ Telegram Conflict detected - another bot instance is running")
        logger.warning("This deployment will continue and override the previous instance")
        return

    logger.error(f"Exception while handling an update: {context.error}")

    if update and update.effective_message:
        await update.effective_message.reply_text(
            "❌ Xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring yoki /help ni ko'ring."
        )


# ============================================
# APPLICATION SETUP
# ============================================

# Global application instance (shared with main.py for webhooks)
telegram_application = None

def get_application():
    """Get or create the Telegram application instance"""
    global telegram_application
    if telegram_application is None:
        telegram_application = Application.builder().token(BOT_TOKEN).build()

        # Register handlers
        telegram_application.add_handler(CommandHandler("start", start_command))
        telegram_application.add_handler(CommandHandler("help", help_command))
        telegram_application.add_handler(CommandHandler("info", info_command))

        # Photo handler - to'g'ridan-to'g'ri skanerlash
        telegram_application.add_handler(MessageHandler(filters.PHOTO, handle_photo))

        # Document handler - fayl sifatida yuborilgan rasmlar
        telegram_application.add_handler(MessageHandler(filters.Document.ALL, handle_document))

        telegram_application.add_error_handler(error_handler)

    return telegram_application


def run_fastapi():
    """Run FastAPI server in a separate thread"""
    from main import app as fastapi_app
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        fastapi_app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )


async def main():
    """Main function to start bot and FastAPI server"""
    # Start FastAPI in background thread
    fastapi_thread = Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()
    logger.info("FastAPI server started in background thread")

    # Small delay to let FastAPI start
    await asyncio.sleep(2)

    # Build Telegram bot application
    application = get_application()

    # Start bot
    logger.info("Starting Telegram bot...")
    await application.initialize()
    await application.start()

    # Check if we're running on Railway (use webhooks) or locally (use polling)
    is_railway = "railway.app" in WEBAPP_URL or os.environ.get("RAILWAY_ENVIRONMENT")

    if is_railway:
        # Use webhooks for Railway deployment (prevents conflicts)
        webhook_url = f"{WEBAPP_URL}/telegram-webhook"
        logger.info(f"🌐 Setting up webhook at: {webhook_url}")

        # Set webhook
        await application.bot.set_webhook(
            url=webhook_url,
            drop_pending_updates=True
        )

        logger.info("✅ Bojxona Passport Scanner is now running in WEBHOOK mode!")
        logger.info(f"📱 Mini App URL: {WEBAPP_URL}")
        logger.info(f"🔗 Webhook: {webhook_url}")

    else:
        # Use polling for local development
        logger.info("🔄 Running in POLLING mode (local development)")

        # Delete webhook to ensure clean polling
        logger.info("Clearing any existing webhooks...")
        await application.bot.delete_webhook(drop_pending_updates=True)

        await application.updater.start_polling(drop_pending_updates=True)

        logger.info("✅ Bojxona Passport Scanner is now running in POLLING mode!")
        logger.info(f"📱 Mini App URL: {WEBAPP_URL}")

    # Keep running
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        if not is_railway and application.updater.running:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()


if __name__ == "__main__":
    asyncio.run(main())

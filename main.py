"""
Telegram-бот: aiogram 3 + Groq (OpenAI-совместимый API).
- В группах: ответ на @бот (берётся текст после упоминания), ответ боту, /команда@бот.
- Inline: в любом чате набери @username_бота и текст запроса — выбери результат, чтобы вставить ответ.
  Включи у @BotFather: /setinline → выбрать бота → задать placeholder.
- Поддержка фото: отправь фото с текстом или без — бот проанализирует изображение.
- Память диалога: бот помнит контекст разговора в личных сообщениях.
"""

import asyncio
import base64
import hashlib
import logging
import os
import re
import sys
from io import BytesIO
from typing import List, Dict, Any

from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from aiogram import Bot, Dispatcher, F
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ChatType, MessageEntityType
from aiogram.exceptions import TelegramNetworkError, TelegramNotFound
from aiogram.filters import BaseFilter, Command, CommandStart
from aiogram.types import (
    InlineQuery,
    InlineQueryResultArticle,
    InputTextMessageContent,
    Message,
    ReplyKeyboardMarkup,
    KeyboardButton,
    PhotoSize,
)
from dotenv import load_dotenv
from openai import APIStatusError, AsyncOpenAI, RateLimitError

load_dotenv()

def _env_str(name: str) -> str | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    v = raw.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        v = v[1:-1].strip()
    return v or None


TELEGRAM_BOT_TOKEN = _env_str("TELEGRAM_BOT_TOKEN")
TELEGRAM_PROXY = _env_str("TELEGRAM_PROXY")
GROQ_API_KEY = _env_str("GROQ_API_KEY")
GROQ_MODEL = _env_str("GROQ_MODEL") or "llama-3.3-70b-versatile"
GROQ_VISION_MODEL = _env_str("GROQ_VISION_MODEL") or "llama-3.2-90b-vision-preview"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Хранилище истории диалогов
chat_history: Dict[int, List[Dict[str, str]]] = {}
MAX_HISTORY_LENGTH = 1000

if not TELEGRAM_BOT_TOKEN:
    print("Задайте TELEGRAM_BOT_TOKEN в файле .env", file=sys.stderr)
    sys.exit(1)

if ":" not in TELEGRAM_BOT_TOKEN or not TELEGRAM_BOT_TOKEN.split(":", 1)[0].isdigit():
    print(
        "TELEGRAM_BOT_TOKEN выглядит неверно. Ожидается формат: 123456789:AAH... "
        "(без пробелов; один токен от @BotFather).",
        file=sys.stderr,
    )
    sys.exit(1)

if not GROQ_API_KEY:
    print("Задайте GROQ_API_KEY в .env (https://console.groq.com/keys )", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

llm_client = AsyncOpenAI(api_key=GROQ_API_KEY, base_url=GROQ_BASE_URL)

if TELEGRAM_PROXY:
    try:
        bot = Bot(token=TELEGRAM_BOT_TOKEN, session=AiohttpSession(proxy=TELEGRAM_PROXY))
    except RuntimeError as e:
        print(
            "Задан TELEGRAM_PROXY, но не установлен пакет прокси.\n"
            "Выполни: pip install aiohttp-socks",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    logger.info("Используется TELEGRAM_PROXY для api.telegram.org")
else:
    bot = Bot(token=TELEGRAM_BOT_TOKEN)

dp = Dispatcher()

SYSTEM_PROMPT = (
    "Ты дружелюбный ассистент в Telegram. Отвечай кратко и по делу, "
    "на языке пользователя. Если просят подробно, отвечай подробно. "
    "Если тебе отправляют фото — опиши что на нём изображено и ответь на вопросы пользователя. "
    "Помни контекст предыдущих сообщений в этом диалоге."
)


class AddressedToBot(BaseFilter):
    """В личке — всегда; в группе/супергруппе — @бот, ответ боту или /команда@бот."""

    async def __call__(self, message: Message, bot: Bot) -> bool:
        if message.chat.type == ChatType.PRIVATE:
            return True
        if message.chat.type not in (ChatType.GROUP, ChatType.SUPERGROUP):
            return False

        me = await bot.get_me()
        if not me.username:
            return False
        un = me.username.lower()

        rp = message.reply_to_message
        if rp and rp.from_user and rp.from_user.id == me.id:
            return True

        text = message.text or message.caption or ""
        if text.startswith("/"):
            first = text.split(None, 1)[0]
            if "@" in first:
                _, _, suffix = first.partition("@")
                if suffix.lower() == un:
                    return True

        full = message.text or message.caption or ""
        for e in (*(message.entities or ()), *(message.caption_entities or ())):
            if e.type == MessageEntityType.MENTION:
                frag = full[e.offset : e.offset + e.length]
                if frag.lstrip("@").lower() == un:
                    return True
            if e.type == MessageEntityType.TEXT_MENTION and e.user and e.user.id == me.id:
                return True

        return False


addressed = AddressedToBot()


class NotTelegramCommand(BaseFilter):
    """Не обрабатывать как диалог строки, где в начале стоит /команда (entity bot_command)."""

    async def __call__(self, message: Message) -> bool:
        for e in message.entities or ():
            if e.type == MessageEntityType.BOT_COMMAND and e.offset == 0:
                return False
        return True


not_cmd = NotTelegramCommand()


def _strip_self_mentions(text: str, bot_username: str) -> str:
    pattern = re.compile(re.escape("@" + bot_username), re.IGNORECASE)
    cleaned = pattern.sub("", text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


async def _prompt_for_llm(message: Message, bot: Bot) -> str:
    """В личке — весь текст. В группе после @бот — только хвост после первого упоминания; ответ боту — весь текст."""
    text = (message.text or message.caption or "").strip()
    if message.chat.type == ChatType.PRIVATE:
        return text

    me = await bot.get_me()
    if not me.username:
        return text

    un = me.username.lower()
    uid = me.id

    rp = message.reply_to_message
    if rp and rp.from_user and rp.from_user.id == uid:
        return text

    mention_end: int | None = None
    for e in sorted(message.entities or (), key=lambda x: x.offset):
        if e.type == MessageEntityType.MENTION:
            frag = text[e.offset : e.offset + e.length]
            if frag.lstrip("@").lower() == un:
                mention_end = e.offset + e.length
                break
        if e.type == MessageEntityType.TEXT_MENTION and e.user and e.user.id == uid:
            mention_end = e.offset + e.length
            break

    if mention_end is not None:
        return text[mention_end:].strip()

    if text.startswith("/"):
        parts = text.split(None, 1)
        first = parts[0]
        if "@" in first:
            _, _, suffix = first.partition("@")
            if suffix.lower() == un and len(parts) > 1:
                return parts[1].strip()

    return _strip_self_mentions(text, me.username) or text


async def download_photo(photo: PhotoSize) -> bytes:
    """Скачивает фото из Telegram и возвращает байты."""
    file = await bot.get_file(photo.file_id)
    bio = BytesIO()
    await bot.download_file(file.file_path, bio)
    return bio.getvalue()


async def download_file(file_id: str) -> bytes:
    """Скачивает файл из Telegram по file_id и возвращает байты."""
    file = await bot.get_file(file_id)
    bio = BytesIO()
    await bot.download_file(file.file_path, bio)
    bio.seek(0)
    return bio.getvalue()


async def transcribe_voice(voice_bytes: bytes) -> str:
    """Преобразует голос в текст используя OpenAI Whisper API."""
    try:
        # Для Whisper используем OpenAI API напрямую (Groq не поддерживает audio)
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError(
                "Транскрипция голоса требует OPENAI_API_KEY.\n"
                "Получи ключ на https://platform.openai.com/api-keys"
            )
        
        # Создаём клиент OpenAI
        from openai import OpenAI
        openai_client = OpenAI(api_key=openai_key)
        
        # Используем asyncio.to_thread для синхронного API вызова
        def _transcribe():
            bio = BytesIO(voice_bytes)
            bio.name = "audio.ogg"
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=bio,
            )
            return transcript.text.strip()
        
        transcript = await asyncio.to_thread(_transcribe)
        return transcript
    except ValueError as e:
        # Переопубликуем ошибку о недостающем ключе
        raise
    except Exception as e:
        logger.exception("Whisper transcription error")
        raise


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Кодирует байты изображения в base64 строку."""
    return base64.b64encode(image_bytes).decode('utf-8')


async def ask_llm(user_text: str, user_id: int, reply_context: str | None = None) -> str:
    """Отправляет текстовый запрос в Groq API с учётом истории диалога и контекста ответа."""
    # Получаем историю пользователя
    history = chat_history.get(user_id, [])
    
    # Формируем сообщения: system + история + текущее
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    
    # Если есть контекст reply, добавляем его перед основным текстом
    if reply_context:
        full_text = f"Контекст (предыдущее сообщение):\n{reply_context}\n\nВопрос:\n{user_text}"
    else:
        full_text = user_text
    
    messages.append({"role": "user", "content": full_text})
    
    response = await llm_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    
    choice = response.choices[0]
    if not choice.message.content:
        return "Не удалось сформировать ответ."
    
    answer = choice.message.content.strip()
    
    # Сохраняем в историю (без контекста reply)
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    
    # Ограничиваем длину истории
    if len(history) > MAX_HISTORY_LENGTH * 2:
        history = history[-(MAX_HISTORY_LENGTH * 2):]
    
    chat_history[user_id] = history
    
    return answer


async def ask_llm_with_image(user_text: str, image_bytes: bytes, user_id: int, reply_context: str | None = None) -> str:
    """Отправляет запрос с изображением в Groq API с учётом истории диалога и контекста ответа."""
    base64_image = encode_image_to_base64(image_bytes)
    
    # Получаем историю пользователя
    history = chat_history.get(user_id, [])
    
    # Формируем сообщения
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    
    # Текущее сообщение с фото
    current_message = {
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            }
        ]
    }
    
    # Формируем текстовую часть с контекстом если есть
    if reply_context:
        text_content = f"Контекст (предыдущее сообщение):\n{reply_context}\n\nВопрос:\n{user_text}"
    else:
        text_content = user_text or "Что изображено на этом фото? Опиши подробно."
    
    current_message["content"].append({"type": "text", "text": text_content})
    
    messages.append(current_message)
    
    response = await llm_client.chat.completions.create(
        model=GROQ_VISION_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    
    choice = response.choices[0]
    if not choice.message.content:
        return "Не удалось сформировать ответ."
    
    answer = choice.message.content.strip()
    
    # Сохраняем в историю (без контекста reply)
    history_text = f"[Фото] {user_text}" if user_text else "[Фото]"
    history.append({"role": "user", "content": history_text})
    history.append({"role": "assistant", "content": answer})
    
    # Ограничиваем длину истории
    if len(history) > MAX_HISTORY_LENGTH * 2:
        history = history[-(MAX_HISTORY_LENGTH * 2):]
    
    chat_history[user_id] = history
    
    return answer


def _truncate_telegram(s: str, max_len: int = 4096) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"


def _split_message_into_chunks(text: str, max_len: int = 4096) -> List[str]:
    """Разделяет большой текст на несколько сообщений для Telegram."""
    if len(text) <= max_len:
        return [text]
    
    chunks = []
    remaining = text
    
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        
        # Берём кусок до max_len
        chunk = remaining[:max_len]
        
        # Стараемся разбить по последнему переводу строки или пробелу
        last_newline = chunk.rfind('\n')
        last_space = chunk.rfind(' ')
        
        split_pos = max(last_newline, last_space)
        if split_pos > max_len // 2:  # Если точка разбиения не слишком близко к краю
            chunk = remaining[:split_pos].rstrip()
            remaining = remaining[split_pos:].lstrip()
        else:
            chunk = remaining[:max_len - 1]
            remaining = remaining[max_len - 1:]
        
        chunks.append(chunk)
    
    return chunks


async def _get_reply_context(message: Message) -> str | None:
    """Получает контекст из сообщения, на которое отвечает пользователь."""
    if not message.reply_to_message:
        return None
    
    reply_msg = message.reply_to_message
    context_parts = []
    
    # Имя/упоминание автора сообщения
    if reply_msg.from_user:
        user_name = reply_msg.from_user.first_name or reply_msg.from_user.username or "Пользователь"
        context_parts.append(f"[Ответ {user_name}]")
    
    # Текст сообщения
    if reply_msg.text:
        context_parts.append(reply_msg.text)
    elif reply_msg.caption:
        context_parts.append(f"[Фото с подписью: {reply_msg.caption}]")
    elif reply_msg.photo:
        context_parts.append("[Сообщение содержит фото]")
    elif reply_msg.document:
        doc_name = reply_msg.document.file_name or "документ"
        context_parts.append(f"[Сообщение содержит документ: {doc_name}]")
    else:
        context_parts.append("[Сообщение без текста]")
    
    return "\n".join(context_parts) if context_parts else None


def get_private_keyboard() -> ReplyKeyboardMarkup:
    """Создаёт клавиатуру с кнопками, скрытую по умолчанию."""
    return ReplyKeyboardMarkup(
        keyboard=[
            [
                KeyboardButton(text="/start"),
                KeyboardButton(text="/help"),
                KeyboardButton(text="/clear"),
            ]
        ],
        resize_keyboard=True,
        one_time_keyboard=False,
        is_persistent=True,
    )


@dp.inline_query()
async def inline_query_handler(inline_query: InlineQuery) -> None:
    q = (inline_query.query or "").strip()
    if not q:
        await inline_query.answer(
            [
                InlineQueryResultArticle(
                    id="hint",
                    title="Введи запрос после @бота",
                    description="Например: кратко объясни async/await",
                    input_message_content=InputTextMessageContent(
                        message_text="Набери в поле ввода: @бот твой вопрос — затем выбери карточку с ответом."
                    ),
                )
            ],
            cache_time=1,
            is_personal=True,
        )
        return

    rid = hashlib.sha256(
        f"{inline_query.id}:{q}:{inline_query.from_user.id}".encode()
    ).hexdigest()[:32]

    try:
        # Для inline-запросов не используем историю
        answer = await ask_llm(q, inline_query.from_user.id)
    except RateLimitError:
        answer = "Лимит Groq (429). Попробуй позже: https://console.groq.com"
    except APIStatusError as e:
        answer = f"Ошибка API ({e.status_code}): {e.message}"
    except Exception as e:
        logger.exception("Inline Groq error")
        answer = f"Ошибка: {e}"

    answer = _truncate_telegram(answer)
    title = q[:64] if q else "Ответ"
    desc = answer[:200] if len(answer) > 200 else answer

    await inline_query.answer(
        [
            InlineQueryResultArticle(
                id=rid,
                title=title,
                description=desc,
                input_message_content=InputTextMessageContent(message_text=answer),
            )
        ],
        cache_time=0,
        is_personal=True,
    )


@dp.message(CommandStart(), addressed)
async def cmd_start(message: Message) -> None:
    # Очищаем историю при старте
    user_id = message.from_user.id
    if user_id in chat_history:
        del chat_history[user_id]
    
    if message.chat.type == ChatType.PRIVATE:
        await message.answer(
            "Привет! Сообщения обрабатывает Groq.\n"
            "• Текст → ответ ИИ\n"
            "• Голос → транскрипция + ответ (требует OpenAI API ключ)\n"
            "• Фото → анализ изображения\n"
            "• В личке — просто напиши текст или отправь фото/голос.\n"
            "• В группе — @username_бота и вопрос в том же сообщении, ответ на сообщение бота или /команда@бот.\n"
            "• Inline: в любом чате набери @бота в поле ввода и текст — выбери результат.\n"
            "• Память: бот помнит контекст разговора в личке.\n"
            "Команды: /help, /model, /clear\n",
            reply_markup=get_private_keyboard(),
        )
    else:
        await message.answer(
            "Привет! Сообщения обрабатывает Groq.\n"
            "• Текст → ответ ИИ\n"
            "• Голос → транскрипция + ответ\n"
            "• Фото → анализ изображения\n"
            "• В группе — @username_бота и вопрос в том же сообщении, ответ на сообщение бота или /команда@бот.\n"
            "• Inline: в любом чате набери @бота в поле ввода и текст — выбери результат.\n"
            "Команды: /help, /model\n"
        )


@dp.message(Command("help"), addressed)
async def cmd_help(message: Message) -> None:
    if message.chat.type == ChatType.PRIVATE:
        await message.answer(
            "📝 Текст → ответ Groq\n"
            "🎤 Голос → транскрипция (Whisper) + ответ\n"
            "🖼️ Фото → анализ изображения\n"
            "🎥 Видеозаписи → справка по обработке\n"
            "💬 Inline: @бот запрос\n\n"
            f"Модель текста: <code>{GROQ_MODEL}</code>\n"
            f"Модель vision: <code>{GROQ_VISION_MODEL}</code>\n\n"
            "Бот помнит историю диалога в личных сообщениях.\n"
            "/clear — очистить историю.",
            parse_mode="HTML",
            reply_markup=get_private_keyboard(),
        )
    else:
        await message.answer(
            "📝 Текст → ответ Groq\n"
            "🎤 Голос → транскрипция + ответ\n"
            "🖼️ Фото → анализ изображения\n"
            f"Модель текста: <code>{GROQ_MODEL}</code>\n"
            f"Модель vision: <code>{GROQ_VISION_MODEL}</code>",
            parse_mode="HTML",
        )


@dp.message(Command("model"), addressed)
async def cmd_model(message: Message) -> None:
    await message.answer(
        f"Текстовая модель: <code>{GROQ_MODEL}</code>\n"
        f"Vision модель: <code>{GROQ_VISION_MODEL}</code>",
        parse_mode="HTML"
    )


@dp.message(Command("clear"), addressed)
async def cmd_clear(message: Message) -> None:
    """Очищает историю диалога пользователя."""
    user_id = message.from_user.id
    if user_id in chat_history:
        del chat_history[user_id]
        await message.answer("🧹 История диалога очищена.")
    else:
        await message.answer("📭 История диалога уже пуста.")


@dp.message(F.photo, addressed, not_cmd)
async def handle_photo(message: Message, bot: Bot) -> None:
    """Обработчик фотографий."""
    if not message.photo:
        return
    
    user_prompt = await _prompt_for_llm(message, bot)
    
    # Получаем самое большое фото (последнее в списке)
    photo = message.photo[-1]
    
    wait = await message.answer("🔍 Анализирую фото…")
    
    try:
        # Скачиваем фото
        image_bytes = await download_photo(photo)
        
        # Получаем контекст из reply_to_message если есть
        reply_context = await _get_reply_context(message)
        
        # Отправляем на анализ с учётом истории (только в личке)
        user_id = message.from_user.id if message.chat.type == ChatType.PRIVATE else -message.chat.id
        answer = await ask_llm_with_image(user_prompt, image_bytes, user_id, reply_context)
        
        # Разделяем ответ на несколько сообщений если нужно
        chunks = _split_message_into_chunks(answer)
        
        if len(chunks) == 1:
            # Если один кусок, просто отредактируем сообщение
            await wait.edit_text(chunks[0])
        else:
            # Если несколько кусков, удалим ожидающее сообщение и отправим куски
            await wait.delete()
            for i, chunk in enumerate(chunks):
                await message.answer(chunk)
        
    except RateLimitError as e:
        logger.warning("Groq rate limit / quota: %s", e)
        await wait.edit_text(
            "Лимит Groq (429): слишком много запросов или квота.\n"
            "https://console.groq.com — проверь лимиты."
        )
    except APIStatusError as e:
        if "model_decommissioned" in str(e):
            logger.error("Vision model decommissioned: %s", GROQ_VISION_MODEL)
            await wait.edit_text(
                f"❌ Модель {GROQ_VISION_MODEL} выведена из эксплуатации.\n\n"
                f"Пожалуйста, обновите GROQ_VISION_MODEL в .env на актуальную:\n"
                f"llama-3.2-90b-vision-preview"
            )
        else:
            logger.exception("Groq API error")
            await wait.edit_text(f"Ошибка API ({e.status_code}): {e.message}")
    except Exception as e:
        logger.exception("Photo processing error")
        await wait.edit_text(f"Ошибка при обработке фото: {e}")


@dp.message(F.voice, addressed, not_cmd)
async def handle_voice(message: Message, bot: Bot) -> None:
    """Обработчик голосовых сообщений — преобразует в текст и отправляет в LLM."""
    if not message.voice:
        return
    
    user_prompt = await _prompt_for_llm(message, bot)
    
    wait = await message.answer("🎤 Преобразую голос в текст…")
    
    try:
        # Скачиваем голосовой файл
        voice_bytes = await download_file(message.voice.file_id)
        
        # Преобразуем в текст через Whisper
        transcribed_text = await transcribe_voice(voice_bytes)
        
        if not transcribed_text:
            await wait.edit_text("Не удалось разобрать аудио. Попробуй говорить четче.")
            return
        
        # Если есть дополнительный текст вместе с голосом, добавляем его
        if user_prompt:
            full_prompt = f"{user_prompt}\n\n[Голосовое сообщение]: {transcribed_text}"
        else:
            full_prompt = transcribed_text
        
        # Получаем контекст из reply_to_message если есть
        reply_context = await _get_reply_context(message)
        
        # Отправляем в LLM с учётом истории
        user_id = message.from_user.id if message.chat.type == ChatType.PRIVATE else -message.chat.id
        answer = await ask_llm(full_prompt, user_id, reply_context)
        
        # Разделяем ответ на несколько сообщений если нужно
        chunks = _split_message_into_chunks(answer)
        
        if len(chunks) == 1:
            await wait.edit_text(f"🗣️ *Расшифровка:*\n`{transcribed_text}`\n\n*Ответ:*\n{chunks[0]}", parse_mode="Markdown")
        else:
            await wait.delete()
            await message.answer(f"🗣️ *Расшифровка:*\n`{transcribed_text}`", parse_mode="Markdown")
            for chunk in chunks:
                await message.answer(chunk)
    
    except ValueError as e:
        # Ошибка о недостающем API ключе
        logger.warning("Configuration error: %s", e)
        await wait.edit_text(
            f"⚠️ {str(e)}\n\n"
            "Добавь OPENAI_API_KEY в .env файл для использования транскрипции голоса."
        )
    except RateLimitError as e:
        logger.warning("API rate limit: %s", e)
        await wait.edit_text(
            "Лимит API (429): слишком много запросов.\n"
            "Попробуй позже или проверь лимиты на платформе."
        )
    except APIStatusError as e:
        logger.exception("API error during voice processing")
        await wait.edit_text(f"Ошибка API ({e.status_code}): {e.message}")
    except Exception as e:
        logger.exception("Voice processing error")
        await wait.edit_text(f"Ошибка при обработке голоса: {e}")


@dp.message(F.video_note, addressed, not_cmd)
async def handle_video_note(message: Message, bot: Bot) -> None:
    """Обработчик видеозаписей (кружков) — преобразует в текст где возможно."""
    if not message.video_note:
        return
    
    wait = await message.answer("🎥 Обработка видеозаписи…")
    
    try:
        # Скачиваем видеозаметку (обычно это маленькое видео)
        video_bytes = await download_file(message.video_note.file_id)
        
        # К сожалению, Groq vision API работает с изображениями, а не видео
        # Попытаемся обработать как медиа-файл
        await wait.edit_text(
            "⚠️ Видеозаписи обычно требуют специальной обработки.\n\n"
            "Лучше всего:\n"
            "1. Отправь скриншот видео (фотографию)\n"
            "2. Или напиши текст с описанием\n\n"
            "Если в видео есть речь, отправь как голосовое сообщение — я переведу в текст."
        )
    
    except Exception as e:
        logger.exception("Video note processing error")
        await wait.edit_text(f"Ошибка при обработке видеозаписи: {e}")


@dp.message(F.text, addressed, not_cmd)
async def handle_text(message: Message, bot: Bot) -> None:
    if not message.text:
        return

    user_prompt = await _prompt_for_llm(message, bot)

    if (
        message.chat.type in (ChatType.GROUP, ChatType.SUPERGROUP)
        and not user_prompt
    ):
        me = await bot.get_me()
        un = f"@{me.username}" if me.username else "бота"
        await message.reply(f"Напиши текст после {un}, например: {un} что такое pytest?")
        return

    wait = await message.answer("…")
    try:
        # Получаем контекст из reply_to_message если есть
        reply_context = await _get_reply_context(message)
        
        # Используем историю только в личных сообщениях
        user_id = message.from_user.id if message.chat.type == ChatType.PRIVATE else -message.chat.id
        answer = await ask_llm(user_prompt, user_id, reply_context)
        
        # Разделяем ответ на несколько сообщений если нужно
        chunks = _split_message_into_chunks(answer)
        
        if len(chunks) == 1:
            # Если один кусок, просто отредактируем сообщение
            await wait.edit_text(chunks[0])
        else:
            # Если несколько кусков, удалим ожидающее сообщение и отправим куски
            await wait.delete()
            for i, chunk in enumerate(chunks):
                if i == 0:
                    # Первый кусок отправляем как ответ
                    await message.answer(chunk)
                else:
                    # Остальные куски просто отправляем
                    await message.answer(chunk)
    
    except RateLimitError as e:
        logger.warning("Groq rate limit / quota: %s", e)
        await wait.edit_text(
            "Лимит Groq (429): слишком много запросов или квота.\n"
            "https://console.groq.com — проверь лимиты."
        )
    except APIStatusError as e:
        logger.exception("Groq API error")
        await wait.edit_text(f"Ошибка API ({e.status_code}): {e.message}")
    except Exception as e:
        logger.exception("LLM error")
        await wait.edit_text(f"Ошибка: {e}")


async def on_startup(bot: Bot) -> None:
    # Получаем URL от Render
    render_url = os.getenv("RENDER_EXTERNAL_URL")
    if not render_url:
        logger.error("RENDER_EXTERNAL_URL not set!")
        return
    
    webhook_url = f"{render_url}/webhook"
    await bot.set_webhook(webhook_url)
    logger.info(f"Webhook set to {webhook_url}")

async def on_shutdown(bot: Bot) -> None:
    await bot.delete_webhook()
    logger.info("Webhook deleted")

async def health_check(request):
    return web.Response(text="OK")


async def main() -> None:
    try:
        dp.startup.register(on_startup)
        dp.shutdown.register(on_shutdown)
        app = web.Application()
        app.router.add_get("/", health_check)
        webhook_requests_handler = SimpleRequestHandler(
            dispatcher=dp,
            bot=bot,
        )
        webhook_requests_handler.register(app, path="/webhook")
        setup_application(app, dp, bot=bot)
        port = int(os.getenv("PORT", 10000))
        await web._run_app(app, host="0.0.0.0", port=port)
    except TelegramNotFound:
        logger.error("Telegram вернул Not Found — обычно это неверный или отозванный TELEGRAM_BOT_TOKEN.")
        print(
            "Ошибка: токен бота не принят Telegram (Not Found).\n"
            "Проверь .env: без лишних пробелов/кавычек, полная строка от @BotFather (/token).\n"
            "Если бот пересоздавался — выпусти новый токен.",
            file=sys.stderr,
        )
        raise
    except TelegramNetworkError as e:
        logger.error("Нет соединения с Telegram API: %s", e)
        print(
            "Не удаётся подключиться к api.telegram.org (HTTPS, порт 443).\n"
            "Частые причины: нет интернета, файрвол/антивирус, блокировка провайдером или региона.\n"
            "Попробуй: другую сеть, VPN, или укажи прокси в .env:\n"
            "  TELEGRAM_PROXY=socks5://127.0.0.1:1080\n"
            "(нужен пакет: pip install aiohttp-socks)",
            file=sys.stderr,
        )
        raise
    except Exception as e:
        logger.error(f"Critical error: {e}")
        raise
        await dp.start_polling(bot, allowed_updates=["message", "inline_query"])

if __name__ == "__main__":
    asyncio.run(main())
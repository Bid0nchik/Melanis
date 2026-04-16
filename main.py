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
MAX_HISTORY_LENGTH = 100

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


def encode_image_to_base64(image_bytes: bytes) -> str:
    """Кодирует байты изображения в base64 строку."""
    return base64.b64encode(image_bytes).decode('utf-8')


async def ask_llm(user_text: str, user_id: int) -> str:
    """Отправляет текстовый запрос в Groq API с учётом истории диалога."""
    # Получаем историю пользователя
    history = chat_history.get(user_id, [])
    
    # Формируем сообщения: system + история + текущее
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    messages.append({"role": "user", "content": user_text})
    
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
    
    # Сохраняем в историю
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": answer})
    
    # Ограничиваем длину истории
    if len(history) > MAX_HISTORY_LENGTH * 2:
        history = history[-(MAX_HISTORY_LENGTH * 2):]
    
    chat_history[user_id] = history
    
    return answer


async def ask_llm_with_image(user_text: str, image_bytes: bytes, user_id: int) -> str:
    """Отправляет запрос с изображением в Groq API с учётом истории диалога."""
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
    
    if user_text:
        current_message["content"].append({"type": "text", "text": user_text})
    else:
        current_message["content"].append({"type": "text", "text": "Что изображено на этом фото? Опиши подробно."})
    
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
    
    # Сохраняем в историю
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
            "• В личке — просто напиши текст или отправь фото.\n"
            "• В группе — @username_бота и вопрос в том же сообщении, ответ на сообщение бота или /команда@бот.\n"
            "• Inline: в любом чате набери @бота в поле ввода и текст — выбери результат.\n"
            "• Фото: отправь фото с описанием или без — бот проанализирует изображение.\n"
            "• Память: бот помнит контекст разговора в личке.\n"
            "Команды: /help, /model, /clear\n",
            reply_markup=get_private_keyboard(),
        )
    else:
        await message.answer(
            "Привет! Сообщения обрабатывает Groq.\n"
            "• В личке — просто напиши текст или отправь фото.\n"
            "• В группе — @username_бота и вопрос в том же сообщении, ответ на сообщение бота или /команда@бот.\n"
            "• Inline: в любом чате набери @бота в поле ввода и текст — выбери результат.\n"
            "• Фото: отправь фото с описанием или без — бот проанализирует изображение.\n"
            "Команды: /help, /model\n"
        )


@dp.message(Command("help"), addressed)
async def cmd_help(message: Message) -> None:
    if message.chat.type == ChatType.PRIVATE:
        await message.answer(
            "Текст → ответ Groq. Inline: @бот запрос.\n"
            "Фото → анализ изображения.\n"
            f"Модель текста: <code>{GROQ_MODEL}</code>\n"
            f"Модель vision: <code>{GROQ_VISION_MODEL}</code>\n\n"
            "Бот помнит историю диалога в личных сообщениях.\n"
            "/clear — очистить историю.",
            parse_mode="HTML",
            reply_markup=get_private_keyboard(),
        )
    else:
        await message.answer(
            "Текст → ответ Groq. Inline: @бот запрос.\n"
            "Фото → анализ изображения.\n"
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
        
        # Отправляем на анализ с учётом истории (только в личке)
        user_id = message.from_user.id if message.chat.type == ChatType.PRIVATE else -message.chat.id
        answer = await ask_llm_with_image(user_prompt, image_bytes, user_id)
        
        await wait.edit_text(_truncate_telegram(answer))
        
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
        # Используем историю только в личных сообщениях
        user_id = message.from_user.id if message.chat.type == ChatType.PRIVATE else -message.chat.id
        answer = await ask_llm(user_prompt, user_id)
        await wait.edit_text(_truncate_telegram(answer))
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


async def main() -> None:
    try:
        await dp.start_polling(bot, allowed_updates=["message", "inline_query"])
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


if __name__ == "__main__":
    asyncio.run(main())
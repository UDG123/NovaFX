"""Telegram /status command handler using getUpdates polling."""

import logging

import httpx

from app.config import settings
from app.services.bot_state import BotState

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"

_last_update_id = 0


def _format_status_message() -> str:
    state = BotState.get()

    # Engine status
    if settings.SIGNAL_ENGINE_ENABLED:
        engine_status = "\u2705 Enabled"
    else:
        engine_status = "\u274c Disabled"

    # Active strategy
    strategy = state.active_strategy or "None yet"

    # Last signal
    if state.last_signal and state.last_signal_time:
        sig = state.last_signal
        direction = "\U0001f7e2 BUY" if sig.action == "BUY" else "\U0001f534 SELL"
        last_signal_text = (
            f"{direction} <code>{sig.symbol}</code> @ {sig.price}\n"
            f"    <i>{state.last_signal_time.strftime('%Y-%m-%d %H:%M:%S UTC')}</i>"
        )
    else:
        last_signal_text = "No signals generated yet"

    # Next scheduled run
    next_run = state.get_next_run_time()
    if next_run:
        next_run_text = next_run.strftime("%Y-%m-%d %H:%M:%S UTC")
    else:
        next_run_text = "Not scheduled"

    return (
        "<b>\u2501\u2501\u2501 NovaFX Status \u2501\u2501\u2501</b>\n\n"
        f"<b>Signal Engine:</b> {engine_status}\n"
        f"<b>Interval:</b> every {settings.SIGNAL_ENGINE_INTERVAL_MINUTES}m\n"
        f"<b>Next Run:</b> {next_run_text}\n\n"
        f"<b>Active Strategy:</b> {strategy}\n\n"
        f"<b>Last Signal:</b>\n"
        f"  {last_signal_text}"
    )


async def poll_telegram_commands() -> None:
    """Poll Telegram getUpdates for /status commands and reply."""
    global _last_update_id

    if not settings.TELEGRAM_BOT_TOKEN:
        return

    base = TELEGRAM_API.format(token=settings.TELEGRAM_BOT_TOKEN)
    params = {"offset": _last_update_id + 1, "timeout": 0, "allowed_updates": '["message"]'}

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{base}/getUpdates", params=params)
            resp.raise_for_status()
            data = resp.json()

            if not data.get("ok") or not data.get("result"):
                return

            for update in data["result"]:
                update_id = update["update_id"]
                if update_id > _last_update_id:
                    _last_update_id = update_id

                message = update.get("message", {})
                text = message.get("text", "")
                chat_id = message.get("chat", {}).get("id")

                if not text.strip().startswith("/status") or not chat_id:
                    continue

                if settings.TELEGRAM_CHAT_ID and str(chat_id) != settings.TELEGRAM_CHAT_ID:
                    logger.warning("Unauthorized /status from chat %s — ignored", chat_id)
                    continue

                reply = _format_status_message()
                await client.post(
                    f"{base}/sendMessage",
                    json={
                        "chat_id": chat_id,
                        "text": reply,
                        "parse_mode": "HTML",
                    },
                )
                logger.info("Replied to /status from chat %s", chat_id)

    except httpx.HTTPError:
        logger.warning("Telegram poll failed")

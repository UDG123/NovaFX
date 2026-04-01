import logging

import httpx

from app.config import settings
from app.models.signals import ProcessedSignal

logger = logging.getLogger(__name__)

TELEGRAM_SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"


def format_signal_message(signal: ProcessedSignal) -> str:
    direction = "\u2705 LONG (BUY)" if signal.action == "BUY" else "\ud83d\udd34 SHORT (SELL)"
    return (
        f"<b>\u2501\u2501\u2501 NovaFX Signal \u2501\u2501\u2501</b>\n\n"
        f"<b>{direction}</b>\n"
        f"<b>Symbol:</b> <code>{signal.symbol}</code>\n"
        f"<b>Timeframe:</b> {signal.timeframe}\n\n"
        f"\u25b8 <b>Entry:</b>  <code>{signal.entry_price}</code>\n"
        f"\u25b8 <b>Stop Loss:</b>  <code>{signal.stop_loss}</code>\n"
        f"\u25b8 <b>Take Profit:</b>  <code>{signal.take_profit}</code>\n\n"
        f"\ud83d\udcca <b>R:R</b> {signal.risk_reward}  |  "
        f"<b>Size</b> {signal.position_size}  |  "
        f"<b>Risk</b> ${signal.risk_amount}\n\n"
        f"<i>Source: {signal.source}</i>"
        + (f"  \u2022  <i>{signal.indicator}</i>" if signal.indicator else "")
        + f"\n<i>{signal.timestamp.strftime('%Y-%m-%d %H:%M UTC')}</i>"
    )


async def send_signal(signal: ProcessedSignal) -> bool:
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured - skipping alert")
        return False

    url = TELEGRAM_SEND_URL.format(token=settings.TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": format_signal_message(signal),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            logger.info("Telegram alert sent: %s %s", signal.action, signal.symbol)
            return True
    except httpx.HTTPError:
        logger.error("Failed to send Telegram alert for %s %s", signal.action, signal.symbol)
        return False

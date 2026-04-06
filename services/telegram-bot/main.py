"""
NovaFX Telegram Bot Service.

Subscribes to Redis pub/sub channel "telegram:signals" and forwards
confluence-validated signals to Telegram channels. Also handles
/status, /health, and /performance bot commands.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone

import httpx
import redis.asyncio as redis

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.models import AssetClass, ConfluenceResult, SignalAction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("novafx.telegram")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
DISPATCHER_URL = os.getenv("DISPATCHER_URL", "http://localhost:8000")

SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"
UPDATES_URL = "https://api.telegram.org/bot{token}/getUpdates"

# Desk channel mapping (env vars)
DESK_CHANNELS = {
    AssetClass.CRYPTO: os.getenv("TG_DESK_CRYPTO", ""),
    AssetClass.FOREX: os.getenv("TG_DESK_FOREX", ""),
    AssetClass.STOCKS: os.getenv("TG_DESK_STOCKS", ""),
    AssetClass.COMMODITIES: os.getenv("TG_DESK_COMMODITIES", ""),
    AssetClass.FUTURES: os.getenv("TG_DESK_FUTURES", ""),
}

# Stats tracking
_stats = {
    "signals_received": 0,
    "signals_sent": 0,
    "errors": 0,
    "started_at": None,
}


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------

CONFIDENCE_EMOJI = {
    "HIGH": "\u2705",
    "MEDIUM": "\u26a0\ufe0f",
    "LOW": "\U0001f534",
}

ASSET_EMOJI = {
    AssetClass.CRYPTO: "\u20bf",
    AssetClass.FOREX: "\U0001f30d",
    AssetClass.STOCKS: "\U0001f4c8",
    AssetClass.COMMODITIES: "\U0001f947",
    AssetClass.FUTURES: "\U0001f4ca",
}


def format_confluence_message(result: ConfluenceResult) -> str:
    """Format a ConfluenceResult as a Telegram HTML message."""
    action = result.consensus_action
    direction = "\U0001f4c8 BUY" if action == SignalAction.BUY else "\U0001f4c9 SELL"
    pair_emoji = "\U0001f537" if action == SignalAction.BUY else "\U0001f536"
    asset_emoji = ASSET_EMOJI.get(result.asset_class, "\U0001f4e1")

    conf_pct = round(result.weighted_confidence * 100)
    if conf_pct >= 80:
        conf_level = "HIGH"
    elif conf_pct >= 60:
        conf_level = "MEDIUM"
    else:
        conf_level = "LOW"
    conf_emoji = CONFIDENCE_EMOJI.get(conf_level, "\u2753")

    sources_count = len(result.contributing_signals)
    ts = result.timestamp.strftime("%d %b %Y  %H:%M UTC")

    return (
        f"\u26a1 <b>NOVAFX CONFLUENCE SIGNAL</b>\n\n"
        f"{pair_emoji} <b>{result.symbol}</b>  {direction}\n"
        f"{asset_emoji} {result.asset_class.value.capitalize()}\n\n"
        f"{conf_emoji} <b>Confidence:</b> {conf_pct}% ({conf_level})\n"
        f"\U0001f4ca <b>Sources:</b> {sources_count} independent signals agree\n\n"
        f"\U0001f4c5 <i>{ts}</i>\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u26a0\ufe0f <i>Risk max 1-2% per trade. Not financial advice.</i>"
    )


# ---------------------------------------------------------------------------
# Telegram API
# ---------------------------------------------------------------------------


async def send_message(chat_id: str, text: str) -> bool:
    """Send an HTML message to a Telegram chat."""
    if not TELEGRAM_BOT_TOKEN or not chat_id:
        return False

    url = SEND_URL.format(token=TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return True
    except Exception:
        logger.error("Failed to send message to %s", chat_id)
        _stats["errors"] += 1
        return False


async def send_to_desk(result: ConfluenceResult, text: str) -> bool:
    """Send to the appropriate desk channel based on asset class."""
    desk_id = DESK_CHANNELS.get(result.asset_class, "")
    sent = False

    if desk_id:
        sent = await send_message(desk_id, text)

    # Always mirror to main channel
    if TELEGRAM_CHAT_ID:
        await send_message(TELEGRAM_CHAT_ID, text)

    return sent


# ---------------------------------------------------------------------------
# Bot commands
# ---------------------------------------------------------------------------


async def handle_command(command: str, chat_id: str) -> None:
    """Handle incoming bot commands."""
    cmd = command.strip().lower().split("@")[0]

    if cmd == "/status":
        uptime = ""
        if _stats["started_at"]:
            delta = datetime.now(timezone.utc) - _stats["started_at"]
            hours, remainder = divmod(int(delta.total_seconds()), 3600)
            minutes = remainder // 60
            uptime = f"\u23f1 <b>Uptime:</b> {hours}h {minutes}m\n"

        msg = (
            f"\U0001f4e1 <b>NovaFX Status</b>\n\n"
            f"{uptime}"
            f"\U0001f4e8 <b>Signals received:</b> {_stats['signals_received']}\n"
            f"\u2705 <b>Signals sent:</b> {_stats['signals_sent']}\n"
            f"\u274c <b>Errors:</b> {_stats['errors']}\n"
        )
        await send_message(chat_id, msg)

    elif cmd == "/health":
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{DISPATCHER_URL}/health")
                data = resp.json()
                status = data.get("status", "unknown")
                msg = f"\U0001f3e5 <b>Dispatcher:</b> {status}\n"
        except Exception:
            msg = "\U0001f3e5 <b>Dispatcher:</b> \u274c unreachable\n"

        await send_message(chat_id, msg)

    elif cmd == "/help":
        msg = (
            "\U0001f4d6 <b>NovaFX Bot Commands</b>\n\n"
            "/status \u2014 Bot stats and uptime\n"
            "/health \u2014 Check dispatcher health\n"
            "/help \u2014 Show this message\n"
        )
        await send_message(chat_id, msg)


async def poll_commands() -> None:
    """Poll for Telegram bot commands."""
    if not TELEGRAM_BOT_TOKEN:
        return

    url = UPDATES_URL.format(token=TELEGRAM_BOT_TOKEN)
    offset = 0

    while True:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    url,
                    params={"offset": offset, "timeout": 25},
                )
                data = resp.json()

            for update in data.get("result", []):
                offset = update["update_id"] + 1
                message = update.get("message", {})
                text = message.get("text", "")
                chat_id = str(message.get("chat", {}).get("id", ""))

                if text.startswith("/"):
                    await handle_command(text, chat_id)

        except asyncio.CancelledError:
            raise
        except Exception:
            await asyncio.sleep(5)


# ---------------------------------------------------------------------------
# Redis pub/sub listener
# ---------------------------------------------------------------------------


async def listen_signals() -> None:
    """Subscribe to Redis pub/sub and forward signals to Telegram."""
    redis_client = redis.from_url(
        REDIS_URL, decode_responses=True, socket_connect_timeout=5
    )

    try:
        await redis_client.ping()
        logger.info("Redis connected for pub/sub")
    except Exception:
        logger.error("Redis connection failed — telegram bot cannot receive signals")
        return

    pubsub = redis_client.pubsub()
    await pubsub.subscribe("telegram:signals")
    logger.info("Subscribed to telegram:signals channel")

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            _stats["signals_received"] += 1

            try:
                result = ConfluenceResult.model_validate_json(message["data"])
                text = format_confluence_message(result)
                sent = await send_to_desk(result, text)

                if sent:
                    _stats["signals_sent"] += 1
                    logger.info(
                        "Signal forwarded: %s %s (confidence=%.0f%%)",
                        result.consensus_action.value,
                        result.symbol,
                        result.weighted_confidence * 100,
                    )

            except Exception:
                logger.exception("Failed to process signal message")
                _stats["errors"] += 1

    except asyncio.CancelledError:
        raise
    finally:
        await pubsub.unsubscribe("telegram:signals")
        await redis_client.aclose()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Start all bot tasks."""
    _stats["started_at"] = datetime.now(timezone.utc)

    # Send startup message
    if TELEGRAM_CHAT_ID:
        await send_message(
            TELEGRAM_CHAT_ID,
            "\U0001f7e2 <b>NovaFX Telegram Bot started</b>\n"
            f"\U0001f4c5 {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')}",
        )

    logger.info("NovaFX Telegram Bot started")

    # Run signal listener and command poller concurrently
    await asyncio.gather(
        listen_signals(),
        poll_commands(),
        return_exceptions=True,
    )


if __name__ == "__main__":
    asyncio.run(main())

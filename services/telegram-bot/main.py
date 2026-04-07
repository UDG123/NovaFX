"""
NovaFX Telegram Bot Service.

Subscribes to Redis pub/sub channels:
  - "telegram:signals" — confluence-validated trading signals
  - "telegram:health"  — system health alerts (source failures, fallbacks)

Forwards formatted HTML messages to Telegram desk channels.
Runs in dry mode (log only) when TELEGRAM_BOT_TOKEN is empty.
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
_raw_chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_CHAT_ID = int(_raw_chat_id) if _raw_chat_id.lstrip("-").isdigit() else _raw_chat_id
DISPATCHER_URL = os.getenv("DISPATCHER_URL", "http://localhost:8000")

DRY_MODE = not TELEGRAM_BOT_TOKEN

SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"
UPDATES_URL = "https://api.telegram.org/bot{token}/getUpdates"

DESK_CHANNELS = {
    AssetClass.CRYPTO: os.getenv("TG_DESK_CRYPTO", ""),
    AssetClass.FOREX: os.getenv("TG_DESK_FOREX", ""),
    AssetClass.STOCKS: os.getenv("TG_DESK_STOCKS", ""),
    AssetClass.COMMODITIES: os.getenv("TG_DESK_COMMODITIES", ""),
    AssetClass.FUTURES: os.getenv("TG_DESK_FUTURES", ""),
}

_stats = {
    "signals_received": 0,
    "signals_sent": 0,
    "health_alerts": 0,
    "errors": 0,
    "started_at": None,
}


# ---------------------------------------------------------------------------
# Emoji maps
# ---------------------------------------------------------------------------

ACTION_EMOJI = {
    SignalAction.BUY: "\U0001f7e2",     # green circle
    SignalAction.SELL: "\U0001f534",     # red circle
    SignalAction.CLOSE: "\u26aa",        # white circle
    SignalAction.HOLD: "\U0001f7e1",     # yellow circle
}

ASSET_EMOJI = {
    AssetClass.CRYPTO: "\u20bf",
    AssetClass.FOREX: "\U0001f4b1",
    AssetClass.STOCKS: "\U0001f4c8",
    AssetClass.COMMODITIES: "\U0001f947",
    AssetClass.FUTURES: "\U0001f4ca",
}


# ---------------------------------------------------------------------------
# Message formatting
# ---------------------------------------------------------------------------


def format_confluence_message(result: ConfluenceResult) -> str:
    """Format a ConfluenceResult as Telegram HTML."""
    action = result.consensus_action
    action_emoji = ACTION_EMOJI.get(action, "\u2753")
    direction = f"{action_emoji} {action.value.upper()}"
    asset_emoji = ASSET_EMOJI.get(result.asset_class, "\U0001f4e1")

    conf_pct = round(result.weighted_confidence * 100)
    if conf_pct >= 80:
        conf_bar = "\u2705 HIGH"
    elif conf_pct >= 60:
        conf_bar = "\u26a0\ufe0f MEDIUM"
    else:
        conf_bar = "\U0001f534 LOW"

    sources_count = len(result.contributing_signals)
    ts = result.timestamp.strftime("%d %b %Y  %H:%M UTC")

    return (
        f"\u26a1 <b>NOVAFX CONFLUENCE SIGNAL</b>\n\n"
        f"{direction}  <b>{result.symbol}</b>\n"
        f"{asset_emoji} {result.asset_class.value.capitalize()}\n\n"
        f"\U0001f3af <b>Confidence:</b> {conf_pct}% {conf_bar}\n"
        f"\U0001f4ca <b>Sources:</b> {sources_count} independent signals agree\n\n"
        f"\U0001f4c5 <i>{ts}</i>\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u26a0\ufe0f <i>Risk max 1-2% per trade. Not financial advice.</i>"
    )


def format_health_alert(data: dict) -> str:
    """Format a health alert from telegram:health channel."""
    severity = data.get("severity", "warning")
    source = data.get("source", "unknown")
    message = data.get("message", "")
    fallback = data.get("fallback", "")
    ts = data.get("timestamp", datetime.now(timezone.utc).isoformat())

    if severity == "critical":
        header = "\U0001f534 <b>CRITICAL ALERT</b>"
    else:
        header = "\u26a0\ufe0f <b>WARNING</b>"

    lines = [
        f"{header}\n",
        f"\U0001f50c <b>Source:</b> {source}",
        f"\U0001f4ac <b>Message:</b> {message}",
    ]

    if fallback:
        lines.append(f"\U0001f504 <b>Fallback:</b> {fallback}")

    lines.append(f"\n\U0001f4c5 <i>{ts}</i>")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Telegram API
# ---------------------------------------------------------------------------


async def send_message(chat_id: str, text: str) -> bool:
    """Send an HTML message to Telegram. Logs in dry mode."""
    if DRY_MODE:
        logger.info("[DRY MODE] Would send to %s:\n%s", chat_id, text)
        return True

    if not chat_id:
        return False

    url = SEND_URL.format(token=TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code != 200:
                logger.error(
                    "Telegram API error for chat %s: %s — %s",
                    chat_id, resp.status_code, resp.text,
                )
                _stats["errors"] += 1
                return False
            return True
    except Exception as exc:
        logger.error("Failed to send message to chat %s: %s", chat_id, exc)
        _stats["errors"] += 1
        return False


async def send_to_desk(result: ConfluenceResult, text: str) -> bool:
    """Send to desk channel by asset class + mirror to main channel."""
    desk_id = DESK_CHANNELS.get(result.asset_class, "")
    sent = False

    if desk_id:
        sent = await send_message(desk_id, text)

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

        mode = "\U0001f7e1 DRY MODE" if DRY_MODE else "\U0001f7e2 LIVE"

        msg = (
            f"\U0001f4e1 <b>NovaFX Status</b>  |  {mode}\n\n"
            f"{uptime}"
            f"\U0001f4e8 <b>Signals received:</b> {_stats['signals_received']}\n"
            f"\u2705 <b>Signals sent:</b> {_stats['signals_sent']}\n"
            f"\u26a0\ufe0f <b>Health alerts:</b> {_stats['health_alerts']}\n"
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
            "/status \u2014 Bot stats, uptime, mode\n"
            "/health \u2014 Check dispatcher health\n"
            "/help \u2014 Show this message\n"
        )
        await send_message(chat_id, msg)


async def poll_commands() -> None:
    """Poll for Telegram bot commands."""
    if DRY_MODE:
        logger.info("Dry mode — command polling disabled")
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
# Redis pub/sub listeners
# ---------------------------------------------------------------------------


async def listen_signals(redis_client: redis.Redis) -> None:
    """Subscribe to telegram:signals and forward to Telegram."""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("telegram:signals")
    logger.info("Subscribed to telegram:signals")

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


async def listen_health(redis_client: redis.Redis) -> None:
    """Subscribe to telegram:health and forward alerts."""
    pubsub = redis_client.pubsub()
    await pubsub.subscribe("telegram:health")
    logger.info("Subscribed to telegram:health")

    try:
        async for message in pubsub.listen():
            if message["type"] != "message":
                continue

            _stats["health_alerts"] += 1

            try:
                data = json.loads(message["data"])
                text = format_health_alert(data)

                if TELEGRAM_CHAT_ID:
                    await send_message(TELEGRAM_CHAT_ID, text)

                system_id = os.getenv("TG_SYSTEM", "")
                if system_id:
                    await send_message(system_id, text)

                logger.info(
                    "Health alert: [%s] %s — %s",
                    data.get("severity", "?"),
                    data.get("source", "?"),
                    data.get("message", ""),
                )
            except Exception:
                logger.exception("Failed to process health alert")
                _stats["errors"] += 1

    except asyncio.CancelledError:
        raise
    finally:
        await pubsub.unsubscribe("telegram:health")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Start all bot tasks."""
    _stats["started_at"] = datetime.now(timezone.utc)

    if DRY_MODE:
        logger.warning("TELEGRAM_BOT_TOKEN not set — running in DRY MODE (log only)")
    else:
        logger.info("NovaFX Telegram Bot started in LIVE mode")

    # Connect to Redis
    redis_client = redis.from_url(
        REDIS_URL, decode_responses=True, socket_connect_timeout=5
    )
    try:
        await redis_client.ping()
        logger.info("Redis connected")
    except Exception:
        logger.error("Redis connection failed — bot cannot receive signals")
        return

    # Send startup message with retry
    if TELEGRAM_CHAT_ID and not DRY_MODE:
        startup_text = (
            f"\U0001f7e2 <b>NovaFX Telegram Bot started</b>\n"
            f"[LIVE]\n"
            f"\U0001f4c5 {datetime.now(timezone.utc).strftime('%d %b %Y %H:%M UTC')}"
        )
        url = SEND_URL.format(token=TELEGRAM_BOT_TOKEN)
        for attempt in range(1, 4):
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    resp = await client.post(
                        url,
                        json={"chat_id": TELEGRAM_CHAT_ID, "text": startup_text, "parse_mode": "HTML"},
                    )
                    if resp.status_code == 200:
                        logger.info("Startup message sent successfully")
                        break
                    else:
                        logger.warning(
                            "Startup message attempt %d/3 failed: %s — %s",
                            attempt, resp.status_code, resp.text,
                        )
            except Exception as exc:
                logger.warning("Startup message attempt %d/3 error: %s", attempt, exc)
            if attempt < 3:
                await asyncio.sleep(5)

    try:
        await asyncio.gather(
            listen_signals(redis_client),
            listen_health(redis_client),
            poll_commands(),
            return_exceptions=True,
        )
    finally:
        await redis_client.aclose()
        logger.info("Telegram bot stopped")


if __name__ == "__main__":
    asyncio.run(main())

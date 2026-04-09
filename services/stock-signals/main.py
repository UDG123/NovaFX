"""
NovaFX Stock Signal Generator.

Scans 10 US stocks/ETFs on 1H candles with four-tier data source
failover: Alpaca -> TwelveData -> Finnhub -> Alpha Vantage.
Only active during US market hours (9:30 AM - 4:00 PM ET).
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone, timedelta

import httpx
import redis as redis_client

# ---------------------------------------------------------------------------
# Redis signal deduplication
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL")
_redis = redis_client.from_url(REDIS_URL) if REDIS_URL else None
SIGNAL_COOLDOWN_SECONDS = 3600  # 1 hour cooldown per pair+direction


async def write_signal_to_stream(symbol: str, direction: str, price: float, confidence: float) -> None:
    """Write signal to Redis stream for paper trader to pick up."""
    if not _redis:
        return
    try:
        import time
        _redis.xadd("novafx:signals:stocks", {
            "symbol": symbol,
            "direction": direction,
            "entry": str(price),
            "confidence": str(confidence),
            "timestamp": str(time.time()),
        }, maxlen=100)
    except Exception as e:
        logging.getLogger("novafx.stocks").warning(f"Failed to write to Redis stream: {e}")


def is_duplicate_signal(symbol: str, direction: str) -> bool:
    """Check if signal was sent recently; if not, mark it as sent."""
    if not _redis:
        return False
    key = f"novafx:signal_sent:{symbol}:{direction}"
    if _redis.exists(key):
        return True
    _redis.setex(key, SIGNAL_COOLDOWN_SECONDS, "1")
    return False

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.models import Signal
from shared.resilience import AllSourcesFailedError, DataSourceManager
from signals import analyze_candles
from sources import (
    AlpacaStockSource,
    AlphaVantageStockSource,
    FinnhubStockSource,
    TwelveDataStockSource,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("novafx.stocks")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DISPATCHER_URL = os.getenv("DISPATCHER_URL", "http://localhost:8000")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "300"))

# Telegram direct send config (fallback when confluence doesn't fire)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003614474777")
TELEGRAM_DIRECT_SEND = os.getenv("TELEGRAM_DIRECT_SEND", "true").lower() == "true"

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "SPY", "QQQ", "AMD",
]


# ---------------------------------------------------------------------------
# Market hours check
# ---------------------------------------------------------------------------


def is_market_open() -> bool:
    """Check if US stock market is open (9:30 AM - 4:00 PM ET, Mon-Fri)."""
    now_utc = datetime.now(timezone.utc)

    # Simple ET offset: UTC-5 (EST) or UTC-4 (EDT)
    # DST: second Sunday in March to first Sunday in November
    month = now_utc.month
    if 3 < month < 11:
        et_offset = timedelta(hours=-4)  # EDT
    elif month == 3:
        # Approximate: after March 10
        et_offset = timedelta(hours=-4) if now_utc.day > 10 else timedelta(hours=-5)
    elif month == 11:
        # Approximate: before November 3
        et_offset = timedelta(hours=-4) if now_utc.day < 3 else timedelta(hours=-5)
    else:
        et_offset = timedelta(hours=-5)  # EST

    now_et = now_utc + et_offset

    # Monday=0, Friday=4
    if now_et.weekday() > 4:
        return False

    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)

    return market_open <= now_et <= market_close


# ---------------------------------------------------------------------------
# Telegram direct send (fallback)
# ---------------------------------------------------------------------------


def format_telegram_signal(signal: Signal) -> str:
    """Format signal as Telegram message (HTML)."""
    from datetime import datetime, timezone

    action_emoji = "\U0001f7e2" if signal.action.value == "BUY" else "\U0001f534"
    direction = f"{action_emoji} {signal.action.value.upper()}"

    # Calculate percentages for SL/TP display
    sl_pct = abs(signal.stop_loss - signal.price) / signal.price * 100 if signal.stop_loss else 0
    tp_pct = abs(signal.take_profit[0] - signal.price) / signal.price * 100 if signal.take_profit else 0

    # Get RSI from metadata
    rsi = signal.metadata.get("rsi", "N/A") if signal.metadata else "N/A"

    conf_pct = round(signal.confidence * 100)
    ts = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")

    return (
        f"\u26a1 <b>NOVAFX STOCK SIGNAL</b>\n\n"
        f"{direction}  <b>{signal.symbol}</b>\n"
        f"\U0001f4c8 Stocks\n\n"
        f"\U0001f4cd <b>Entry:</b> ${signal.price:.2f}\n"
        f"\U0001f534 <b>Stop Loss:</b> ${signal.stop_loss:.2f}  ({sl_pct:.2f}%)\n"
        f"\u2705 <b>Take Profit:</b> ${signal.take_profit[0]:.2f}  ({tp_pct:.2f}%)\n\n"
        f"\U0001f3af <b>Confidence:</b> {conf_pct}%\n"
        f"\U0001f4ca <b>RSI:</b> {rsi}\n"
        f"\U0001f9e0 <b>Strategy:</b> {signal.strategy}\n\n"
        f"\U0001f4c5 <i>{ts}</i>\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u26a0\ufe0f <i>Risk max 1-2% per trade. Not financial advice.</i>"
    )


async def send_telegram_direct(signal: Signal) -> bool:
    """Send signal directly to Telegram (bypasses confluence filter)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_DIRECT_SEND:
        logger.debug("Telegram direct send disabled or no token")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    text = format_telegram_signal(signal)

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                logger.info(
                    "Telegram direct send SUCCESS: %s %s @ $%.2f",
                    signal.action.value,
                    signal.symbol,
                    signal.price,
                )
                return True
            else:
                logger.error(
                    "Telegram direct send FAILED: %s — %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
    except Exception as exc:
        logger.error("Telegram direct send ERROR: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Signal dispatch
# ---------------------------------------------------------------------------


async def post_signal(signal: Signal) -> None:
    """POST signal to the dispatcher service."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{DISPATCHER_URL}/signals/ingest",
                json=signal.model_dump(mode="json"),
            )
            resp.raise_for_status()
            result = resp.json()
            confluence = result.get("confluence")
            logger.info(
                "Signal dispatched: %s %s @ $%.2f (confidence=%.2f)%s",
                signal.action.value,
                signal.symbol,
                signal.price,
                signal.confidence,
                " — CONFLUENCE HIT" if confluence else "",
            )
    except Exception:
        logger.exception("Failed to dispatch signal for %s", signal.symbol)


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------


async def scan_cycle(manager: DataSourceManager) -> None:
    """Run one full scan of all stock symbols."""
    signals_generated = 0
    sources_used: dict[str, int] = {}
    failures = 0

    for symbol in WATCHLIST:
        try:
            result = await manager.get_candles(symbol, "1h", 250)
            candles = result["candles"]
            source = result["source"]
            confidence = result["confidence"]
            stale = result["stale"]

            sources_used[source] = sources_used.get(source, 0) + 1

            signal = analyze_candles(
                symbol=symbol,
                candles=candles,
                data_source=source,
                data_confidence=confidence,
                data_stale=stale,
            )

            if signal:
                # Check for duplicate signal (same pair+direction within cooldown)
                signal_direction = signal.action.value
                if is_duplicate_signal(symbol, signal_direction):
                    logger.info(f"Skipping duplicate signal: {signal_direction} {symbol}")
                    continue
                # Send to dispatcher for confluence tracking
                await post_signal(signal)
                # Also send directly to Telegram (bypasses confluence filter)
                await send_telegram_direct(signal)
                # Write to Redis stream for paper trader
                await write_signal_to_stream(
                    symbol,
                    signal_direction,
                    signal.price,
                    signal.confidence,
                )
                signals_generated += 1

        except AllSourcesFailedError:
            logger.error("ALL sources failed for %s — skipping", symbol)
            failures += 1
        except Exception:
            logger.exception("Scan error for %s", symbol)
            failures += 1

    source_summary = ", ".join(f"{s}={n}" for s, n in sources_used.items())
    logger.info(
        "Scan complete: %d/%d stocks scanned, %d signals, %d failures | Sources: %s",
        len(WATCHLIST) - failures,
        len(WATCHLIST),
        signals_generated,
        failures,
        source_summary or "none",
    )


async def main() -> None:
    """Initialize data sources and run the scan loop."""
    sources = [
        AlpacaStockSource(),
        TwelveDataStockSource(),
        FinnhubStockSource(),
        AlphaVantageStockSource(),
    ]

    manager = DataSourceManager(sources)
    await manager.start_health_checks()

    # Start WebSocket standby in background
    ws_task = None
    try:
        from ws_standby import main as ws_main
        ws_task = asyncio.create_task(ws_main())
        logger.info("Alpaca WebSocket hot standby launched")
    except Exception:
        logger.warning("WebSocket standby failed to start — continuing without it")

    logger.info(
        "NovaFX Stock Scanner started — %d symbols, %d sources, interval=%ds",
        len(WATCHLIST),
        len(sources),
        SCAN_INTERVAL,
    )

    try:
        while True:
            if is_market_open():
                await scan_cycle(manager)
                await asyncio.sleep(SCAN_INTERVAL)
            else:
                logger.info("Market closed, sleeping 60s...")
                await asyncio.sleep(60)
    except asyncio.CancelledError:
        pass
    finally:
        if ws_task and not ws_task.done():
            ws_task.cancel()
        await manager.stop_health_checks()
        logger.info("Stock scanner stopped")


if __name__ == "__main__":
    asyncio.run(main())

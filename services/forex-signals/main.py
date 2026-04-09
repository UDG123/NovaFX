"""
NovaFX Forex Signal Generator.

Scans 8 major forex pairs on 1H candles with four-tier data source
failover: TwelveData -> Finnhub -> FMP -> Alpha Vantage.
Generates RSI + EMA crossover signals and POSTs to the dispatcher.
Also sends directly to Telegram as a fallback (bypasses confluence filter).
"""

import asyncio
import logging
import os
import sys
from datetime import datetime, timezone

import httpx

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from shared.models import Signal
from shared.resilience import AllSourcesFailedError, DataSourceManager
from signals import analyze_candles
from sources import (
    AlphaVantageForexSource,
    FinnhubForexSource,
    FMPForexSource,
    TwelveDataForexSource,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("novafx.forex")
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
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CAD", "EUR/GBP", "NZD/USD", "USD/CHF",
]


# ---------------------------------------------------------------------------
# Telegram direct send (fallback)
# ---------------------------------------------------------------------------


def format_telegram_signal(signal: Signal) -> str:
    """Format signal as Telegram message (HTML)."""
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
        f"\u26a1 <b>NOVAFX FOREX SIGNAL</b>\n\n"
        f"{direction}  <b>{signal.symbol}</b>\n"
        f"\U0001f4b1 Forex\n\n"
        f"\U0001f4cd <b>Entry:</b> {signal.price:.5f}\n"
        f"\U0001f534 <b>Stop Loss:</b> {signal.stop_loss:.5f}  ({sl_pct:.2f}%)\n"
        f"\u2705 <b>Take Profit:</b> {signal.take_profit[0]:.5f}  ({tp_pct:.2f}%)\n\n"
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
                    "Telegram direct send SUCCESS: %s %s @ %.5f",
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
                "Signal dispatched: %s %s @ %.5f (confidence=%.2f)%s",
                signal.action.value,
                signal.symbol,
                signal.price,
                signal.confidence,
                f" — CONFLUENCE HIT" if confluence else "",
            )
    except Exception:
        logger.exception("Failed to dispatch signal for %s", signal.symbol)


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------


async def scan_cycle(manager: DataSourceManager) -> None:
    """Run one full scan of all forex pairs."""
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
                # Send to dispatcher for confluence tracking
                await post_signal(signal)
                # Also send directly to Telegram (bypasses confluence filter)
                await send_telegram_direct(signal)
                signals_generated += 1

        except AllSourcesFailedError:
            logger.error("ALL sources failed for %s — skipping", symbol)
            failures += 1
        except Exception:
            logger.exception("Scan error for %s", symbol)
            failures += 1

    # Log summary
    source_summary = ", ".join(f"{s}={n}" for s, n in sources_used.items())
    logger.info(
        "Scan complete: %d/%d pairs scanned, %d signals, %d failures | Sources: %s",
        len(WATCHLIST) - failures,
        len(WATCHLIST),
        signals_generated,
        failures,
        source_summary or "none",
    )


async def main() -> None:
    """Initialize data sources and run the scan loop."""
    sources = [
        TwelveDataForexSource(),
        FinnhubForexSource(),
        FMPForexSource(),
        AlphaVantageForexSource(),
    ]

    manager = DataSourceManager(sources)
    await manager.start_health_checks()

    logger.info(
        "NovaFX Forex Scanner started — %d pairs, %d sources, interval=%ds",
        len(WATCHLIST),
        len(sources),
        SCAN_INTERVAL,
    )

    try:
        while True:
            await scan_cycle(manager)
            await asyncio.sleep(SCAN_INTERVAL)
    except asyncio.CancelledError:
        pass
    finally:
        await manager.stop_health_checks()
        logger.info("Forex scanner stopped")


if __name__ == "__main__":
    asyncio.run(main())

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
            result = await manager.get_candles(symbol, "1h", 100)
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
                await post_signal(signal)
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

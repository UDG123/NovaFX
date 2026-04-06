"""
NovaFX Forex Signal Generator.

Scans 8 major forex pairs on 1H candles with four-tier data source
failover: TwelveData -> Finnhub -> FMP -> Alpha Vantage.
Generates RSI + EMA crossover signals and POSTs to the dispatcher.
"""

import asyncio
import logging
import os
import sys

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

WATCHLIST = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD",
    "USD/CAD", "EUR/GBP", "NZD/USD", "USD/CHF",
]


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

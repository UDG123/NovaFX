"""
NovaFX Alpaca WebSocket Hot Standby.

Always-on WebSocket connection to Alpaca IEX for real-time minute bars.
Aggregates into 1H bars client-side and caches as emergency data source.
Auto-reconnects with exponential backoff.
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("novafx.stocks.ws")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
WS_URL = "wss://stream.data.alpaca.markets/v2/iex"

WATCHLIST = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "META", "TSLA", "SPY", "QQQ", "AMD",
]

# Cache: symbol -> latest complete 1H bar
_hourly_cache: dict[str, dict] = {}

# Accumulator: symbol -> list of minute bars for current hour
_minute_accum: dict[str, list[dict]] = defaultdict(list)
_current_hour: dict[str, int] = {}


def get_cached_bar(symbol: str) -> Optional[dict]:
    """Get the latest cached 1H bar for a symbol."""
    return _hourly_cache.get(symbol)


def _aggregate_to_hourly(symbol: str, minute_bars: list[dict]) -> dict:
    """Aggregate minute bars into a single 1H bar."""
    return {
        "timestamp": minute_bars[0]["timestamp"],
        "open": minute_bars[0]["open"],
        "high": max(b["high"] for b in minute_bars),
        "low": min(b["low"] for b in minute_bars),
        "close": minute_bars[-1]["close"],
        "volume": sum(b["volume"] for b in minute_bars),
        "cached_at": time.time(),
    }


def _process_bar(symbol: str, bar: dict) -> None:
    """Process a minute bar, aggregate into hourly when hour rolls."""
    ts = bar.get("timestamp", "")
    # Extract hour from ISO timestamp (e.g. "2026-04-06T14:30:00Z" -> 14)
    try:
        hour = int(ts[11:13]) if isinstance(ts, str) and len(ts) > 13 else 0
    except (ValueError, IndexError):
        hour = 0

    prev_hour = _current_hour.get(symbol)

    if prev_hour is not None and hour != prev_hour and _minute_accum[symbol]:
        # Hour rolled — aggregate and cache
        _hourly_cache[symbol] = _aggregate_to_hourly(symbol, _minute_accum[symbol])
        _minute_accum[symbol] = []
        logger.debug("1H bar cached: %s", symbol)

    _current_hour[symbol] = hour
    _minute_accum[symbol].append(bar)


async def _run_stream() -> None:
    """Connect to Alpaca WebSocket and process minute bars."""
    import websockets

    backoff = 1

    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                logger.info("Alpaca WebSocket connected")
                backoff = 1

                # Authenticate
                auth_msg = json.dumps({
                    "action": "auth",
                    "key": ALPACA_API_KEY,
                    "secret": ALPACA_SECRET_KEY,
                })
                await ws.send(auth_msg)
                auth_resp = await ws.recv()
                logger.debug("Auth response: %s", auth_resp)

                # Subscribe to minute bars for all watchlist symbols
                sub_msg = json.dumps({
                    "action": "subscribe",
                    "bars": WATCHLIST,
                })
                await ws.send(sub_msg)
                sub_resp = await ws.recv()
                logger.info("Subscribed to %d symbols", len(WATCHLIST))

                # Process incoming bars
                async for raw in ws:
                    try:
                        messages = json.loads(raw)
                        if not isinstance(messages, list):
                            continue

                        for msg in messages:
                            if msg.get("T") != "b":  # "b" = bar
                                continue

                            symbol = msg.get("S", "")
                            if symbol not in WATCHLIST:
                                continue

                            bar = {
                                "timestamp": msg.get("t", ""),
                                "open": float(msg.get("o", 0)),
                                "high": float(msg.get("h", 0)),
                                "low": float(msg.get("l", 0)),
                                "close": float(msg.get("c", 0)),
                                "volume": int(msg.get("v", 0)),
                            }
                            _process_bar(symbol, bar)

                    except Exception:
                        continue

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(
                "Alpaca WebSocket disconnected: %s (reconnecting in %ds)",
                exc, backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def main() -> None:
    """Entry point for WebSocket hot standby."""
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        logger.error("Alpaca credentials not set — WebSocket standby disabled")
        return

    logger.info("Alpaca WebSocket hot standby started for %d symbols", len(WATCHLIST))
    await _run_stream()


if __name__ == "__main__":
    asyncio.run(main())

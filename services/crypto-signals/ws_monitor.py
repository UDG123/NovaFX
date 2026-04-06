"""
NovaFX WebSocket Hot Standby Monitor.

Maintains real-time WebSocket connections to Binance and Kraken
for emergency candle data when both Freqtrade and fallback scanner fail.

Auto-reconnects with exponential backoff on disconnect.
"""

import asyncio
import json
import logging
import os
import sys
import time
from typing import Optional

sys.path.insert(0, "/freqtrade")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("novafx.crypto.ws")

# In-memory cache for latest complete candles
# Keyed by f"{exchange}:{symbol}"
_candle_cache: dict[str, dict] = {}


def get_cached_candle(exchange: str, symbol: str) -> Optional[dict]:
    """Retrieve a cached candle from WebSocket data."""
    return _candle_cache.get(f"{exchange}:{symbol}")


async def _binance_stream() -> None:
    """Connect to Binance kline WebSocket for BTC/USDT 1h candles."""
    import websockets

    url = "wss://data-stream.binance.vision/ws/btcusdt@kline_1h"
    backoff = 1

    while True:
        try:
            async with websockets.connect(url) as ws:
                logger.info("Binance WebSocket connected")
                backoff = 1

                async for raw in ws:
                    try:
                        msg = json.loads(raw)
                        kline = msg.get("k", {})
                        if not kline:
                            continue

                        # Only cache completed candles
                        if kline.get("x", False):
                            _candle_cache["binance:BTCUSDT"] = {
                                "timestamp": kline["t"],
                                "open": float(kline["o"]),
                                "high": float(kline["h"]),
                                "low": float(kline["l"]),
                                "close": float(kline["c"]),
                                "volume": float(kline["v"]),
                                "cached_at": time.time(),
                            }
                            logger.debug("Binance BTC/USDT 1h candle cached")
                    except Exception:
                        continue

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Binance WebSocket disconnected: %s (reconnecting in %ds)", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def _kraken_stream() -> None:
    """Connect to Kraken WebSocket for XBT/USD OHLC data."""
    import websockets

    url = "wss://ws.kraken.com"
    backoff = 1

    while True:
        try:
            async with websockets.connect(url) as ws:
                logger.info("Kraken WebSocket connected")
                backoff = 1

                subscribe_msg = json.dumps({
                    "event": "subscribe",
                    "pair": ["XBT/USD"],
                    "subscription": {"name": "ohlc", "interval": 60},
                })
                await ws.send(subscribe_msg)

                async for raw in ws:
                    try:
                        msg = json.loads(raw)

                        # Skip system messages
                        if isinstance(msg, dict):
                            continue

                        # OHLC data comes as [channelID, data, channelName, pair]
                        if isinstance(msg, list) and len(msg) >= 4:
                            ohlc_data = msg[1]
                            if isinstance(ohlc_data, list) and len(ohlc_data) >= 6:
                                _candle_cache["kraken:XBTUSD"] = {
                                    "timestamp": float(ohlc_data[0]),
                                    "open": float(ohlc_data[2]),
                                    "high": float(ohlc_data[3]),
                                    "low": float(ohlc_data[4]),
                                    "close": float(ohlc_data[5]),
                                    "volume": float(ohlc_data[7]) if len(ohlc_data) > 7 else 0,
                                    "cached_at": time.time(),
                                }
                                logger.debug("Kraken XBT/USD OHLC cached")
                    except Exception:
                        continue

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Kraken WebSocket disconnected: %s (reconnecting in %ds)", exc, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60)


async def main() -> None:
    """Run both WebSocket streams concurrently."""
    logger.info("WebSocket hot standby monitor started")
    await asyncio.gather(
        _binance_stream(),
        _kraken_stream(),
        return_exceptions=True,
    )


if __name__ == "__main__":
    asyncio.run(main())

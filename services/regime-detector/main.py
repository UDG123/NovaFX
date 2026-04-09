"""
NovaFX v2 Regime Detector
Classifies market regime every 15 minutes for FOREX, CRYPTO, and STOCKS.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Literal

import httpx
import numpy as np
import redis.asyncio as redis
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Environment
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PORT = int(os.getenv("PORT", 8000))
DETECT_INTERVAL_SECONDS = int(os.getenv("DETECT_INTERVAL_SECONDS", 900))  # 15 min default

# Anchor symbols for each desk
FOREX_ANCHOR = "EUR/USD"
CRYPTO_ANCHOR = "BTC/USD"
STOCKS_ANCHOR = "SPY"

# Redis key
REGIME_KEY = "novafx:regime"

RegimeType = Literal["trending", "mild_trend", "ranging", "risk_off"]


async def fetch_candles(client: httpx.AsyncClient, symbol: str, interval: str, outputsize: int = 100) -> list[dict]:
    """Fetch OHLCV candles from TwelveData."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
    }
    try:
        r = await client.get(url, params=params, timeout=15)
        data = r.json()
        if "values" not in data:
            logger.warning(f"No values for {symbol} {interval}: {data.get('message', 'unknown error')}")
            return []
        return list(reversed(data["values"]))  # oldest first
    except Exception as e:
        logger.error(f"Error fetching {symbol} {interval}: {e}")
        return []


def calc_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA."""
    if len(data) < period:
        return data
    k = 2 / (period + 1)
    ema = np.zeros(len(data))
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = data[i] * k + ema[i - 1] * (1 - k)
    return ema


def calc_adx_proxy(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    """Calculate ADX proxy (simplified)."""
    if len(closes) < period + 1:
        return 15.0

    tr_list = []
    plus_dm_list = []
    minus_dm_list = []

    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_high = highs[i - 1]
        prev_low = lows[i - 1]
        prev_close = closes[i - 1]

        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        plus_dm = max(0, high - prev_high) if high - prev_high > prev_low - low else 0
        minus_dm = max(0, prev_low - low) if prev_low - low > high - prev_high else 0

        tr_list.append(tr)
        plus_dm_list.append(plus_dm)
        minus_dm_list.append(minus_dm)

    if len(tr_list) < period:
        return 15.0

    atr = np.mean(tr_list[-period:])
    plus_di = 100 * np.mean(plus_dm_list[-period:]) / atr if atr > 0 else 0
    minus_di = 100 * np.mean(minus_dm_list[-period:]) / atr if atr > 0 else 0

    if plus_di + minus_di == 0:
        return 15.0

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx


def calc_atr_percent(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    """Calculate ATR as percentage of current price."""
    if len(closes) < period + 1:
        return 1.0

    tr_list = []
    for i in range(1, len(closes)):
        high = highs[i]
        low = lows[i]
        prev_close = closes[i - 1]
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        tr_list.append(tr)

    atr = np.mean(tr_list[-period:])
    current_price = closes[-1]
    return (atr / current_price) * 100 if current_price > 0 else 1.0


def is_prime_session() -> bool:
    """Check if in London-NY overlap (13:00-17:00 UTC)."""
    hour = datetime.now(timezone.utc).hour
    return 13 <= hour < 17


def classify_regime(adx: float, atr_pct: float, ema20: float, ema50: float, price: float) -> RegimeType:
    """
    Classify market regime.
    - trending: ADX > 25
    - mild_trend: ADX 15-25
    - ranging: ADX < 15
    - risk_off: ATR% > 2.0 (high volatility)
    """
    # Risk-off takes priority if volatility is extreme
    if atr_pct > 2.0:
        return "risk_off"

    if adx > 25:
        return "trending"
    elif adx >= 15:
        return "mild_trend"
    else:
        return "ranging"


async def analyze_anchor(client: httpx.AsyncClient, symbol: str) -> dict:
    """Analyze an anchor symbol and return regime data."""
    candles = await fetch_candles(client, symbol, "4h", 60)

    if not candles or len(candles) < 50:
        return {
            "symbol": symbol,
            "regime": "mild_trend",
            "adx": 20.0,
            "atr_pct": 1.0,
            "ema20_vs_ema50": "neutral",
            "error": "insufficient_data",
        }

    closes = [float(c["close"]) for c in candles]
    highs = [float(c["high"]) for c in candles]
    lows = [float(c["low"]) for c in candles]

    current_price = closes[-1]

    # Calculate indicators
    adx = calc_adx_proxy(highs, lows, closes)
    atr_pct = calc_atr_percent(highs, lows, closes)
    ema20 = calc_ema(np.array(closes), 20)[-1]
    ema50 = calc_ema(np.array(closes), 50)[-1]

    # EMA relationship
    if ema20 > ema50 * 1.002:
        ema_rel = "bullish"
    elif ema20 < ema50 * 0.998:
        ema_rel = "bearish"
    else:
        ema_rel = "neutral"

    regime = classify_regime(adx, atr_pct, ema20, ema50, current_price)

    return {
        "symbol": symbol,
        "regime": regime,
        "adx": round(adx, 1),
        "atr_pct": round(atr_pct, 3),
        "ema20": round(ema20, 6),
        "ema50": round(ema50, 6),
        "ema20_vs_ema50": ema_rel,
        "price": round(current_price, 6),
    }


async def detect_regimes() -> dict:
    """Detect regime for all desks."""
    async with httpx.AsyncClient() as client:
        forex_data, crypto_data, stocks_data = await asyncio.gather(
            analyze_anchor(client, FOREX_ANCHOR),
            analyze_anchor(client, CRYPTO_ANCHOR),
            analyze_anchor(client, STOCKS_ANCHOR),
        )

    prime_session = is_prime_session()

    return {
        "forex": {
            **forex_data,
            "prime_session": prime_session,
        },
        "crypto": crypto_data,
        "stocks": stocks_data,
        "prime_session": prime_session,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def run_detector() -> None:
    """Main detector loop."""
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)

    while True:
        try:
            logger.info("Detecting market regimes...")

            regime_data = await detect_regimes()

            # Publish to Redis
            await redis_client.set(REGIME_KEY, json.dumps(regime_data))
            logger.info(
                f"Regime updated: forex={regime_data['forex']['regime']}, "
                f"crypto={regime_data['crypto']['regime']}, "
                f"stocks={regime_data['stocks']['regime']}, "
                f"prime_session={regime_data['prime_session']}"
            )

            await asyncio.sleep(DETECT_INTERVAL_SECONDS)

        except Exception as e:
            logger.error(f"Detector error: {e}")
            await asyncio.sleep(30)


# Health server
async def health_handler(request: web.Request) -> web.Response:
    return web.Response(text='{"status":"ok"}', content_type="application/json")


async def regime_handler(request: web.Request) -> web.Response:
    """Return current regime state."""
    try:
        redis_client = redis.from_url(REDIS_URL, decode_responses=True)
        regime_data = await redis_client.get(REGIME_KEY)
        if regime_data:
            return web.Response(text=regime_data, content_type="application/json")
        return web.Response(text='{"error":"no_data"}', content_type="application/json", status=503)
    except Exception as e:
        return web.Response(text=json.dumps({"error": str(e)}), content_type="application/json", status=500)


async def start_health_server() -> None:
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/", health_handler)
    app.router.add_get("/regime", regime_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info(f"Health server started on port {PORT}")


async def main() -> None:
    logger.info("NovaFX v2 Regime Detector starting...")
    await asyncio.gather(
        start_health_server(),
        run_detector(),
    )


if __name__ == "__main__":
    asyncio.run(main())

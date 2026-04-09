"""
NovaFX v2 Universal Scanner - MTF Ensemble Scanner
Covers FOREX, CRYPTO, and STOCKS desks with multi-timeframe confluence signals.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

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
SCAN_INTERVAL_SECONDS = int(os.getenv("SCAN_INTERVAL_SECONDS", 300))  # 5 min default

# Asset Coverage
FOREX_PAIRS = ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "EUR/GBP", "GBP/JPY"]
CRYPTO_PAIRS = ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "AVAX/USD", "DOGE/USD", "LINK/USD"]
STOCK_SYMBOLS = ["AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "SPY", "QQQ"]

# Redis stream keys
STREAM_FOREX = "novafx:signals:forex"
STREAM_CRYPTO = "novafx:signals:crypto"
STREAM_STOCKS = "novafx:signals:stocks"

# Dedup cache: {symbol_direction: timestamp}
dedup_cache: dict[str, float] = {}
DEDUP_COOLDOWN_SECONDS = 3600  # 1 hour


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


def calc_rsi(closes: list[float], period: int = 14) -> float:
    """Calculate RSI."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA."""
    k = 2 / (period + 1)
    ema = np.zeros(len(data))
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = data[i] * k + ema[i - 1] * (1 - k)
    return ema


def calc_macd(closes: list[float]) -> tuple[float, float]:
    """Calculate MACD line and histogram."""
    if len(closes) < 35:
        return 0.0, 0.0
    arr = np.array(closes)
    ema12 = calc_ema(arr, 12)
    ema26 = calc_ema(arr, 26)
    macd_line = ema12 - ema26
    signal_line = calc_ema(macd_line, 9)
    histogram = macd_line[-1] - signal_line[-1]
    return macd_line[-1], histogram


def calc_bollinger_pct_b(closes: list[float], period: int = 20, std_dev: float = 2.0) -> float:
    """Calculate Bollinger %B (0 = lower band, 1 = upper band)."""
    if len(closes) < period:
        return 0.5
    arr = np.array(closes[-period:])
    sma = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return 0.5
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    current = closes[-1]
    if upper == lower:
        return 0.5
    return (current - lower) / (upper - lower)


def calc_adx(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> float:
    """Calculate ADX (Average Directional Index)."""
    if len(closes) < period + 1:
        return 15.0

    # Calculate True Range and Directional Movement
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

    # Smoothed averages
    atr = np.mean(tr_list[-period:])
    plus_di = 100 * np.mean(plus_dm_list[-period:]) / atr if atr > 0 else 0
    minus_di = 100 * np.mean(minus_dm_list[-period:]) / atr if atr > 0 else 0

    if plus_di + minus_di == 0:
        return 15.0

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    return dx


def is_deduped(symbol: str, direction: str) -> bool:
    """Check if signal was recently sent (1hr cooldown)."""
    key = f"{symbol}_{direction}"
    now = datetime.now(timezone.utc).timestamp()
    if key in dedup_cache:
        if now - dedup_cache[key] < DEDUP_COOLDOWN_SECONDS:
            return True
    dedup_cache[key] = now
    return False


async def analyze_symbol(
    client: httpx.AsyncClient,
    symbol: str,
    desk: str
) -> Optional[dict]:
    """
    Analyze a symbol using MTF ensemble logic.
    All 4 conditions must align for a signal.
    """
    # Session filter for forex: skip 21:00-23:00 UTC
    hour = datetime.now(timezone.utc).hour
    if desk == "forex" and hour in (21, 22):
        return None

    # Fetch 1h and 4h candles concurrently
    candles_1h, candles_4h = await asyncio.gather(
        fetch_candles(client, symbol, "1h", 210),  # Need 200 for EMA200
        fetch_candles(client, symbol, "4h", 50),
    )

    if not candles_1h or len(candles_1h) < 200 or not candles_4h:
        return None

    # Extract price data
    closes_1h = [float(c["close"]) for c in candles_1h]
    highs_1h = [float(c["high"]) for c in candles_1h]
    lows_1h = [float(c["low"]) for c in candles_1h]
    closes_4h = [float(c["close"]) for c in candles_4h]

    current_price = closes_1h[-1]

    # Calculate indicators
    rsi_1h = calc_rsi(closes_1h)
    rsi_4h = calc_rsi(closes_4h)
    macd_line, macd_hist = calc_macd(closes_1h)
    boll_pct_b = calc_bollinger_pct_b(closes_1h)
    ema_200 = calc_ema(np.array(closes_1h), 200)[-1]
    adx = calc_adx(highs_1h, lows_1h, closes_1h)

    # Determine signal direction based on 200 EMA (MOST IMPORTANT FILTER)
    above_ema200 = current_price > ema_200

    signal = None
    score = 0
    reasons = []

    # BUY CONDITIONS (all must align)
    if above_ema200:  # Only BUY signals when above 200 EMA
        buy_conditions = 0

        # 1. 1h RSI oversold (<35)
        if rsi_1h < 35:
            buy_conditions += 1
            reasons.append(f"1h RSI oversold: {rsi_1h:.1f}")
            score += 20 + (35 - rsi_1h)  # Bonus for deeper oversold

        # 2. 4h RSI neutral (42-58) - not exhausted
        if 42 <= rsi_4h <= 58:
            buy_conditions += 1
            reasons.append(f"4h RSI neutral: {rsi_4h:.1f}")
            score += 15

        # 3. MACD histogram positive
        if macd_hist > 0:
            buy_conditions += 1
            reasons.append(f"MACD hist positive: {macd_hist:.6f}")
            score += 15

        # 4. Bollinger %B < 0.2 (near lower band)
        if boll_pct_b < 0.2:
            buy_conditions += 1
            reasons.append(f"Boll %B near lower: {boll_pct_b:.2f}")
            score += 15 + (0.2 - boll_pct_b) * 50

        if buy_conditions == 4:
            signal = "BUY"
            score += 10  # Confluence bonus

    # SELL CONDITIONS (all must align)
    elif not above_ema200:  # Only SELL signals when below 200 EMA
        sell_conditions = 0

        # 1. 1h RSI overbought (>65)
        if rsi_1h > 65:
            sell_conditions += 1
            reasons.append(f"1h RSI overbought: {rsi_1h:.1f}")
            score += 20 + (rsi_1h - 65)

        # 2. 4h RSI neutral (42-58) - not exhausted
        if 42 <= rsi_4h <= 58:
            sell_conditions += 1
            reasons.append(f"4h RSI neutral: {rsi_4h:.1f}")
            score += 15

        # 3. MACD histogram negative
        if macd_hist < 0:
            sell_conditions += 1
            reasons.append(f"MACD hist negative: {macd_hist:.6f}")
            score += 15

        # 4. Bollinger %B > 0.8 (near upper band)
        if boll_pct_b > 0.8:
            sell_conditions += 1
            reasons.append(f"Boll %B near upper: {boll_pct_b:.2f}")
            score += 15 + (boll_pct_b - 0.8) * 50

        if sell_conditions == 4:
            signal = "SELL"
            score += 10  # Confluence bonus

    if not signal:
        return None

    # ADX trending bonus
    if adx > 20:
        score += 5
        reasons.append(f"ADX trending: {adx:.1f}")

    # Cap score at 100
    score = min(100, int(score))

    # Only publish if score >= 65
    if score < 65:
        return None

    # Dedup check
    if is_deduped(symbol, signal):
        logger.info(f"Deduped: {symbol} {signal} (1hr cooldown)")
        return None

    return {
        "symbol": symbol,
        "direction": signal,
        "desk": desk,
        "score": score,
        "entry": round(current_price, 6),
        "rsi_1h": round(rsi_1h, 1),
        "rsi_4h": round(rsi_4h, 1),
        "macd_hist": round(macd_hist, 6),
        "boll_pct_b": round(boll_pct_b, 3),
        "ema_200": round(ema_200, 6),
        "adx": round(adx, 1),
        "reasons": reasons,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def publish_signal(redis_client: redis.Redis, stream: str, signal: dict) -> None:
    """Publish signal to Redis stream."""
    try:
        await redis_client.xadd(stream, {"data": json.dumps(signal)})
        logger.info(f"Published to {stream}: {signal['symbol']} {signal['direction']} score={signal['score']}")
    except Exception as e:
        logger.error(f"Error publishing to {stream}: {e}")


async def scan_desk(
    http_client: httpx.AsyncClient,
    redis_client: redis.Redis,
    symbols: list[str],
    desk: str,
    stream: str,
) -> None:
    """Scan all symbols in a desk."""
    logger.info(f"Scanning {desk} desk: {len(symbols)} symbols")

    # Process symbols with rate limiting (TwelveData has limits)
    for symbol in symbols:
        try:
            signal = await analyze_symbol(http_client, symbol, desk)
            if signal:
                await publish_signal(redis_client, stream, signal)
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

        await asyncio.sleep(2)  # 2s between instruments to stay within rate limits


async def run_scanner() -> None:
    """Main scanner loop."""
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)

    async with httpx.AsyncClient() as http_client:
        while True:
            try:
                logger.info("Starting scan cycle...")

                # Scan desks with 65s delays to stay within TwelveData rate limits (55/min)
                # Each symbol makes 2 API calls (1h + 4h), so staggering prevents quota exhaustion
                await scan_desk(http_client, redis_client, FOREX_PAIRS, "forex", STREAM_FOREX)
                logger.info("Forex scan complete. Waiting 65s before crypto scan...")
                await asyncio.sleep(65)

                await scan_desk(http_client, redis_client, CRYPTO_PAIRS, "crypto", STREAM_CRYPTO)
                logger.info("Crypto scan complete. Waiting 65s before stocks scan...")
                await asyncio.sleep(65)

                await scan_desk(http_client, redis_client, STOCK_SYMBOLS, "stocks", STREAM_STOCKS)

                # Remaining interval after ~130s desk stagger overhead
                remaining_sleep = max(0, SCAN_INTERVAL_SECONDS - 130)
                logger.info(f"Scan cycle complete. Sleeping {remaining_sleep}s...")
                await asyncio.sleep(remaining_sleep)

            except Exception as e:
                logger.error(f"Scanner error: {e}")
                await asyncio.sleep(30)


# Health server
async def health_handler(request: web.Request) -> web.Response:
    return web.Response(text='{"status":"ok"}', content_type="application/json")


async def start_health_server() -> None:
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info(f"Health server started on port {PORT}")


async def main() -> None:
    logger.info("NovaFX v2 Universal Scanner starting...")
    await asyncio.gather(
        start_health_server(),
        run_scanner(),
    )


if __name__ == "__main__":
    asyncio.run(main())

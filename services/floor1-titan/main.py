"""
FLOOR 1 - TITAN: London Breakout on Forex Majors
Wake at 07:00 UTC daily. Asian range breakout with SMA(200) trend filter.
"""
import asyncio
import os
import json
from datetime import datetime, timezone, timedelta
import httpx
import redis.asyncio as redis
from aiohttp import web
import numpy as np

# Config
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PORT = int(os.environ.get("PORT", "8001"))

# Instruments
INSTRUMENTS = ["GBP/USD", "EUR/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD"]

# Pip sizes
def get_pip_size(symbol: str) -> float:
    return 0.01 if "JPY" in symbol else 0.0001

def get_range_filter(symbol: str) -> tuple[int, int]:
    if symbol in ["GBP/USD", "EUR/USD"]:
        return 15, 80
    return 15, 60


async def fetch_candles(client: httpx.AsyncClient, symbol: str, interval: str, outputsize: int) -> list[dict]:
    """Fetch OHLCV candles from TwelveData."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
    }
    resp = await client.get(url, params=params)
    data = resp.json()
    if "values" not in data:
        print(f"[TITAN] Error fetching {symbol} {interval}: {data.get('message', 'unknown')}")
        return []
    return data["values"]


def calc_sma(closes: list[float], period: int) -> float:
    if len(closes) < period:
        return 0.0
    return np.mean(closes[-period:])


def calc_atr(candles: list[dict], period: int = 14) -> float:
    if len(candles) < period + 1:
        return 0.0
    trs = []
    for i in range(1, len(candles)):
        h = float(candles[i]["high"])
        l = float(candles[i]["low"])
        pc = float(candles[i - 1]["close"])
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    return np.mean(trs[-period:])


async def scan_instrument(client: httpx.AsyncClient, r: redis.Redis, symbol: str):
    """Scan a single instrument for London breakout signal."""
    pip_size = get_pip_size(symbol)
    min_pips, max_pips = get_range_filter(symbol)

    # Fetch 15-min candles (60 bars = 15 hours)
    candles_15m = await fetch_candles(client, symbol, "15min", 60)
    if len(candles_15m) < 28:  # Need at least 7 hours of data
        print(f"[TITAN] {symbol}: insufficient 15m data")
        return

    # Reverse to chronological order
    candles_15m = candles_15m[::-1]

    # Asian range: bars where hour < 7 UTC
    asian_highs = []
    asian_lows = []
    for c in candles_15m:
        dt = datetime.fromisoformat(c["datetime"].replace(" ", "T"))
        if dt.hour < 7:
            asian_highs.append(float(c["high"]))
            asian_lows.append(float(c["low"]))

    if not asian_highs:
        print(f"[TITAN] {symbol}: no Asian session data")
        return

    asian_high = max(asian_highs)
    asian_low = min(asian_lows)
    asian_range = asian_high - asian_low
    range_pips = int(asian_range / pip_size)

    # Range filter
    if range_pips < min_pips or range_pips > max_pips:
        print(f"[TITAN] {symbol}: range {range_pips} pips outside filter ({min_pips}-{max_pips})")
        return

    # Fetch daily candles for SMA(200)
    candles_daily = await fetch_candles(client, symbol, "1day", 210)
    if len(candles_daily) < 200:
        print(f"[TITAN] {symbol}: insufficient daily data for SMA200")
        return

    candles_daily = candles_daily[::-1]
    daily_closes = [float(c["close"]) for c in candles_daily]
    sma200 = calc_sma(daily_closes, 200)
    daily_close = daily_closes[-1]

    # Current price (latest 15m close)
    current_price = float(candles_15m[-1]["close"])

    # Determine direction
    direction = None
    buffer = 0.0005

    if current_price > asian_high + buffer and daily_close > sma200:
        direction = "LONG"
        entry = current_price
        sl = asian_low - buffer
        tp = entry + (asian_range * 1.5)
    elif current_price < asian_low - buffer and daily_close < sma200:
        direction = "SHORT"
        entry = current_price
        sl = asian_high + buffer
        tp = entry - (asian_range * 1.5)
    else:
        print(f"[TITAN] {symbol}: no breakout signal (price={current_price}, high={asian_high}, low={asian_low})")
        return

    # Dedup check (24hr)
    dedup_key = f"novafx:dedup:floor1:{symbol}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    if await r.exists(dedup_key):
        print(f"[TITAN] {symbol}: already signaled today")
        return

    # Calculate ATR for context
    atr = calc_atr([{"high": c["high"], "low": c["low"], "close": c["close"]} for c in candles_daily], 14)

    # Create signal
    rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0
    signal = {
        "floor": "TITAN",
        "symbol": symbol,
        "direction": direction,
        "entry": round(entry, 5),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "rr": round(rr, 2),
        "range_pips": range_pips,
        "sma200": round(sma200, 5),
        "atr": round(atr, 5),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Publish to Redis stream
    await r.xadd("novafx:signals:floor1", {k: str(v) for k, v in signal.items()})
    await r.set(dedup_key, "1", ex=86400)

    # Send Telegram
    await send_telegram(signal)
    print(f"[TITAN] Signal: {symbol} {direction} @ {entry}")


async def send_telegram(signal: dict):
    """Send signal to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[TITAN] Telegram not configured")
        return

    direction_emoji = "📈" if signal["direction"] == "LONG" else "📉"
    trend = "Above" if signal["direction"] == "LONG" else "Below"

    msg = f"""⚡ TITAN FOREX | {signal["symbol"]} | {direction_emoji} {signal["direction"]} @ {signal["entry"]}

🔴 SL: {signal["sl"]} | ✅ TP: {signal["tp"]}
⚖️ R:R 1:{signal["rr"]} | 📊 Range: {signal["range_pips"]} pips

🧠 London Breakout | SMA200: {signal["sma200"]} ({trend})
⏰ Valid until 10:00 UTC
📅 {signal["timestamp"][:19]} UTC

────────────────
⚠️ Risk max 1-2% per trade. Not financial advice."""

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"})


async def wait_until_target_time():
    """Sleep until 07:00 UTC."""
    now = datetime.now(timezone.utc)
    target = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)

    wait_seconds = (target - now).total_seconds()
    print(f"[TITAN] Sleeping {wait_seconds:.0f}s until {target.isoformat()}")
    await asyncio.sleep(wait_seconds)


async def scan_loop(r: redis.Redis):
    """Main scanning loop - wake at 07:00 UTC daily."""
    while True:
        await wait_until_target_time()
        print(f"[TITAN] Starting London session scan at {datetime.now(timezone.utc).isoformat()}")

        async with httpx.AsyncClient(timeout=30) as client:
            for symbol in INSTRUMENTS:
                try:
                    await scan_instrument(client, r, symbol)
                    await asyncio.sleep(2)  # Rate limit buffer
                except Exception as e:
                    print(f"[TITAN] Error scanning {symbol}: {e}")


async def health_handler(request):
    return web.Response(text="OK", status=200)


async def run_health_server():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"[TITAN] Health server on port {PORT}")


async def main():
    print("[TITAN] Floor 1 - London Breakout Scanner starting...")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    await run_health_server()
    await scan_loop(r)


if __name__ == "__main__":
    asyncio.run(main())

"""
FLOOR 4 - VERTEX: VWAP Mean Reversion on US Equities
Active 14:30-21:00 UTC only. Scan every 5 minutes.
"""
import asyncio
import os
import json
from datetime import datetime, timezone, time
import httpx
import redis.asyncio as redis
from aiohttp import web
import numpy as np

# Config
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PORT = int(os.environ.get("PORT", "8004"))

# Instruments
INSTRUMENTS = ["NVDA", "TSLA", "AAPL", "META", "AMD"]

SCAN_INTERVAL_MINUTES = 5
MARKET_START = time(14, 30)  # UTC
MARKET_END = time(21, 0)     # UTC


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
        print(f"[VERTEX] Error fetching {symbol}: {data.get('message', 'unknown')}")
        return []
    return data["values"]


def calc_vwap(candles: list[dict]) -> tuple[float, float, float, float]:
    """
    Calculate VWAP and standard deviation bands.
    Returns: (vwap, upper_2sd, lower_2sd, upper_3sd)
    """
    if not candles:
        return 0, 0, 0, 0

    typical_prices = []
    volumes = []
    tp_vol = []

    for c in candles:
        h = float(c["high"])
        l = float(c["low"])
        close = float(c["close"])
        vol = float(c.get("volume", 0))

        tp = (h + l + close) / 3
        typical_prices.append(tp)
        volumes.append(vol)
        tp_vol.append(tp * vol)

    cum_vol = np.cumsum(volumes)
    cum_tp_vol = np.cumsum(tp_vol)

    if cum_vol[-1] == 0:
        return 0, 0, 0, 0

    vwap = cum_tp_vol[-1] / cum_vol[-1]

    # Rolling std of closes for bands
    closes = [float(c["close"]) for c in candles]
    if len(closes) >= 20:
        std = np.std(closes[-20:])
    else:
        std = np.std(closes)

    upper_2sd = vwap + 2 * std
    lower_2sd = vwap - 2 * std
    upper_3sd = vwap + 3 * std
    lower_3sd = vwap - 3 * std
    upper_1sd = vwap + std
    lower_1sd = vwap - std

    return vwap, upper_2sd, lower_2sd, upper_3sd, lower_3sd, upper_1sd, lower_1sd, std


def calc_rsi(closes: list[float], period: int = 2) -> float:
    """Calculate RSI(2) for mean reversion."""
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes[-(period + 1):])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.mean(gains)
    avg_loss = np.mean(losses)

    if avg_loss == 0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_ema(closes: list[float], period: int) -> float:
    """Calculate latest EMA value."""
    if len(closes) < period:
        return 0.0

    multiplier = 2 / (period + 1)
    ema = np.mean(closes[:period])

    for i in range(period, len(closes)):
        ema = (closes[i] - ema) * multiplier + ema

    return ema


def is_market_hours() -> bool:
    """Check if within US market hours (14:30-21:00 UTC)."""
    now = datetime.now(timezone.utc).time()
    return MARKET_START <= now <= MARKET_END


async def check_spy_trend(client: httpx.AsyncClient) -> bool:
    """Check if SPY is above EMA(20) on 15-min - bullish bias."""
    candles = await fetch_candles(client, "SPY", "15min", 30)
    if len(candles) < 20:
        return True  # Default bullish if no data

    candles = candles[::-1]
    closes = [float(c["close"]) for c in candles]
    ema20 = calc_ema(closes, 20)

    return closes[-1] > ema20


async def scan_instrument(client: httpx.AsyncClient, r: redis.Redis, symbol: str, spy_bull: bool):
    """Scan a single equity for VWAP reversion setup."""

    # Fetch 5-min candles (full day = 390 bars)
    candles = await fetch_candles(client, symbol, "5min", 390)
    if len(candles) < 50:
        print(f"[VERTEX] {symbol}: insufficient data")
        return

    candles = candles[::-1]  # Chronological

    # Calculate VWAP and bands
    result = calc_vwap(candles)
    if len(result) < 8:
        print(f"[VERTEX] {symbol}: VWAP calculation failed")
        return

    vwap, upper_2sd, lower_2sd, upper_3sd, lower_3sd, upper_1sd, lower_1sd, std = result

    if vwap == 0:
        print(f"[VERTEX] {symbol}: zero VWAP")
        return

    closes = [float(c["close"]) for c in candles]
    current_price = closes[-1]
    volumes = [float(c.get("volume", 0)) for c in candles]

    # RSI(2)
    rsi2 = calc_rsi(closes, 2)

    # Volume filter: current > 1.5x average
    avg_vol = np.mean(volumes[-20:]) if len(volumes) >= 20 else np.mean(volumes)
    current_vol = volumes[-1]
    vol_spike = current_vol > 1.5 * avg_vol if avg_vol > 0 else False

    # Distance from VWAP
    dist_pct = ((current_price - vwap) / vwap) * 100 if vwap > 0 else 0

    direction = None

    # LONG: at lower 2SD, RSI2 < 30, volume spike, SPY bullish
    if current_price <= lower_2sd and rsi2 < 30 and vol_spike and spy_bull:
        direction = "LONG"
        entry = current_price
        sl = lower_3sd
        tp1 = lower_1sd  # Exit at -1SD (50%)
        tp2 = vwap       # Exit at VWAP (50%)

    # SHORT: at upper 2SD, RSI2 > 70, volume spike
    elif current_price >= upper_2sd and rsi2 > 70 and vol_spike:
        direction = "SHORT"
        entry = current_price
        sl = upper_3sd
        tp1 = upper_1sd
        tp2 = vwap

    else:
        print(f"[VERTEX] {symbol}: no setup (price={current_price:.2f}, vwap={vwap:.2f}, rsi2={rsi2:.0f})")
        return

    # Dedup: 1 per symbol per day
    dedup_key = f"novafx:dedup:floor4:{symbol}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    if await r.exists(dedup_key):
        print(f"[VERTEX] {symbol}: already signaled today")
        return

    rel_vol = (current_vol / avg_vol) if avg_vol > 0 else 1.0

    signal = {
        "floor": "VERTEX",
        "symbol": symbol,
        "direction": direction,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "vwap": round(vwap, 2),
        "dist_pct": round(dist_pct, 2),
        "rsi2": round(rsi2, 1),
        "rel_vol": round(rel_vol, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Publish
    await r.xadd("novafx:signals:floor4", {k: str(v) for k, v in signal.items()})
    await r.set(dedup_key, "1", ex=86400)

    # Telegram
    await send_telegram(signal)
    print(f"[VERTEX] Signal: {symbol} {direction}")


async def send_telegram(signal: dict):
    """Send signal to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    direction_emoji = "📈" if signal["direction"] == "LONG" else "📉"

    msg = f"""📐 VERTEX EQUITIES | {signal["symbol"]} | {direction_emoji} {signal["direction"]} @ {signal["entry"]}

🔴 SL: {signal["sl"]}
✅ TP1: {signal["tp1"]} (50%) | TP2: {signal["tp2"]} (50%)

📊 VWAP: {signal["vwap"]} | Dist: {signal["dist_pct"]}%
📈 RSI(2): {signal["rsi2"]:.0f} | Vol: {signal["rel_vol"]:.1f}x avg
⏰ Exit by 20:45 UTC
📅 {signal["timestamp"][:19]} UTC

────────────────
⚠️ Risk max 1-2% per trade. Not financial advice."""

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})


async def scan_loop(r: redis.Redis):
    """Main scanning loop - every 5 minutes during market hours."""
    while True:
        if not is_market_hours():
            now = datetime.now(timezone.utc)
            print(f"[VERTEX] Outside market hours ({now.time()}). Sleeping 60s...")
            await asyncio.sleep(60)
            continue

        print(f"[VERTEX] Starting scan at {datetime.now(timezone.utc).isoformat()}")

        async with httpx.AsyncClient(timeout=30) as client:
            # Check SPY trend
            spy_bull = await check_spy_trend(client)
            print(f"[VERTEX] SPY bias: {'BULLISH' if spy_bull else 'BEARISH'}")

            for symbol in INSTRUMENTS:
                try:
                    await scan_instrument(client, r, symbol, spy_bull)
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"[VERTEX] Error scanning {symbol}: {e}")

        print(f"[VERTEX] Scan complete. Next scan in {SCAN_INTERVAL_MINUTES}m")
        await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)


async def health_handler(request):
    return web.Response(text="OK", status=200)


async def run_health_server():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"[VERTEX] Health server on port {PORT}")


async def main():
    print("[VERTEX] Floor 4 - VWAP Mean Reversion Scanner starting...")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    await run_health_server()
    await scan_loop(r)


if __name__ == "__main__":
    asyncio.run(main())

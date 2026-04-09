"""
FLOOR 6 - BULLION: Gold/Silver Session Breakout + DXY Bias
Wake at 07:00 UTC daily. Asian session breakout with EMA trend and DXY filter.
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
PORT = int(os.environ.get("PORT", "8006"))

# Instruments
INSTRUMENTS = [
    {"symbol": "XAU/USD", "min_range": 5, "max_range": 30, "name": "Gold"},
    {"symbol": "XAG/USD", "min_range": 0.2, "max_range": 2.0, "name": "Silver"},
]

# Strong months for gold seasonality
STRONG_MONTHS = [1, 2, 8, 9, 10, 11, 12]


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
        print(f"[BULLION] Error fetching {symbol}: {data.get('message', 'unknown')}")
        return []
    return data["values"]


def calc_ema_series(closes: list[float], period: int) -> list[float]:
    """Calculate EMA series."""
    if len(closes) < period:
        return []

    multiplier = 2 / (period + 1)
    ema = [np.mean(closes[:period])]

    for i in range(period, len(closes)):
        ema.append((closes[i] - ema[-1]) * multiplier + ema[-1])

    return ema


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


def get_seasonal_note() -> tuple[str, bool]:
    """Get seasonal sizing note based on month."""
    month = datetime.now(timezone.utc).month
    if month in STRONG_MONTHS:
        return "FULL SIZE", True
    return "REDUCED SIZE (weak season)", False


async def scan_instrument(
    client: httpx.AsyncClient,
    r: redis.Redis,
    instrument: dict,
    gold_bias: str
) -> dict:
    """Scan gold or silver for session breakout. Returns result dict for logging."""
    symbol = instrument["symbol"]
    min_range = instrument["min_range"]
    max_range = instrument["max_range"]
    name = instrument["name"]

    # Fetch 15-min candles (60 bars = 15 hours)
    candles_15m = await fetch_candles(client, symbol, "15min", 60)
    if len(candles_15m) < 28:
        print(f"[BULLION] {symbol}: insufficient 15m data")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "insufficient_15m_data"}

    candles_15m = candles_15m[::-1]  # Chronological

    # Asian range (00:00-07:00 UTC)
    asian_highs = []
    asian_lows = []
    for c in candles_15m:
        dt = datetime.fromisoformat(c["datetime"].replace(" ", "T"))
        if dt.hour < 7:
            asian_highs.append(float(c["high"]))
            asian_lows.append(float(c["low"]))

    if not asian_highs:
        print(f"[BULLION] {symbol}: no Asian session data")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "no_asian_session_data"}

    asian_high = max(asian_highs)
    asian_low = min(asian_lows)
    range_dollars = asian_high - asian_low

    # Range filter
    if range_dollars < min_range or range_dollars > max_range:
        print(f"[BULLION] {symbol}: range ${range_dollars:.2f} outside filter (${min_range}-${max_range})")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": f"range_{range_dollars:.1f}_outside_filter"}

    # Fetch 4H candles for EMA(200) and ATR
    candles_4h = await fetch_candles(client, symbol, "4h", 100)
    if len(candles_4h) < 50:
        print(f"[BULLION] {symbol}: insufficient 4H data")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "insufficient_4h_data"}

    candles_4h = candles_4h[::-1]
    closes_4h = [float(c["close"]) for c in candles_4h]

    ema200_series = calc_ema_series(closes_4h, 200) if len(closes_4h) >= 200 else calc_ema_series(closes_4h, len(closes_4h) - 1)
    if len(ema200_series) < 6:
        print(f"[BULLION] {symbol}: insufficient EMA data")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "insufficient_ema_data"}

    ema200_slope_up = ema200_series[-1] > ema200_series[-5]
    atr = calc_atr(candles_4h, 14)

    current_price = float(candles_15m[-1]["close"])

    # Direction determination
    direction = None

    # LONG: price > asian_high, EMA slope up, not bearish gold
    if current_price > asian_high and ema200_slope_up and gold_bias != "BEARISH_GOLD":
        direction = "LONG"
        entry = current_price
        sl = asian_low - (0.5 * atr)
        tp = entry + (2 * (entry - sl))

    # SHORT: price < asian_low, EMA slope down, not bullish gold
    elif current_price < asian_low and not ema200_slope_up and gold_bias != "BULLISH_GOLD":
        direction = "SHORT"
        entry = current_price
        sl = asian_high + (0.5 * atr)
        tp = entry - (2 * (sl - entry))

    else:
        trend = "UP" if ema200_slope_up else "DOWN"
        print(f"[BULLION] {symbol}: no breakout (price={current_price:.2f}, high={asian_high:.2f}, low={asian_low:.2f}, trend={trend}, bias={gold_bias})")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": f"no_breakout_trend_{trend}_bias_{gold_bias}"}

    # Dedup: 1 per instrument per day
    dedup_key = f"novafx:dedup:floor6:{symbol}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    if await r.exists(dedup_key):
        print(f"[BULLION] {symbol}: already signaled today")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "already_signaled_today"}

    size_note, is_strong = get_seasonal_note()
    rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0

    signal = {
        "floor": "BULLION",
        "symbol": symbol,
        "name": name,
        "direction": direction,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
        "rr": round(rr, 2),
        "range_dollars": round(range_dollars, 2),
        "asian_high": round(asian_high, 2),
        "asian_low": round(asian_low, 2),
        "ema_trend": "UP" if ema200_slope_up else "DOWN",
        "dxy_state": gold_bias,
        "size_note": size_note,
        "atr": round(atr, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Publish
    await r.xadd("novafx:signals:floor6", {k: str(v) for k, v in signal.items()})
    await r.set(dedup_key, "1", ex=86400)

    # Telegram
    await send_telegram(signal)
    print(f"[BULLION] Signal: {symbol} {direction}")

    return {
        "symbol": symbol,
        "signal": direction,
        "score": int(rr * 30 + (30 if is_strong else 15)),
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp": round(tp, 2),
    }


async def send_telegram(signal: dict):
    """Send signal to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    direction_emoji = "📈" if signal["direction"] == "LONG" else "📉"
    dxy_emoji = "🟢" if signal["dxy_state"] == "BULLISH_GOLD" else ("🔴" if signal["dxy_state"] == "BEARISH_GOLD" else "⚪")

    msg = f"""🥇 BULLION COMMAND | {signal["symbol"]} ({signal["name"]}) | {direction_emoji} {signal["direction"]} @ {signal["entry"]}

🔴 SL: {signal["sl"]} | ✅ TP: {signal["tp"]}
⚖️ R:R 1:{signal["rr"]}

📊 Asian Range: ${signal["range_dollars"]:.1f} ({signal["asian_low"]}-{signal["asian_high"]})
📈 EMA200 Trend: {signal["ema_trend"]}
{dxy_emoji} DXY: {signal["dxy_state"]}
💰 {signal["size_note"]}

⏰ Valid until 10:00 UTC
📅 {signal["timestamp"][:19]} UTC

────────────────
⚠️ Risk max 1-2% per trade. Not financial advice."""

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})


async def wait_until_target_time():
    """Sleep until 07:00 UTC."""
    now = datetime.now(timezone.utc)
    target = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)

    wait_seconds = (target - now).total_seconds()
    print(f"[BULLION] Sleeping {wait_seconds:.0f}s until {target.isoformat()}")
    await asyncio.sleep(wait_seconds)


async def write_scan_log(r: redis.Redis, results: list[dict], signals_fired: int, scan_time: datetime, gold_bias: str):
    """Write scan summary to Redis for status-api."""
    vix_regime = await r.get("novafx:cross:vix_regime") or "UNKNOWN"

    # Next scan is tomorrow at 07:00 UTC
    next_scan = (scan_time + timedelta(days=1)).replace(hour=7, minute=0, second=0, microsecond=0)

    summary = {
        "last_scan": scan_time.isoformat() + "Z",
        "next_scan": next_scan.isoformat() + "Z",
        "results": results,
        "signals_fired": signals_fired,
        "dxy_state": gold_bias,
        "vix_regime": vix_regime,
    }

    await r.set("novafx:log:floor6", json.dumps(summary))
    print(f"[BULLION] Wrote scan log: {signals_fired} signals, {len(results)} instruments scanned")


async def scan_loop(r: redis.Redis):
    """Main loop - wake at 07:00 UTC daily."""
    while True:
        await wait_until_target_time()
        scan_time = datetime.now(timezone.utc)
        print(f"[BULLION] Starting scan at {scan_time.isoformat()}")

        # Get DXY state from cross-floor intel
        gold_bias = await r.get("novafx:cross:dxy_state")
        if gold_bias is None:
            gold_bias = "NEUTRAL"
        print(f"[BULLION] DXY bias: {gold_bias}")

        results = []
        signals_fired = 0

        async with httpx.AsyncClient(timeout=30) as client:
            for instrument in INSTRUMENTS:
                try:
                    result = await scan_instrument(client, r, instrument, gold_bias)
                    if result:
                        results.append(result)
                        if result.get("signal") not in [None, "NO_SIGNAL"]:
                            signals_fired += 1
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"[BULLION] Error scanning {instrument['symbol']}: {e}")
                    results.append({"symbol": instrument["symbol"], "signal": "NO_SIGNAL", "reason": f"error_{str(e)[:50]}"})

        # Write scan summary to Redis
        await write_scan_log(r, results, signals_fired, scan_time, gold_bias)


async def health_handler(request):
    return web.Response(text="OK", status=200)


async def run_health_server():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"[BULLION] Health server on port {PORT}")


async def main():
    print("[BULLION] Floor 6 - Gold/Silver Session Breakout starting...")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    await run_health_server()
    await scan_loop(r)


if __name__ == "__main__":
    asyncio.run(main())

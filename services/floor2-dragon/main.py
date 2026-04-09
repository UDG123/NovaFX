"""
FLOOR 2 - DRAGON: RSI Divergence + Carry Trade on JPY Crosses
Scan every 4 hours. Detect bullish/bearish divergence with carry rate filter.
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
PORT = int(os.environ.get("PORT", "8002"))

# Instruments (TwelveData format with slash)
INSTRUMENTS = ["GBP/JPY", "EUR/JPY", "AUD/JPY", "NZD/JPY", "CAD/JPY", "GBP/AUD", "EUR/GBP"]

# Carry rates (central bank rates as of 2024)
CARRY_RATES = {
    "GBP": 4.50,
    "EUR": 2.65,
    "AUD": 4.10,
    "NZD": 5.25,
    "CAD": 3.00,
    "JPY": 0.50,
}

SCAN_INTERVAL_HOURS = 4


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
        print(f"[DRAGON] Error fetching {symbol} {interval}: {data.get('message', 'unknown')}")
        return []
    return data["values"]


def calc_rsi(closes: list[float], period: int = 14) -> list[float]:
    """Calculate RSI series."""
    if len(closes) < period + 1:
        return []

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsi_values = []
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    return rsi_values


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


def find_swing_lows(prices: list[float], window: int = 3) -> list[tuple[int, float]]:
    """Find swing low indices and values."""
    swings = []
    for i in range(window, len(prices) - window):
        if all(prices[i] <= prices[i - j] for j in range(1, window + 1)) and \
           all(prices[i] <= prices[i + j] for j in range(1, window + 1)):
            swings.append((i, prices[i]))
    return swings


def find_swing_highs(prices: list[float], window: int = 3) -> list[tuple[int, float]]:
    """Find swing high indices and values."""
    swings = []
    for i in range(window, len(prices) - window):
        if all(prices[i] >= prices[i - j] for j in range(1, window + 1)) and \
           all(prices[i] >= prices[i + j] for j in range(1, window + 1)):
            swings.append((i, prices[i]))
    return swings


def detect_bullish_divergence(prices: list[float], rsi: list[float], lookback: int = 20) -> tuple[bool, float, float]:
    """Detect bullish divergence: lower price low but higher RSI low."""
    if len(prices) < lookback or len(rsi) < lookback:
        return False, 0, 0

    # Get last N bars
    recent_prices = prices[-lookback:]
    recent_rsi = rsi[-lookback:]

    price_lows = find_swing_lows(recent_prices, 2)
    if len(price_lows) < 2:
        return False, 0, 0

    # Check last two swing lows
    idx1, price1 = price_lows[-2]
    idx2, price2 = price_lows[-1]

    rsi1 = recent_rsi[idx1] if idx1 < len(recent_rsi) else 0
    rsi2 = recent_rsi[idx2] if idx2 < len(recent_rsi) else 0

    # Bullish: price lower low, RSI higher low, first RSI < 35
    if price2 < price1 and rsi2 > rsi1 and rsi1 < 35:
        return True, rsi1, rsi2

    return False, 0, 0


def detect_bearish_divergence(prices: list[float], rsi: list[float], lookback: int = 20) -> tuple[bool, float, float]:
    """Detect bearish divergence: higher price high but lower RSI high."""
    if len(prices) < lookback or len(rsi) < lookback:
        return False, 0, 0

    recent_prices = prices[-lookback:]
    recent_rsi = rsi[-lookback:]

    price_highs = find_swing_highs(recent_prices, 2)
    if len(price_highs) < 2:
        return False, 0, 0

    idx1, price1 = price_highs[-2]
    idx2, price2 = price_highs[-1]

    rsi1 = recent_rsi[idx1] if idx1 < len(recent_rsi) else 0
    rsi2 = recent_rsi[idx2] if idx2 < len(recent_rsi) else 0

    # Bearish: price higher high, RSI lower high, first RSI > 65
    if price2 > price1 and rsi2 < rsi1 and rsi1 > 65:
        return True, rsi1, rsi2

    return False, 0, 0


def get_carry_diff(symbol: str) -> float:
    """Calculate carry rate differential for a pair."""
    base = symbol[:3]
    quote = symbol[4:7]
    base_rate = CARRY_RATES.get(base, 0)
    quote_rate = CARRY_RATES.get(quote, 0)
    return base_rate - quote_rate


def is_jpy_pair(symbol: str) -> bool:
    return "JPY" in symbol


async def scan_instrument(client: httpx.AsyncClient, r: redis.Redis, symbol: str) -> dict:
    """Scan a single instrument for RSI divergence. Returns result dict for logging."""

    # Fetch 100 daily candles
    candles = await fetch_candles(client, symbol, "1day", 100)
    if len(candles) < 50:
        print(f"[DRAGON] {symbol}: insufficient data")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "insufficient_data"}

    candles = candles[::-1]  # Chronological order
    closes = [float(c["close"]) for c in candles]

    # Calculate RSI(14)
    rsi = calc_rsi(closes, 14)
    if len(rsi) < 20:
        print(f"[DRAGON] {symbol}: insufficient RSI data")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "insufficient_rsi_data"}

    # Calculate ATR(14)
    atr = calc_atr(candles, 14)

    # Carry rate
    carry_diff = get_carry_diff(symbol)

    # Detect divergences
    bullish, rsi1_b, rsi2_b = detect_bullish_divergence(closes, rsi)
    bearish, rsi1_s, rsi2_s = detect_bearish_divergence(closes, rsi)

    direction = None
    div_type = None
    rsi_pivot1 = 0
    rsi_pivot2 = 0

    if bullish:
        # Carry filter for JPY pairs
        if is_jpy_pair(symbol) and carry_diff < 1.0:
            print(f"[DRAGON] {symbol}: bullish blocked by carry filter ({carry_diff:.2f}%)")
            return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": f"bullish_blocked_carry_{carry_diff:.2f}"}
        direction = "LONG"
        div_type = "Bullish"
        rsi_pivot1 = rsi1_b
        rsi_pivot2 = rsi2_b
    elif bearish:
        if is_jpy_pair(symbol) and carry_diff > -1.0:
            print(f"[DRAGON] {symbol}: bearish blocked by carry filter ({carry_diff:.2f}%)")
            return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": f"bearish_blocked_carry_{carry_diff:.2f}"}
        direction = "SHORT"
        div_type = "Bearish"
        rsi_pivot1 = rsi1_s
        rsi_pivot2 = rsi2_s
    else:
        print(f"[DRAGON] {symbol}: no divergence detected")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "no_divergence_detected"}

    # Dedup check (1 week)
    week_num = datetime.now(timezone.utc).strftime("%Y-W%W")
    dedup_key = f"novafx:dedup:floor2:{symbol}:{week_num}"
    if await r.exists(dedup_key):
        print(f"[DRAGON] {symbol}: already signaled this week")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "already_signaled_this_week"}

    # Calculate SL/TP
    current_price = closes[-1]
    sl_multiplier = 2.5 if symbol == "GBP/JPY" else 2.0
    sl_distance = atr * sl_multiplier
    tp_distance = sl_distance * 1.5

    if direction == "LONG":
        entry = current_price
        sl = entry - sl_distance
        tp = entry + tp_distance
    else:
        entry = current_price
        sl = entry + sl_distance
        tp = entry - tp_distance

    rr = tp_distance / sl_distance if sl_distance > 0 else 0

    signal = {
        "floor": "DRAGON",
        "symbol": symbol,
        "direction": direction,
        "div_type": div_type,
        "entry": round(entry, 5),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "rr": round(rr, 2),
        "carry_diff": round(carry_diff, 2),
        "rsi_pivot1": round(rsi_pivot1, 1),
        "rsi_pivot2": round(rsi_pivot2, 1),
        "atr": round(atr, 5),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Publish to Redis
    await r.xadd("novafx:signals:floor2", {k: str(v) for k, v in signal.items()})
    await r.set(dedup_key, "1", ex=168 * 3600)  # 1 week

    # Send Telegram
    await send_telegram(signal)
    print(f"[DRAGON] Signal: {symbol} {direction} ({div_type} divergence)")

    return {
        "symbol": symbol,
        "signal": direction,
        "score": int(abs(rsi_pivot2 - rsi_pivot1) + abs(carry_diff) * 10),
        "entry": round(entry, 5),
        "sl": round(sl, 5),
        "tp": round(tp, 5),
    }


async def send_telegram(signal: dict):
    """Send signal to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    direction_emoji = "📈" if signal["direction"] == "LONG" else "📉"
    carry_note = f"+{signal['carry_diff']}%" if signal["carry_diff"] > 0 else f"{signal['carry_diff']}%"

    msg = f"""🐉 DRAGON CROSS | {signal["symbol"]} | {direction_emoji} {signal["direction"]} @ {signal["entry"]}

🔴 SL: {signal["sl"]} | ✅ TP: {signal["tp"]}
⚖️ R:R 1:{signal["rr"]}

🧠 {signal["div_type"]} Divergence | RSI: {signal["rsi_pivot1"]:.0f}→{signal["rsi_pivot2"]:.0f}
💰 Carry: {carry_note}
📅 {signal["timestamp"][:19]} UTC

────────────────
⚠️ Risk max 1-2% per trade. Not financial advice."""

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})


async def write_scan_log(r: redis.Redis, results: list[dict], signals_fired: int, scan_time: datetime):
    """Write scan summary to Redis for status-api."""
    dxy_state = await r.get("novafx:cross:dxy_state") or "UNKNOWN"
    vix_regime = await r.get("novafx:cross:vix_regime") or "UNKNOWN"

    next_scan = scan_time + timedelta(hours=SCAN_INTERVAL_HOURS)

    summary = {
        "last_scan": scan_time.isoformat() + "Z",
        "next_scan": next_scan.isoformat() + "Z",
        "results": results,
        "signals_fired": signals_fired,
        "dxy_state": dxy_state,
        "vix_regime": vix_regime,
    }

    await r.set("novafx:log:floor2", json.dumps(summary))
    print(f"[DRAGON] Wrote scan log: {signals_fired} signals, {len(results)} instruments scanned")


async def scan_loop(r: redis.Redis):
    """Main scanning loop - every 4 hours."""
    while True:
        scan_time = datetime.now(timezone.utc)
        print(f"[DRAGON] Starting scan at {scan_time.isoformat()}")

        results = []
        signals_fired = 0

        async with httpx.AsyncClient(timeout=30) as client:
            for symbol in INSTRUMENTS:
                try:
                    result = await scan_instrument(client, r, symbol)
                    if result:
                        results.append(result)
                        if result.get("signal") not in [None, "NO_SIGNAL"]:
                            signals_fired += 1
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"[DRAGON] Error scanning {symbol}: {e}")
                    results.append({"symbol": symbol, "signal": "NO_SIGNAL", "reason": f"error_{str(e)[:50]}"})

        # Write scan summary to Redis
        await write_scan_log(r, results, signals_fired, scan_time)

        print(f"[DRAGON] Scan complete. Next scan in {SCAN_INTERVAL_HOURS}h")
        await asyncio.sleep(SCAN_INTERVAL_HOURS * 3600)


async def health_handler(request):
    return web.Response(text="OK", status=200)


async def run_health_server():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"[DRAGON] Health server on port {PORT}")


async def main():
    print("[DRAGON] Floor 2 - RSI Divergence + Carry Scanner starting...")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    await run_health_server()
    await scan_loop(r)


if __name__ == "__main__":
    asyncio.run(main())

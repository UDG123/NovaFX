"""
FLOOR 3 - CIPHER: Crypto EMA/MACD Confluence + Fear & Greed Filter
Scan every 4 hours. Multi-indicator confluence with sentiment overlay.
"""
import asyncio
import os
import json
from datetime import datetime, timezone
import httpx
import redis.asyncio as redis
from aiohttp import web
import numpy as np

# Config
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")
PORT = int(os.environ.get("PORT", "8003"))

# Instruments (TwelveData format)
INSTRUMENTS = ["BTC/USD", "ETH/USD", "SOL/USD", "BNB/USD", "XRP/USD"]

SCAN_INTERVAL_HOURS = 4


async def fetch_candles(client: httpx.AsyncClient, symbol: str, interval: str, outputsize: int) -> list[dict]:
    """Fetch OHLCV candles from TwelveData (oldest-first for correct EMA calculation)."""
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": interval,
        "outputsize": outputsize,
        "order": "ASC",  # Oldest-first for correct EMA calculation
        "apikey": TWELVEDATA_API_KEY,
    }
    resp = await client.get(url, params=params)
    data = resp.json()
    if "values" not in data:
        print(f"[CIPHER] Error fetching {symbol}: {data.get('message', 'unknown')}")
        return []
    candles = data["values"]
    if candles:
        print(f"[CIPHER] {symbol}: fetched {len(candles)} candles, range {candles[0]['datetime']} to {candles[-1]['datetime']}")
    return candles


async def fetch_fear_greed(client: httpx.AsyncClient) -> int:
    """Fetch Fear & Greed index from alternative.me."""
    try:
        resp = await client.get("https://api.alternative.me/fng/?limit=1")
        data = resp.json()
        return int(data["data"][0]["value"])
    except Exception as e:
        print(f"[CIPHER] F&G fetch error: {e}")
        return 50  # Neutral default


async def fetch_funding_rate(client: httpx.AsyncClient, symbol: str) -> float:
    """Fetch Binance funding rate for context."""
    try:
        binance_symbol = symbol.replace("/USD", "USDT").replace("/", "")
        resp = await client.get(
            f"https://fapi.binance.com/fapi/v1/fundingRate",
            params={"symbol": binance_symbol, "limit": 1}
        )
        data = resp.json()
        if data:
            return float(data[0]["fundingRate"]) * 100  # Convert to percentage
    except Exception as e:
        print(f"[CIPHER] Funding rate fetch error: {e}")
    return 0.0


def calc_ema(closes: list[float], period: int) -> list[float]:
    """Calculate EMA series."""
    if len(closes) < period:
        return []

    multiplier = 2 / (period + 1)
    ema = [np.mean(closes[:period])]

    for i in range(period, len(closes)):
        ema.append((closes[i] - ema[-1]) * multiplier + ema[-1])

    return ema


def calc_macd(closes: list[float]) -> tuple[list[float], list[float], list[float]]:
    """Calculate MACD(12,26,9) components."""
    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)

    if len(ema12) < len(ema26):
        return [], [], []

    # Align EMAs
    offset = len(ema12) - len(ema26)
    macd_line = [ema12[offset + i] - ema26[i] for i in range(len(ema26))]

    signal_line = calc_ema(macd_line, 9)

    if not signal_line:
        return [], [], []

    # Align for histogram
    offset2 = len(macd_line) - len(signal_line)
    histogram = [macd_line[offset2 + i] - signal_line[i] for i in range(len(signal_line))]

    return macd_line[-len(signal_line):], signal_line, histogram


def calc_rsi(closes: list[float], period: int = 14) -> float:
    """Calculate latest RSI value."""
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


async def scan_instrument(client: httpx.AsyncClient, r: redis.Redis, symbol: str, fear_greed: int):
    """Scan a single crypto instrument."""

    # Fetch 4H candles (250 bars for EMA200, already ordered ASC)
    candles = await fetch_candles(client, symbol, "4h", 250)
    if len(candles) < 200:
        print(f"[CIPHER] {symbol}: insufficient data ({len(candles)} candles, need 200+)")
        return

    # Already in chronological order (ASC)
    closes = [float(c["close"]) for c in candles]
    current_price = closes[-1]

    # Calculate indicators
    ema8 = calc_ema(closes, 8)
    ema34 = calc_ema(closes, 34)
    ema200 = calc_ema(closes, 200)

    if not ema8 or not ema34 or not ema200:
        print(f"[CIPHER] {symbol}: EMA calculation failed")
        return

    macd_line, signal_line, _ = calc_macd(closes)
    if not macd_line or not signal_line:
        print(f"[CIPHER] {symbol}: MACD calculation failed")
        return

    rsi = calc_rsi(closes, 14)
    atr = calc_atr(candles, 14)

    # Get latest values
    ema8_val = ema8[-1]
    ema34_val = ema34[-1]
    ema200_val = ema200[-1]
    macd_val = macd_line[-1]
    signal_val = signal_line[-1]

    # Determine direction
    direction = None

    # LONG conditions
    if (ema8_val > ema34_val and
        macd_val > signal_val and
        45 < rsi < 70 and
        current_price > ema200_val):

        # F&G filter: block longs if extreme greed
        if fear_greed > 80:
            print(f"[CIPHER] {symbol}: LONG blocked by F&G={fear_greed}")
            return
        direction = "LONG"

    # SHORT conditions
    elif (ema8_val < ema34_val and
          macd_val < signal_val and
          30 < rsi < 55 and
          current_price < ema200_val):

        # F&G filter: block shorts if extreme fear
        if fear_greed < 20:
            print(f"[CIPHER] {symbol}: SHORT blocked by F&G={fear_greed}")
            return
        direction = "SHORT"

    else:
        direction = None

    # Always log computed values for debugging
    print(f"[CIPHER] {symbol}: EMA8={ema8_val:.4f} EMA34={ema34_val:.4f} RSI={rsi:.1f} -> {direction or 'NO_SIGNAL'}")

    if direction is None:
        return

    # Dedup (8hr)
    hour_block = datetime.now(timezone.utc).hour // 8
    dedup_key = f"novafx:dedup:floor3:{symbol}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}:{hour_block}"
    if await r.exists(dedup_key):
        print(f"[CIPHER] {symbol}: already signaled this 8hr block")
        return

    # Get funding rate for context
    funding_rate = await fetch_funding_rate(client, symbol)

    # Calculate SL/TP
    if direction == "LONG":
        entry = current_price
        sl = entry - (1.5 * atr)
        tp1 = entry + (2 * atr)
        tp2 = entry + (3 * atr)
    else:
        entry = current_price
        sl = entry + (1.5 * atr)
        tp1 = entry - (2 * atr)
        tp2 = entry - (3 * atr)

    rr = (2 * atr) / (1.5 * atr) if atr > 0 else 0

    signal = {
        "floor": "CIPHER",
        "symbol": symbol,
        "direction": direction,
        "entry": round(entry, 2),
        "sl": round(sl, 2),
        "tp1": round(tp1, 2),
        "tp2": round(tp2, 2),
        "rr": round(rr, 2),
        "fear_greed": fear_greed,
        "funding_rate": round(funding_rate, 4),
        "rsi": round(rsi, 1),
        "ema8": round(ema8_val, 2),
        "ema34": round(ema34_val, 2),
        "ema200": round(ema200_val, 2),
        "atr": round(atr, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Publish to Redis
    await r.xadd("novafx:signals:floor3", {k: str(v) for k, v in signal.items()})
    await r.set(dedup_key, "1", ex=8 * 3600)

    # Send Telegram
    await send_telegram(signal)
    print(f"[CIPHER] Signal: {symbol} {direction}")


async def send_telegram(signal: dict):
    """Send signal to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    direction_emoji = "📈" if signal["direction"] == "LONG" else "📉"
    ema_note = "EMA8>34>200" if signal["direction"] == "LONG" else "EMA8<34<200"

    msg = f"""🔐 CIPHER CRYPTO | {signal["symbol"]} | {direction_emoji} {signal["direction"]} @ {signal["entry"]}

🔴 SL: {signal["sl"]}
✅ TP1: {signal["tp1"]} | TP2: {signal["tp2"]}
⚖️ R:R 1:{signal["rr"]}

🧠 {ema_note} | RSI: {signal["rsi"]:.0f}
😱 F&G: {signal["fear_greed"]} | 💰 Funding: {signal["funding_rate"]:.3f}%
📅 {signal["timestamp"][:19]} UTC

────────────────
⚠️ Risk max 1-2% per trade. Not financial advice."""

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})


async def scan_loop(r: redis.Redis):
    """Main scanning loop - every 4 hours."""
    while True:
        print(f"[CIPHER] Starting scan at {datetime.now(timezone.utc).isoformat()}")

        async with httpx.AsyncClient(timeout=30) as client:
            # Fetch F&G once for all instruments
            fear_greed = await fetch_fear_greed(client)
            print(f"[CIPHER] Fear & Greed Index: {fear_greed}")

            for symbol in INSTRUMENTS:
                try:
                    await scan_instrument(client, r, symbol, fear_greed)
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"[CIPHER] Error scanning {symbol}: {e}")

        print(f"[CIPHER] Scan complete. Next scan in {SCAN_INTERVAL_HOURS}h")
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
    print(f"[CIPHER] Health server on port {PORT}")


async def main():
    print("[CIPHER] Floor 3 - Crypto EMA/MACD Scanner starting...")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    await run_health_server()
    await scan_loop(r)


if __name__ == "__main__":
    asyncio.run(main())

"""
CROSS-FLOOR INTEL: Macro Signal Bus
Run every 15 minutes. Publish DXY state and VIX regime for all floors.
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
TELEGRAM_SYSTEM_CHAT_ID = os.environ.get("TELEGRAM_SYSTEM_CHAT_ID", "-1003710749613")
PORT = int(os.environ.get("PORT", "8007"))

SCAN_INTERVAL_MINUTES = 15


async def send_system_alert(message: str):
    """Send alert to system Telegram channel."""
    if not TELEGRAM_BOT_TOKEN:
        print(f"[INTEL] System alert (no token): {message}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_SYSTEM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                print(f"[INTEL] System alert sent: {message[:50]}...")
            else:
                print(f"[INTEL] System alert failed: {resp.status_code}")
    except Exception as e:
        print(f"[INTEL] System alert error: {e}")


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
        print(f"[INTEL] Error fetching {symbol}: {data.get('message', 'unknown')}")
        return []
    return data["values"]


def calc_ema(closes: list[float], period: int) -> float:
    if len(closes) < period:
        return 0.0
    multiplier = 2 / (period + 1)
    ema = np.mean(closes[:period])
    for i in range(period, len(closes)):
        ema = (closes[i] - ema) * multiplier + ema
    return ema


def calc_atr(candles: list[dict], period: int = 10) -> float:
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


async def update_dxy_state(client: httpx.AsyncClient, r: redis.Redis) -> str:
    """Update DXY state using EUR/USD as inverse proxy (DX-Y.NYB unavailable on free tier)."""
    candles = await fetch_candles(client, "EUR/USD", "1day", 30)

    if not candles or len(candles) < 20:
        print("[INTEL] Insufficient EUR/USD data for DXY state")
        return "NEUTRAL"

    candles = candles[::-1]  # Chronological
    closes = [float(c["close"]) for c in candles]
    eurusd_close = closes[-1]
    ema20 = calc_ema(closes, 20)

    # EUR/USD is inverse to DXY: EUR/USD up = DXY down = BULLISH_GOLD
    if eurusd_close > ema20:
        state = "BULLISH_GOLD"
    elif eurusd_close < ema20:
        state = "BEARISH_GOLD"
    else:
        state = "NEUTRAL"

    await r.set("novafx:cross:dxy_state", state, ex=1800)
    print(f"[INTEL] EUR/USD proxy: {eurusd_close:.5f} vs EMA20={ema20:.5f} -> {state}")
    return state


async def update_vix_regime(client: httpx.AsyncClient, r: redis.Redis) -> str:
    """Update VIX regime based on SPY ATR%."""
    candles = await fetch_candles(client, "SPY", "1h", 50)

    if not candles or len(candles) < 11:
        print("[INTEL] Insufficient SPY data")
        return "ELEVATED"

    candles = candles[::-1]  # Chronological
    spy_close = float(candles[-1]["close"])
    atr = calc_atr(candles, 10)
    atr_pct = (atr / spy_close) * 100 if spy_close > 0 else 1.5

    if atr_pct < 1.0:
        regime = "LOW_VOL"
    elif atr_pct > 2.0:
        regime = "HIGH_VOL"
    else:
        regime = "ELEVATED"

    await r.set("novafx:cross:vix_regime", regime, ex=1800)
    print(f"[INTEL] SPY ATR%: {atr_pct:.2f}% -> {regime}")

    # Publish risk-off alert if HIGH_VOL
    if regime == "HIGH_VOL":
        await r.publish("novafx:cross:risk_off", "HIGH_VOL")
        print("[INTEL] Published HIGH_VOL risk-off alert")

    return regime


async def scan_loop(r: redis.Redis):
    """Main loop - run every 15 minutes."""
    while True:
        print(f"[INTEL] Updating cross-floor intel at {datetime.now(timezone.utc).isoformat()}")

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                dxy_state = await update_dxy_state(client, r)
                await asyncio.sleep(2)
                vix_regime = await update_vix_regime(client, r)

                print(f"[INTEL] Published: DXY={dxy_state}, VIX={vix_regime}")

            except Exception as e:
                print(f"[INTEL] Error: {e}")
                timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                await send_system_alert(f"⚠️ NovaFX Alert | cross-floor-intel error: {e} | {timestamp}")

        print(f"[INTEL] Next update in {SCAN_INTERVAL_MINUTES}m")
        await asyncio.sleep(SCAN_INTERVAL_MINUTES * 60)


async def health_handler(request):
    return web.Response(text="OK", status=200)


async def status_handler(request):
    """Return current intel state."""
    r = redis.from_url(REDIS_URL, decode_responses=True)
    dxy = await r.get("novafx:cross:dxy_state") or "UNKNOWN"
    vix = await r.get("novafx:cross:vix_regime") or "UNKNOWN"
    await r.close()

    return web.json_response({
        "dxy_state": dxy,
        "vix_regime": vix,
        "updated": datetime.now(timezone.utc).isoformat(),
    })


async def run_health_server():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/status", status_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"[INTEL] Health server on port {PORT}")


async def main():
    print("[INTEL] Cross-Floor Intel Service starting...")
    print("[INTEL] DXY via EUR/USD inverse proxy (EUR/USD above 20-EMA = weak USD = BULLISH_GOLD)")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    await run_health_server()

    # Send startup alert with initial state
    dxy_state = await r.get("novafx:cross:dxy_state") or "UNKNOWN"
    vix_regime = await r.get("novafx:cross:vix_regime") or "UNKNOWN"
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    await send_system_alert(f"✅ NovaFX System Online | DXY={dxy_state} | VIX={vix_regime} | {timestamp}")

    await scan_loop(r)


if __name__ == "__main__":
    asyncio.run(main())

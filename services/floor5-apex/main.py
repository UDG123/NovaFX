"""
FLOOR 5 - APEX: Triple RSI on Indices + Cross-Floor Intel Publisher
Scan at 21:00 UTC daily. Multi-tier RSI signals + DXY/VIX regime publishing.
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
PORT = int(os.environ.get("PORT", "8005"))

# Instruments
INSTRUMENTS = ["SPY", "QQQ", "DIA"]


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
        print(f"[APEX] Error fetching {symbol}: {data.get('message', 'unknown')}")
        return []
    return data["values"]


def calc_sma(closes: list[float], period: int) -> float:
    if len(closes) < period:
        return 0.0
    return np.mean(closes[-period:])


def calc_rsi_series(closes: list[float], period: int = 2) -> list[float]:
    """Calculate RSI series for multi-bar analysis."""
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


async def scan_instrument(client: httpx.AsyncClient, r: redis.Redis, symbol: str) -> dict:
    """Scan a single index for Triple RSI setup. Returns result dict for logging."""

    # Fetch 250 daily candles
    candles = await fetch_candles(client, symbol, "1day", 250)
    if len(candles) < 200:
        print(f"[APEX] {symbol}: insufficient data")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "insufficient_data"}

    candles = candles[::-1]  # Chronological
    closes = [float(c["close"]) for c in candles]
    current_close = closes[-1]

    # SMA(200)
    sma200 = calc_sma(closes, 200)

    # RSI(2) series
    rsi_series = calc_rsi_series(closes, 2)
    if len(rsi_series) < 3:
        print(f"[APEX] {symbol}: insufficient RSI data")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "insufficient_rsi_data"}

    rsi2 = rsi_series[-1]
    rsi_prev = rsi_series[-2]
    rsi_prev2 = rsi_series[-3]

    # Triple patterns
    triple_declining = rsi2 < rsi_prev < rsi_prev2
    triple_rising = rsi2 > rsi_prev > rsi_prev2

    cumulative_2bar = rsi2 + rsi_prev

    direction = None
    tier = None
    win_rate = None

    # LONG tiers
    if current_close > sma200:
        if triple_declining and rsi2 < 10:
            direction = "LONG"
            tier = "TRIPLE_RSI"
            win_rate = "90%"
        elif cumulative_2bar < 5:
            direction = "LONG"
            tier = "CUMULATIVE_RSI"
            win_rate = "83%"
        elif rsi2 < 10:
            direction = "LONG"
            tier = "BASIC_RSI"
            win_rate = "75%"

    # SHORT tiers (mirror)
    elif current_close < sma200:
        triple_rising = rsi2 > rsi_prev > rsi_prev2
        if triple_rising and rsi2 > 90:
            direction = "SHORT"
            tier = "TRIPLE_RSI"
            win_rate = "90%"
        elif (rsi2 + rsi_prev) > 195:
            direction = "SHORT"
            tier = "CUMULATIVE_RSI"
            win_rate = "83%"
        elif rsi2 > 90:
            direction = "SHORT"
            tier = "BASIC_RSI"
            win_rate = "75%"

    if not direction:
        print(f"[APEX] {symbol}: no signal (RSI2={rsi2:.1f}, SMA200={sma200:.2f})")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": f"no_signal_rsi2_{rsi2:.1f}"}

    # Dedup: 1 per day
    dedup_key = f"novafx:dedup:floor5:{symbol}:{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    if await r.exists(dedup_key):
        print(f"[APEX] {symbol}: already signaled today")
        return {"symbol": symbol, "signal": "NO_SIGNAL", "reason": "already_signaled_today"}

    # Get yesterday's high for exit note
    prev_high = float(candles[-2]["high"])

    signal = {
        "floor": "APEX",
        "symbol": symbol,
        "direction": direction,
        "tier": tier,
        "win_rate": win_rate,
        "entry": round(current_close, 2),
        "sma200": round(sma200, 2),
        "rsi2": round(rsi2, 1),
        "rsi_prev": round(rsi_prev, 1),
        "exit_above": round(prev_high, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Publish
    await r.xadd("novafx:signals:floor5", {k: str(v) for k, v in signal.items()})
    await r.set(dedup_key, "1", ex=86400)

    # Telegram
    await send_telegram(signal)
    print(f"[APEX] Signal: {symbol} {direction} ({tier})")

    return {
        "symbol": symbol,
        "signal": direction,
        "score": 90 if tier == "TRIPLE_RSI" else (83 if tier == "CUMULATIVE_RSI" else 75),
        "entry": round(current_close, 2),
        "sl": round(sma200, 2),
        "tp": round(prev_high, 2),
    }


async def publish_cross_floor_intel(client: httpx.AsyncClient, r: redis.Redis):
    """Publish DXY state and VIX regime for other floors to consume."""

    # DXY analysis
    dxy_candles = await fetch_candles(client, "DXY", "1day", 100)
    dxy_state = "NEUTRAL"

    if dxy_candles:
        dxy_candles = dxy_candles[::-1]
        dxy_closes = [float(c["close"]) for c in dxy_candles]
        dxy_ema20 = calc_ema(dxy_closes, 20)
        dxy_close = dxy_closes[-1]

        if dxy_close < dxy_ema20 * 0.998:
            dxy_state = "BULLISH_GOLD"
        elif dxy_close > dxy_ema20 * 1.002:
            dxy_state = "BEARISH_GOLD"

    await r.set("novafx:cross:dxy_state", dxy_state, ex=86400)
    print(f"[APEX] Published DXY state: {dxy_state}")

    # SPY ATR% for VIX regime
    spy_candles = await fetch_candles(client, "SPY", "1day", 20)
    vix_regime = "ELEVATED"

    if spy_candles:
        spy_candles = spy_candles[::-1]
        atr = calc_atr(spy_candles, 10)
        spy_close = float(spy_candles[-1]["close"])
        atr_pct = (atr / spy_close) * 100 if spy_close > 0 else 1.5

        if atr_pct < 1.0:
            vix_regime = "LOW_VOL"
        elif atr_pct > 2.0:
            vix_regime = "HIGH_VOL"
            # Publish risk-off alert
            await r.publish("novafx:cross:risk_off", "HIGH_VOL")
            print("[APEX] Published HIGH_VOL risk-off alert")
        else:
            vix_regime = "ELEVATED"

    await r.set("novafx:cross:vix_regime", vix_regime, ex=86400)
    print(f"[APEX] Published VIX regime: {vix_regime}")

    # Send intel summary to Telegram
    await send_intel_telegram(dxy_state, vix_regime)

    return dxy_state, vix_regime


async def send_telegram(signal: dict):
    """Send signal to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    direction_emoji = "📈" if signal["direction"] == "LONG" else "📉"

    msg = f"""🏔️ APEX MACRO | {signal["symbol"]} | {direction_emoji} {signal["direction"]}

📊 Tier: {signal["tier"]} | WR: {signal["win_rate"]}
💰 Entry: {signal["entry"]} | SMA200: {signal["sma200"]}
📈 RSI(2): {signal["rsi2"]:.0f} (prev: {signal["rsi_prev"]:.0f})

🎯 Exit: Close > {signal["exit_above"]} (prev high)
📅 {signal["timestamp"][:19]} UTC

────────────────
⚠️ Risk max 1-2% per trade. Not financial advice."""

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})


async def send_intel_telegram(dxy_state: str, vix_regime: str):
    """Send daily cross-floor intel to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return

    dxy_emoji = "🟢" if dxy_state == "BULLISH_GOLD" else ("🔴" if dxy_state == "BEARISH_GOLD" else "⚪")
    vol_emoji = "🟢" if vix_regime == "LOW_VOL" else ("🔴" if vix_regime == "HIGH_VOL" else "🟡")

    msg = f"""📡 Cross-Floor Intel | {datetime.now(timezone.utc).strftime('%Y-%m-%d')}

{dxy_emoji} DXY Bias: {dxy_state}
{vol_emoji} SPY Regime: {vix_regime}

Use for Floor 2 (Dragon), Floor 6 (Bullion) filters."""

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient() as client:
        await client.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": msg})


async def wait_until_target_time():
    """Sleep until 21:00 UTC."""
    now = datetime.now(timezone.utc)
    target = now.replace(hour=21, minute=0, second=0, microsecond=0)
    if now >= target:
        target += timedelta(days=1)

    wait_seconds = (target - now).total_seconds()
    print(f"[APEX] Sleeping {wait_seconds:.0f}s until {target.isoformat()}")
    await asyncio.sleep(wait_seconds)


async def write_scan_log(r: redis.Redis, results: list[dict], signals_fired: int, scan_time: datetime):
    """Write scan summary to Redis for status-api."""
    dxy_state = await r.get("novafx:cross:dxy_state") or "UNKNOWN"
    vix_regime = await r.get("novafx:cross:vix_regime") or "UNKNOWN"

    # Next scan is tomorrow at 21:00 UTC
    next_scan = (scan_time + timedelta(days=1)).replace(hour=21, minute=0, second=0, microsecond=0)

    summary = {
        "last_scan": scan_time.isoformat() + "Z",
        "next_scan": next_scan.isoformat() + "Z",
        "results": results,
        "signals_fired": signals_fired,
        "dxy_state": dxy_state,
        "vix_regime": vix_regime,
    }

    await r.set("novafx:log:floor5", json.dumps(summary))
    print(f"[APEX] Wrote scan log: {signals_fired} signals, {len(results)} instruments scanned")


async def scan_loop(r: redis.Redis):
    """Main loop - scan at 21:00 UTC daily."""
    while True:
        await wait_until_target_time()
        scan_time = datetime.now(timezone.utc)
        print(f"[APEX] Starting scan at {scan_time.isoformat()}")

        results = []
        signals_fired = 0

        async with httpx.AsyncClient(timeout=30) as client:
            # Scan indices
            for symbol in INSTRUMENTS:
                try:
                    result = await scan_instrument(client, r, symbol)
                    if result:
                        results.append(result)
                        if result.get("signal") not in [None, "NO_SIGNAL"]:
                            signals_fired += 1
                    await asyncio.sleep(2)
                except Exception as e:
                    print(f"[APEX] Error scanning {symbol}: {e}")
                    results.append({"symbol": symbol, "signal": "NO_SIGNAL", "reason": f"error_{str(e)[:50]}"})

            # Publish cross-floor intel
            try:
                dxy_state, vix_regime = await publish_cross_floor_intel(client, r)
            except Exception as e:
                print(f"[APEX] Error publishing intel: {e}")

        # Write scan summary to Redis
        await write_scan_log(r, results, signals_fired, scan_time)


async def health_handler(request):
    return web.Response(text="OK", status=200)


async def run_health_server():
    app = web.Application()
    app.router.add_get("/health", health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    print(f"[APEX] Health server on port {PORT}")


async def main():
    print("[APEX] Floor 5 - Triple RSI + Cross-Floor Intel starting...")
    r = redis.from_url(REDIS_URL, decode_responses=True)

    await run_health_server()
    await scan_loop(r)


if __name__ == "__main__":
    asyncio.run(main())

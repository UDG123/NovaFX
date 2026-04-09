"""
NovaFX Paper Trader v3 - Trades from all 6 floor streams + universal scanner.

Subscribes to all 7 signal streams:
- novafx:signals:floor1 (TITAN - London Breakout)
- novafx:signals:floor2 (DRAGON - RSI Divergence)
- novafx:signals:floor3 (CIPHER - Crypto EMA/MACD)
- novafx:signals:floor4 (VERTEX - VWAP Reversion)
- novafx:signals:floor5 (APEX - Triple RSI)
- novafx:signals:floor6 (BULLION - Gold Breakout)
- novafx:signals:scored (Universal Scanner)

Monitors positions and reports per-floor + per-desk P&L breakdown to Telegram.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

import httpx
import redis.asyncio as aioredis
from aiohttp import web

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("novafx.paper-trader")

# Configuration from environment
REDIS_URL = os.getenv("REDIS_URL")
TG_BOT = os.getenv("TELEGRAM_BOT_TOKEN")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "-1003614474777")
TWELVEDATA_KEY = os.getenv("TWELVEDATA_API_KEY")
ACCOUNT_SIZE = float(os.getenv("PAPER_ACCOUNT_SIZE", "10000"))
RISK_PCT = float(os.getenv("PAPER_RISK_PCT", "1.0"))  # 1% risk per trade
MAX_TRADE_HOURS = int(os.getenv("PAPER_MAX_TRADE_HOURS", "24"))

# Desk classification
FOREX_SYMBOLS = {"EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "EUR/GBP", "GBP/JPY",
                 "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "EURGBP", "GBPJPY"}
CRYPTO_SYMBOLS = {"BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD", "AVAX/USD", "DOGE/USD", "LINK/USD",
                  "BTCUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD", "AVAXUSD", "DOGEUSD", "LINKUSD"}
STOCK_SYMBOLS = {"AAPL", "MSFT", "NVDA", "META", "AMZN", "GOOGL", "SPY", "QQQ"}

# All signal streams to monitor (6 floors + v2 universal scanner)
SIGNAL_STREAMS = {
    "novafx:signals:floor1": "TITAN",
    "novafx:signals:floor2": "DRAGON",
    "novafx:signals:floor3": "CIPHER",
    "novafx:signals:floor4": "VERTEX",
    "novafx:signals:floor5": "APEX",
    "novafx:signals:floor6": "BULLION",
    "novafx:signals:scored": "SCANNER",
}

FLOOR_NAMES = ["TITAN", "DRAGON", "CIPHER", "VERTEX", "APEX", "BULLION", "SCANNER"]


def get_desk(symbol: str) -> str:
    """Classify symbol into desk: forex, crypto, or stocks."""
    sym_upper = symbol.upper().replace("/", "")
    if sym_upper in {s.replace("/", "") for s in FOREX_SYMBOLS}:
        return "forex"
    elif sym_upper in {s.replace("/", "") for s in CRYPTO_SYMBOLS}:
        return "crypto"
    elif sym_upper in STOCK_SYMBOLS:
        return "stocks"
    # Fallback heuristics
    if any(c in sym_upper for c in ["USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "NZD"]):
        if len(sym_upper) == 6:
            return "forex"
    if any(c in sym_upper for c in ["BTC", "ETH", "SOL", "XRP", "ADA", "DOGE", "LINK", "AVAX"]):
        return "crypto"
    return "stocks"


# ---------------------------------------------------------------------------
# Health check HTTP server (for Railway)
# ---------------------------------------------------------------------------


async def health_handler(request) -> web.Response:
    """Health check endpoint for Railway."""
    return web.Response(text='{"status":"ok","version":"v2"}', content_type='application/json')


async def start_health_server() -> None:
    """Start minimal health check HTTP server."""
    app = web.Application()
    app.router.add_get('/health', health_handler)
    app.router.add_get('/', health_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    port = int(os.getenv('PORT', 8000))
    site = web.TCPSite(runner, '0.0.0.0', port)
    await site.start()
    logger.info(f"Health server running on :{port}")


async def send_telegram(msg: str) -> bool:
    """Send message to Telegram channel."""
    if not TG_BOT or not TG_CHAT:
        logger.warning("Telegram not configured, skipping message")
        logger.info(f"Would send: {msg[:100]}...")
        return False

    url = f"https://api.telegram.org/bot{TG_BOT}/sendMessage"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json={
                "chat_id": TG_CHAT,
                "text": msg,
                "parse_mode": "Markdown"
            })
            if resp.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram error: {resp.text[:200]}")
                return False
    except Exception as e:
        logger.error(f"Telegram send error: {e}")
        return False


async def get_price(symbol: str) -> float | None:
    """Fetch current price from TwelveData."""
    if not TWELVEDATA_KEY:
        logger.error("TWELVEDATA_API_KEY not configured")
        return None

    # Convert symbol format: EURUSD -> EUR/USD
    td_symbol = symbol
    if "/" not in symbol and len(symbol) == 6:
        td_symbol = symbol[:3] + "/" + symbol[3:]

    # Handle crypto symbols
    if symbol.endswith("USDT"):
        td_symbol = symbol.replace("USDT", "/USD")
    elif symbol.endswith("USD") and "/" not in symbol:
        # Crypto like BTCUSD -> BTC/USD
        td_symbol = symbol[:-3] + "/USD"

    url = f"https://api.twelvedata.com/price?symbol={td_symbol}&apikey={TWELVEDATA_KEY}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(url)
            data = r.json()
            if "price" in data:
                return float(data["price"])
            else:
                logger.error(f"TwelveData error for {symbol}: {data}")
                return None
    except Exception as e:
        logger.error(f"Failed to fetch price for {symbol}: {e}")
        return None


async def open_position(redis, symbol: str, direction: str, entry: float, sl: float, tp: float, score: float, desk: str, floor_name: str = "SCANNER") -> None:
    """Open a simulated paper trade using pre-calculated entry/SL/TP from signal."""
    key = f"novafx:paper:position:{symbol}:{direction}"

    # Don't double-open same direction
    existing = await redis.get(key)
    if existing:
        logger.info(f"Position already exists: {key}")
        return

    # Calculate position size based on SL distance
    sl_dist = abs(entry - sl)
    risk_amount = ACCOUNT_SIZE * (RISK_PCT / 100)
    position_size = risk_amount / sl_dist if sl_dist > 0 else 1000

    position = {
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "size": round(position_size, 2),
        "risk": round(risk_amount, 2),
        "score": score,
        "desk": desk,
        "floor": floor_name,
        "opened_at": time.time(),
    }

    await redis.setex(key, MAX_TRADE_HOURS * 3600, json.dumps(position))

    dir_emoji = "\U0001f4c8" if direction.upper() in ("BUY", "LONG") else "\U0001f4c9"
    await send_telegram(
        f"{dir_emoji} *PAPER TRADE OPENED*\n"
        f"*{symbol}* {direction.upper()} @ `{entry}`\n"
        f"\U0001f6d1 SL: `{position['sl']}` | \U0001f3af TP: `{position['tp']}`\n"
        f"\U0001f4b0 Risk: ${position['risk']} ({RISK_PCT}%)\n"
        f"\U0001f3c6 Score: {score} | Floor: {floor_name} | Desk: {desk.upper()}"
    )

    logger.info(f"Opened paper position: {direction.upper()} {symbol} @ {entry} (score={score}, floor={floor_name}, desk={desk})")


async def close_position(redis, key: str, position: dict, exit_price: float, reason: str) -> None:
    """Close a paper trade and report P&L with desk and floor tracking."""
    entry = position["entry"]
    direction = position["direction"].upper()
    size = position["size"]
    symbol = position["symbol"]
    desk = position.get("desk", get_desk(symbol))
    floor = position.get("floor", "SCANNER")

    if direction in ("BUY", "LONG"):
        pnl = (exit_price - entry) * size
    else:
        pnl = (entry - exit_price) * size

    duration_h = (time.time() - position["opened_at"]) / 3600
    won = pnl > 0

    # Update global stats
    stats_key = "novafx:paper:stats"
    stats_raw = await redis.get(stats_key)
    stats = json.loads(stats_raw) if stats_raw else {
        "trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0,
        "forex": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        "crypto": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        "stocks": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
        "floors": {},
    }

    # Ensure desk keys exist (migration from v1 stats)
    for d in ["forex", "crypto", "stocks"]:
        if d not in stats:
            stats[d] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}

    # Ensure floors dict exists
    if "floors" not in stats:
        stats["floors"] = {}

    # Ensure floor key exists
    if floor not in stats["floors"]:
        stats["floors"][floor] = {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0}

    # Update totals
    stats["trades"] += 1
    stats["wins" if won else "losses"] += 1
    stats["total_pnl"] = round(stats["total_pnl"] + pnl, 2)

    # Update desk stats
    stats[desk]["trades"] += 1
    stats[desk]["wins" if won else "losses"] += 1
    stats[desk]["pnl"] = round(stats[desk]["pnl"] + pnl, 2)

    # Update floor stats
    stats["floors"][floor]["trades"] += 1
    stats["floors"][floor]["wins" if won else "losses"] += 1
    stats["floors"][floor]["pnl"] = round(stats["floors"][floor]["pnl"] + pnl, 2)

    await redis.set(stats_key, json.dumps(stats))
    await redis.delete(key)

    result_emoji = "\u2705" if won else "\u274c"
    pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
    win_rate = round(stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0

    await send_telegram(
        f"{result_emoji} *PAPER TRADE CLOSED* — {'WIN' if won else 'LOSS'}\n"
        f"*{symbol}* {direction} | {reason}\n"
        f"Entry: `{entry}` -> Exit: `{exit_price}`\n"
        f"P&L: *{pnl_str}* | Duration: {duration_h:.1f}h\n"
        f"Floor: {floor} | \U0001f4ca Session: {stats['trades']} trades | {win_rate}% WR | Total: ${stats['total_pnl']:+.2f}"
    )

    logger.info(f"Closed position: {direction} {symbol} P&L: {pnl_str} (floor={floor}, desk={desk})")


async def monitor_positions(redis) -> None:
    """Check all open positions against current prices."""
    pattern = "novafx:paper:position:*"
    keys = []

    async for key in redis.scan_iter(match=pattern, count=100):
        keys.append(key)

    for key in keys:
        raw = await redis.get(key)
        if not raw:
            continue

        pos = json.loads(raw)
        symbol = pos["symbol"]
        direction = pos["direction"].upper()

        price = await get_price(symbol)
        if not price:
            continue

        sl = pos["sl"]
        tp = pos["tp"]
        age_h = (time.time() - pos["opened_at"]) / 3600

        if direction in ("BUY", "LONG"):
            if price <= sl:
                await close_position(redis, key, pos, price, "SL Hit")
            elif price >= tp:
                await close_position(redis, key, pos, price, "TP Hit \U0001f3af")
            elif age_h >= MAX_TRADE_HOURS:
                await close_position(redis, key, pos, price, f"Timeout {MAX_TRADE_HOURS}h")
        else:
            if price >= sl:
                await close_position(redis, key, pos, price, "SL Hit")
            elif price <= tp:
                await close_position(redis, key, pos, price, "TP Hit \U0001f3af")
            elif age_h >= MAX_TRADE_HOURS:
                await close_position(redis, key, pos, price, f"Timeout {MAX_TRADE_HOURS}h")


async def listen_signals(redis) -> None:
    """Listen for signals on all 7 streams (6 floors + v2 universal scanner)."""
    # Track last ID for each stream
    last_ids = {stream: "$" for stream in SIGNAL_STREAMS}

    logger.info(f"Listening for signals on {len(SIGNAL_STREAMS)} streams: {list(SIGNAL_STREAMS.keys())}")

    while True:
        try:
            entries = await redis.xread(last_ids, count=10, block=2000)
            for stream, messages in (entries or []):
                stream_key = stream.decode() if isinstance(stream, bytes) else stream
                floor_name = SIGNAL_STREAMS.get(stream_key, "UNKNOWN")

                for msg_id, data in messages:
                    msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                    last_ids[stream_key] = msg_id_str

                    # Decode data from bytes
                    def decode_field(key):
                        val = data.get(key.encode() if isinstance(key, str) else key, b"")
                        return val.decode() if isinstance(val, bytes) else str(val) if val else ""

                    symbol = decode_field("symbol")
                    direction = decode_field("direction")

                    # Get pre-calculated entry/sl/tp from signal dict
                    entry_raw = data.get(b"entry", data.get("entry", b"0"))
                    entry = float(entry_raw.decode() if isinstance(entry_raw, bytes) else entry_raw) if entry_raw else 0

                    sl_raw = data.get(b"sl", data.get("sl", b"0"))
                    sl = float(sl_raw.decode() if isinstance(sl_raw, bytes) else sl_raw) if sl_raw else 0

                    tp_raw = data.get(b"tp", data.get("tp", b"0"))
                    tp = float(tp_raw.decode() if isinstance(tp_raw, bytes) else tp_raw) if tp_raw else 0

                    # TP can be TP1 in some signal formats
                    if tp == 0:
                        tp1_raw = data.get(b"tp1", data.get("tp1", b"0"))
                        tp = float(tp1_raw.decode() if isinstance(tp1_raw, bytes) else tp1_raw) if tp1_raw else 0

                    score_raw = data.get(b"final_score", data.get(b"score", data.get("final_score", data.get("score", b"0"))))
                    score = float(score_raw.decode() if isinstance(score_raw, bytes) else score_raw) if score_raw else 0

                    # Get floor_name from message or derive from stream
                    msg_floor = decode_field("floor_name") or decode_field("floor")
                    if msg_floor:
                        floor_name = msg_floor.upper()

                    # Get desk from signal or classify
                    desk = decode_field("desk") or get_desk(symbol)

                    if symbol and direction and entry > 0 and sl > 0 and tp > 0:
                        logger.info(f"[{floor_name}] Signal: {direction} {symbol} @ {entry} (score={score}, desk={desk})")
                        await open_position(redis, symbol, direction, entry, sl, tp, score, desk, floor_name)
                    else:
                        logger.warning(f"[{floor_name}] Invalid signal: symbol={symbol}, direction={direction}, entry={entry}, sl={sl}, tp={tp}")

        except Exception as e:
            if "NOGROUP" not in str(e):
                logger.error(f"Signal listener error: {e}")
            await asyncio.sleep(5)


async def daily_summary(redis) -> None:
    """Send daily P&L summary at midnight UTC with per-desk and per-floor breakdown."""
    while True:
        now = datetime.now(timezone.utc)
        # Wait until next midnight
        seconds_until_midnight = (24 - now.hour) * 3600 - now.minute * 60 - now.second
        if seconds_until_midnight <= 0:
            seconds_until_midnight = 86400  # Full day
        await asyncio.sleep(seconds_until_midnight)

        stats_raw = await redis.get("novafx:paper:stats")
        if stats_raw:
            stats = json.loads(stats_raw)
            total_trades = stats.get("trades", 0)
            total_wins = stats.get("wins", 0)
            total_pnl = stats.get("total_pnl", 0.0)
            win_rate = round(total_wins / total_trades * 100) if total_trades > 0 else 0

            # Per-desk breakdown
            forex = stats.get("forex", {"trades": 0, "wins": 0, "pnl": 0.0})
            crypto = stats.get("crypto", {"trades": 0, "wins": 0, "pnl": 0.0})
            stocks = stats.get("stocks", {"trades": 0, "wins": 0, "pnl": 0.0})

            # Per-floor breakdown
            floors = stats.get("floors", {})

            def stat_line(name, d):
                t = d.get("trades", 0)
                w = d.get("wins", 0)
                p = d.get("pnl", 0.0)
                wr = round(w / t * 100) if t > 0 else 0
                return f"{name}: {t} trades | {wr}% WR | ${p:+.2f}"

            # Build floor lines for active floors only
            floor_lines = []
            for floor_name in FLOOR_NAMES:
                if floor_name in floors and floors[floor_name].get("trades", 0) > 0:
                    floor_lines.append(stat_line(floor_name, floors[floor_name]))

            floor_section = "\n".join(floor_lines) if floor_lines else "No floor trades today"

            await send_telegram(
                f"\U0001f4ca *NovaFX Paper Trading Daily Report*\n"
                f"*v3 | 6-Floor Stack + Scanner*\n"
                f"Date: {now.strftime('%b %d, %Y')}\n\n"
                f"*TOTALS*\n"
                f"Trades: {total_trades} | Wins: {total_wins} | Losses: {total_trades - total_wins}\n"
                f"Win Rate: {win_rate}%\n"
                f"Total P&L: *${total_pnl:+.2f}*\n\n"
                f"*BY FLOOR*\n"
                f"{floor_section}\n\n"
                f"*BY DESK*\n"
                f"\U0001f4b1 {stat_line('Forex', forex)}\n"
                f"\U0001f4b0 {stat_line('Crypto', crypto)}\n"
                f"\U0001f4c8 {stat_line('Stocks', stocks)}\n\n"
                f"Account: ${ACCOUNT_SIZE + total_pnl:,.2f} (started ${ACCOUNT_SIZE:,.0f})"
            )
            # Reset daily stats
            await redis.delete("novafx:paper:stats")
            logger.info("Daily summary sent, stats reset")


async def monitor_positions_loop(redis) -> None:
    """Monitor positions every 5 minutes."""
    while True:
        await monitor_positions(redis)
        await asyncio.sleep(300)  # Check every 5 min


async def main() -> None:
    """Entry point."""
    if not REDIS_URL:
        logger.error("REDIS_URL not configured, exiting")
        return

    redis = await aioredis.from_url(REDIS_URL, decode_responses=False)
    logger.info(f"Paper Trader v3 started | Account: ${ACCOUNT_SIZE:,.0f} | Risk: {RISK_PCT}% | Streams: {len(SIGNAL_STREAMS)}")

    await send_telegram(
        f"\U0001f916 *NovaFX Paper Trader Started*\n"
        f"*v3 | 6-Floor Stack + Scanner*\n\n"
        f"Account: ${ACCOUNT_SIZE:,.0f}\n"
        f"Risk per trade: {RISK_PCT}% (${ACCOUNT_SIZE * RISK_PCT / 100:.0f})\n"
        f"Streams: {len(SIGNAL_STREAMS)} (Titan/Dragon/Cipher/Vertex/Apex/Bullion/Scanner)\n"
        f"Monitoring: Per-floor + Per-desk P&L"
    )

    try:
        await asyncio.gather(
            start_health_server(),
            listen_signals(redis),
            monitor_positions_loop(redis),
            daily_summary(redis),
        )
    except asyncio.CancelledError:
        logger.info("Paper Trader cancelled")
    finally:
        await redis.close()
        logger.info("Paper Trader stopped")


if __name__ == "__main__":
    asyncio.run(main())

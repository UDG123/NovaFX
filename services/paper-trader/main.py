"""
NovaFX Paper Trader - Simulates trades from signal pipeline and reports P&L to Telegram.

Subscribes to Redis signals via novafx:signals:* stream keys, monitors positions,
and sends results to Telegram.
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
SL_PIPS = float(os.getenv("PAPER_SL_PIPS", "50"))  # 50 pips SL for forex
TP_RATIO = float(os.getenv("PAPER_TP_RATIO", "2.0"))  # 2:1 TP:SL ratio
MAX_TRADE_HOURS = int(os.getenv("PAPER_MAX_TRADE_HOURS", "24"))

# Position store key pattern: novafx:paper:position:{symbol}:{direction}
# Stats key: novafx:paper:stats


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


def calc_sl_tp(entry: float, direction: str, symbol: str) -> tuple[float, float, float]:
    """Calculate SL and TP based on direction and instrument type."""
    is_crypto = any(c in symbol.upper() for c in ["BTC", "ETH", "SOL", "XRP", "ADA", "AVAX", "DOGE", "LINK"])
    is_stock = any(c in symbol.upper() for c in ["AAPL", "TSLA", "NVDA", "SPY", "QQQ", "MSFT", "AMZN", "GOOG", "META", "AMD"])

    if is_crypto:
        sl_dist = entry * 0.02  # 2% SL
    elif is_stock:
        sl_dist = entry * 0.015  # 1.5% SL
    else:
        # Forex: pips-based (assume 4-decimal pairs)
        sl_dist = SL_PIPS * 0.0001

    tp_dist = sl_dist * TP_RATIO
    risk_amount = ACCOUNT_SIZE * (RISK_PCT / 100)
    position_size = risk_amount / sl_dist if sl_dist > 0 else 1000

    if direction.upper() in ("BUY", "LONG"):
        return entry - sl_dist, entry + tp_dist, position_size
    else:
        return entry + sl_dist, entry - tp_dist, position_size


async def open_position(redis, symbol: str, direction: str, entry: float) -> None:
    """Open a simulated paper trade."""
    key = f"novafx:paper:position:{symbol}:{direction}"

    # Don't double-open same direction
    existing = await redis.get(key)
    if existing:
        logger.info(f"Position already exists: {key}")
        return

    sl, tp, size = calc_sl_tp(entry, direction, symbol)
    risk_amount = ACCOUNT_SIZE * (RISK_PCT / 100)

    position = {
        "symbol": symbol,
        "direction": direction,
        "entry": entry,
        "sl": round(sl, 5),
        "tp": round(tp, 5),
        "size": round(size, 2),
        "risk": round(risk_amount, 2),
        "opened_at": time.time(),
    }

    await redis.setex(key, MAX_TRADE_HOURS * 3600, json.dumps(position))

    dir_emoji = "\U0001f4c8" if direction.upper() in ("BUY", "LONG") else "\U0001f4c9"
    await send_telegram(
        f"{dir_emoji} *PAPER TRADE OPENED*\n"
        f"*{symbol}* {direction.upper()} @ `{entry}`\n"
        f"\U0001f6d1 SL: `{position['sl']}` | \U0001f3af TP: `{position['tp']}`\n"
        f"\U0001f4b0 Risk: ${position['risk']} ({RISK_PCT}% of ${ACCOUNT_SIZE:,.0f})"
    )

    logger.info(f"Opened paper position: {direction.upper()} {symbol} @ {entry}")


async def close_position(redis, key: str, position: dict, exit_price: float, reason: str) -> None:
    """Close a paper trade and report P&L."""
    entry = position["entry"]
    direction = position["direction"].upper()
    size = position["size"]
    symbol = position["symbol"]

    if direction in ("BUY", "LONG"):
        pnl = (exit_price - entry) * size
    else:
        pnl = (entry - exit_price) * size

    duration_h = (time.time() - position["opened_at"]) / 3600
    won = pnl > 0

    # Update stats
    stats_key = "novafx:paper:stats"
    stats_raw = await redis.get(stats_key)
    stats = json.loads(stats_raw) if stats_raw else {"trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0}
    stats["trades"] += 1
    stats["wins" if won else "losses"] += 1
    stats["total_pnl"] = round(stats["total_pnl"] + pnl, 2)
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
        f"\U0001f4ca Session: {stats['trades']} trades | {win_rate}% WR | Total: ${stats['total_pnl']:+.2f}"
    )

    logger.info(f"Closed position: {direction} {symbol} P&L: {pnl_str}")


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
    """Listen for new signals on Redis stream."""
    # Check the novafx:signals:forex, crypto, stocks streams
    stream_keys = ["novafx:signals:forex", "novafx:signals:crypto", "novafx:signals:stocks"]
    last_ids = {k: "$" for k in stream_keys}

    logger.info(f"Listening for signals on streams: {stream_keys}")

    while True:
        try:
            for stream_key in stream_keys:
                try:
                    entries = await redis.xread({stream_key: last_ids[stream_key]}, count=10, block=1000)
                    for stream, messages in (entries or []):
                        stream_name = stream.decode() if isinstance(stream, bytes) else stream
                        for msg_id, data in messages:
                            msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
                            last_ids[stream_name] = msg_id_str

                            # Decode data
                            symbol = data.get(b"symbol", b"").decode() if isinstance(data.get(b"symbol"), bytes) else data.get("symbol", "")
                            direction = data.get(b"direction", b"").decode() if isinstance(data.get(b"direction"), bytes) else data.get("direction", "")
                            entry_raw = data.get(b"entry", b"0") if b"entry" in data else data.get("entry", "0")
                            entry = float(entry_raw.decode() if isinstance(entry_raw, bytes) else entry_raw) if entry_raw else 0

                            if symbol and direction and entry > 0:
                                logger.info(f"Signal received: {direction} {symbol} @ {entry}")
                                await open_position(redis, symbol, direction, entry)
                except Exception as e:
                    if "NOGROUP" not in str(e):
                        logger.debug(f"Stream {stream_key} read: {e}")
        except Exception as e:
            logger.error(f"Signal listener error: {e}")
            await asyncio.sleep(5)


async def daily_summary(redis) -> None:
    """Send daily P&L summary at midnight UTC."""
    while True:
        now = datetime.now(timezone.utc)
        # Wait until next midnight
        seconds_until_midnight = (24 - now.hour) * 3600 - now.minute * 60 - now.second
        await asyncio.sleep(seconds_until_midnight)

        stats_raw = await redis.get("novafx:paper:stats")
        if stats_raw:
            stats = json.loads(stats_raw)
            win_rate = round(stats["wins"] / stats["trades"] * 100) if stats["trades"] > 0 else 0
            await send_telegram(
                f"\U0001f4ca *NovaFX Daily Paper Trading Report*\n"
                f"Date: {now.strftime('%b %d, %Y')}\n"
                f"Trades: {stats['trades']} | Wins: {stats['wins']} | Losses: {stats['losses']}\n"
                f"Win Rate: {win_rate}%\n"
                f"Total P&L: *${stats['total_pnl']:+.2f}*\n"
                f"Account: ${ACCOUNT_SIZE + stats['total_pnl']:,.2f} (started ${ACCOUNT_SIZE:,.0f})"
            )
            # Reset daily stats
            await redis.delete("novafx:paper:stats")


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
    logger.info(f"Paper Trader started | Account: ${ACCOUNT_SIZE:,.0f} | Risk: {RISK_PCT}% | SL: {SL_PIPS} pips")

    await send_telegram(
        f"\U0001f916 *NovaFX Paper Trader Started*\n"
        f"Account: ${ACCOUNT_SIZE:,.0f}\n"
        f"Risk per trade: {RISK_PCT}% (${ACCOUNT_SIZE * RISK_PCT / 100:.0f})\n"
        f"TP:SL Ratio: {TP_RATIO}:1\n"
        f"Monitoring: Forex, Crypto, Stocks"
    )

    try:
        await asyncio.gather(
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

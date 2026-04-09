"""
NovaFX Paper Trader - Position Management Logic.

Handles simulated position opening, monitoring, and closing.
Tracks P&L and sends results to Telegram.
"""

import json
import logging
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger("novafx.paper-trader")

# Configuration from environment
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
ACCOUNT_BALANCE = float(os.getenv("ACCOUNT_BALANCE", "10000"))
POSITION_SIZE_USD = float(os.getenv("POSITION_SIZE_USD", "1000"))
SL_PIPS = float(os.getenv("SL_PIPS", "30"))
TP_PIPS = float(os.getenv("TP_PIPS", "60"))

# JPY pairs have different pip multiplier
JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "NZDJPY", "CHFJPY"}

# Symbol mapping for TwelveData API
SYMBOL_MAP = {
    # Forex - TwelveData uses slash format
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "AUDUSD": "AUD/USD",
    "USDCAD": "USD/CAD",
    "USDCHF": "USD/CHF",
    "NZDUSD": "NZD/USD",
    "EURGBP": "EUR/GBP",
    "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
    # Crypto
    "BTCUSDT": "BTC/USD",
    "ETHUSDT": "ETH/USD",
    "SOLUSDT": "SOL/USD",
    "BNBUSDT": "BNB/USD",
    "XRPUSDT": "XRP/USD",
    # Stocks
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
    "SPY": "SPY",
    "QQQ": "QQQ",
    # Commodities
    "XAUUSD": "XAU/USD",
    "XAGUSD": "XAG/USD",
}


@dataclass
class Position:
    symbol: str
    direction: str  # "buy" or "sell"
    entry_price: float
    entry_time: str  # ISO format
    size_usd: float
    sl: float
    tp: float
    status: str  # "open" or "closed"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Position":
        return cls(**data)


def get_pip_value(symbol: str) -> float:
    """Get pip value for symbol (JPY pairs have different pip sizing)."""
    clean_symbol = symbol.replace("/", "").replace("_", "").upper()
    if clean_symbol in JPY_PAIRS:
        return 0.01  # 1 pip = 0.01 for JPY pairs
    return 0.0001  # 1 pip = 0.0001 for standard pairs


def normalize_symbol(symbol: str) -> str:
    """Normalize symbol to standard format (no slashes)."""
    return symbol.replace("/", "").replace("_", "").upper()


def get_twelvedata_symbol(symbol: str) -> str:
    """Convert to TwelveData API symbol format."""
    clean = normalize_symbol(symbol)
    return SYMBOL_MAP.get(clean, symbol)


async def fetch_current_price(symbol: str) -> Optional[float]:
    """Fetch current price from TwelveData API."""
    if not TWELVEDATA_API_KEY:
        logger.error("TWELVEDATA_API_KEY not configured")
        return None

    td_symbol = get_twelvedata_symbol(symbol)
    url = f"https://api.twelvedata.com/price?symbol={td_symbol}&apikey={TWELVEDATA_API_KEY}"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            data = response.json()

            if "price" in data:
                return float(data["price"])
            else:
                logger.error(f"TwelveData error for {symbol}: {data}")
                return None
    except Exception as e:
        logger.error(f"Failed to fetch price for {symbol}: {e}")
        return None


def calculate_sl_tp(
    entry_price: float, direction: str, symbol: str
) -> tuple[float, float]:
    """Calculate SL and TP prices based on direction and pip values."""
    pip_value = get_pip_value(symbol)
    sl_distance = SL_PIPS * pip_value
    tp_distance = TP_PIPS * pip_value

    if direction.lower() == "sell":
        sl = round(entry_price + sl_distance, 6)
        tp = round(entry_price - tp_distance, 6)
    else:  # buy
        sl = round(entry_price - sl_distance, 6)
        tp = round(entry_price + tp_distance, 6)

    return sl, tp


def create_position(symbol: str, direction: str, entry_price: float) -> Position:
    """Create a new paper position."""
    sl, tp = calculate_sl_tp(entry_price, direction, symbol)

    return Position(
        symbol=normalize_symbol(symbol),
        direction=direction.lower(),
        entry_price=round(entry_price, 6),
        entry_time=datetime.now(timezone.utc).isoformat(),
        size_usd=POSITION_SIZE_USD,
        sl=sl,
        tp=tp,
        status="open",
    )


def check_position_exit(
    position: Position, current_price: float
) -> tuple[bool, str, float]:
    """
    Check if position should be closed.

    Returns: (should_close, reason, exit_price)
    """
    entry_time = datetime.fromisoformat(position.entry_time.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    duration_hours = (now - entry_time).total_seconds() / 3600

    # Check timeout (24h)
    if duration_hours > 24:
        return True, "timeout", current_price

    if position.direction == "sell":
        # For short: TP is below entry, SL is above entry
        if current_price <= position.tp:
            return True, "tp", position.tp
        if current_price >= position.sl:
            return True, "sl", position.sl
    else:  # buy
        # For long: TP is above entry, SL is below entry
        if current_price >= position.tp:
            return True, "tp", position.tp
        if current_price <= position.sl:
            return True, "sl", position.sl

    return False, "", 0.0


def calculate_pnl(position: Position, exit_price: float) -> tuple[float, float]:
    """
    Calculate P&L for a closed position.

    Returns: (pnl_usd, pnl_percent)
    """
    pip_value = get_pip_value(position.symbol)
    price_diff = exit_price - position.entry_price

    # For sell positions, profit is when price goes down
    if position.direction == "sell":
        price_diff = -price_diff

    # Calculate pips moved
    pips_moved = price_diff / pip_value

    # Simple P&L calculation: (pips * size_usd * pip_value) / entry_price
    # For forex with $1000 position at 1.1000, moving 60 pips = $54.55
    pnl_usd = (price_diff / position.entry_price) * position.size_usd
    pnl_percent = (pnl_usd / position.size_usd) * 100

    return round(pnl_usd, 2), round(pnl_percent, 2)


def format_duration(start_time: str) -> str:
    """Format duration from start time to now as 'Xh Ym'."""
    entry = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    delta = now - entry
    hours = int(delta.total_seconds() // 3600)
    minutes = int((delta.total_seconds() % 3600) // 60)
    return f"{hours}h {minutes}m"


async def send_telegram(message: str) -> bool:
    """Send message to Telegram channel."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured, skipping message")
        logger.info(f"Would send: {message}")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, json=payload)
            if response.status_code == 200:
                logger.info("Telegram message sent successfully")
                return True
            else:
                logger.error(f"Telegram error: {response.text}")
                return False
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False


def format_entry_message(position: Position) -> str:
    """Format the trade entry message for Telegram."""
    direction_emoji = "\U0001F4C9" if position.direction == "sell" else "\U0001F4C8"
    direction_text = position.direction.upper()

    return f"""\U0001F4E5 PAPER TRADE OPENED
{direction_emoji} {direction_text} {position.symbol} @ {position.entry_price}
\U0001F4B0 Size: ${position.size_usd:,.0f} | \u2696\uFE0F Account: ${ACCOUNT_BALANCE:,.0f}
\U0001F6D1 SL: {position.sl} | \U0001F3AF TP: {position.tp}"""


def format_exit_message(
    position: Position, exit_price: float, reason: str, pnl_usd: float, pnl_percent: float
) -> str:
    """Format the trade exit message for Telegram."""
    direction_emoji = "\U0001F4C9" if position.direction == "sell" else "\U0001F4C8"
    direction_text = position.direction.upper()
    duration = format_duration(position.entry_time)

    if reason == "tp" or pnl_usd > 0:
        result_emoji = "\u2705 WIN"
        pnl_sign = "+"
    else:
        result_emoji = "\u274C LOSS"
        pnl_sign = ""

    return f"""{result_emoji}: {direction_text} {position.symbol}
Entry: {position.entry_price} \u2192 Exit: {exit_price}
P&L: {pnl_sign}${pnl_usd:,.2f} ({pnl_sign}{pnl_percent:.2f}%) | Duration: {duration}"""


def format_daily_summary(
    trades: list[dict], account_start: float, account_end: float, date_str: str
) -> str:
    """Format daily summary message for Telegram."""
    total_trades = len(trades)
    wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
    losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]
    win_count = len(wins)
    loss_count = len(losses)
    win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
    total_pnl = sum(t.get("pnl_usd", 0) for t in trades)

    best_trade = max(trades, key=lambda t: t.get("pnl_usd", 0)) if trades else None
    worst_trade = min(trades, key=lambda t: t.get("pnl_usd", 0)) if trades else None

    account_change = ((account_end - account_start) / account_start) * 100
    account_sign = "+" if account_change >= 0 else ""

    msg = f"""\U0001F4CA NovaFX Paper Trading \u2014 Daily Report
Date: {date_str}
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
Trades: {total_trades} | \u2705 Wins: {win_count} | \u274C Losses: {loss_count}
Win Rate: {win_rate:.1f}%
Total P&L: {"+" if total_pnl >= 0 else ""}${total_pnl:,.2f}
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501"""

    if best_trade:
        msg += f"""
Best: {best_trade['direction'].upper()} {best_trade['symbol']} +${best_trade['pnl_usd']:,.2f}"""

    if worst_trade:
        sign = "" if worst_trade["pnl_usd"] < 0 else "+"
        msg += f"""
Worst: {worst_trade['direction'].upper()} {worst_trade['symbol']} {sign}${worst_trade['pnl_usd']:,.2f}"""

    msg += f"""
\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501
Account: ${account_start:,.2f} \u2192 ${account_end:,.2f} ({account_sign}{account_change:.2f}%)"""

    return msg

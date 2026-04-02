import logging

import httpx

from app.config import settings
from app.models.signals import ProcessedSignal

logger = logging.getLogger(__name__)

TELEGRAM_SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"

MARKET_UNITS = {
    "forex": "pips",
    "crypto": "%",
    "indices": "pts",
    "commodities": "pts",
}

FOREX_SYMBOLS = {"EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","NZDUSD","USDCHF","EURGBP","EURJPY","GBPJPY"}
CRYPTO_SYMBOLS = {"BTCUSD","ETHUSD","BTCUSDT","ETHUSDT","SOLUSD","SOLUSDT","BNBUSD","BNBUSDT","XRPUSD","XRPUSDT"}
COMMODITY_SYMBOLS = {"XAUUSD","XAGUSD","USOIL","UKOIL"}


def _detect_market(symbol: str) -> str:
    s = symbol.upper().replace("/","").replace("-","")
    if s in FOREX_SYMBOLS: return "forex"
    if s in CRYPTO_SYMBOLS: return "crypto"
    if s in COMMODITY_SYMBOLS: return "commodities"
    return "indices"


def _format_pip_diff(price: float, ref: float, market: str) -> str:
    if market == "forex":
        diff = abs(price - ref) * 10000
        return f"+{diff:.0f} pips" if price > ref else f"-{diff:.0f} pips"
    elif market == "crypto":
        diff = abs(price - ref) / ref * 100
        return f"+{diff:.2f}%" if price > ref else f"-{diff:.2f}%"
    else:
        diff = abs(price - ref)
        return f"+{diff:.1f} pts" if price > ref else f"-{diff:.1f} pts"


def _format_price(price: float, market: str) -> str:
    if market == "forex":
        return f"{price:.5f}"
    elif market == "crypto":
        if price > 1000:
            return f"{price:,.2f}"
        return f"{price:.4f}"
    else:
        return f"{price:,.2f}"


def format_signal_message(signal: ProcessedSignal) -> str:
    market = _detect_market(signal.symbol)
    direction = "\U0001f4c8 BUY" if signal.action == "BUY" else "\U0001f4c9 SELL"
    pair_emoji = "\U0001f537" if signal.action == "BUY" else "\U0001f536"

    entry = _format_price(signal.entry_price, market)
    sl = _format_price(signal.stop_loss, market)
    tp1 = _format_price(signal.take_profit_1, market)
    tp2 = _format_price(signal.take_profit_2, market)
    tp3 = _format_price(signal.take_profit_3, market)

    sl_diff = _format_pip_diff(signal.stop_loss, signal.entry_price, market)
    tp1_diff = _format_pip_diff(signal.take_profit_1, signal.entry_price, market)
    tp2_diff = _format_pip_diff(signal.take_profit_2, signal.entry_price, market)
    tp3_diff = _format_pip_diff(signal.take_profit_3, signal.entry_price, market)

    indicator_line = f"\n\U0001f9e0 <b>Strategy:</b> {signal.indicator}" if signal.indicator else ""

    return (
        f"\u26a1 <b>NOVAFX SIGNAL</b>\n\n"
        f"{pair_emoji} <b>{signal.symbol}</b>  {direction}\n\n"
        f"\U0001f4cd <b>Entry:</b> <code>{entry}</code>\n"
        f"\U0001f534 <b>Stop Loss:</b> <code>{sl}</code>  <i>({sl_diff})</i>\n\n"
        f"\u2705 <b>TP1:</b> <code>{tp1}</code>  <i>({tp1_diff})</i>\n"
        f"\u2705 <b>TP2:</b> <code>{tp2}</code>  <i>({tp2_diff})</i>\n"
        f"\u2705 <b>TP3:</b> <code>{tp3}</code>  <i>({tp3_diff})</i>\n\n"
        f"\u2696\ufe0f <b>R:R \u2192</b> 1:{signal.risk_reward}  |  "
        f"<b>Risk:</b> ${signal.risk_amount}"
        f"\n\U0001f4ca <b>Timeframe:</b> {signal.timeframe}"
        f"{indicator_line}"
        f"\n\U0001f4c5 <i>{signal.timestamp.strftime('%d %b %Y  %H:%M UTC')}</i>\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u26a0\ufe0f <i>Risk max 1-2% per trade. Not financial advice.</i>"
    )


async def send_signal(signal: ProcessedSignal) -> bool:
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured - skipping alert")
        return False

    url = TELEGRAM_SEND_URL.format(token=settings.TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": format_signal_message(signal),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            logger.info("Signal sent: %s %s", signal.action, signal.symbol)
            return True
    except httpx.HTTPError:
        logger.error("Failed to send signal: %s %s", signal.action, signal.symbol)
        return False

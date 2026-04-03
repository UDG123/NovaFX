import logging

import httpx

from app.config import settings
from app.models.signals import ProcessedSignal

logger = logging.getLogger(__name__)

TELEGRAM_SEND_URL = "https://api.telegram.org/bot{token}/sendMessage"

DESK_MAP = {
    "EURUSD": "TG_DESK1", "GBPUSD": "TG_DESK1", "USDJPY": "TG_DESK1",
    "AUDUSD": "TG_DESK1", "USDCAD": "TG_DESK1", "USDCHF": "TG_DESK1",
    "NZDUSD": "TG_DESK1",
    "EURGBP": "TG_DESK2", "EURJPY": "TG_DESK2", "GBPJPY": "TG_DESK2",
    "BTCUSDT": "TG_DESK3", "ETHUSDT": "TG_DESK3", "SOLUSDT": "TG_DESK3",
    "BNBUSDT": "TG_DESK3", "XRPUSDT": "TG_DESK3",
    "BTCUSD": "TG_DESK3", "ETHUSD": "TG_DESK3", "SOLUSD": "TG_DESK3",
    "BNBUSD": "TG_DESK3", "XRPUSD": "TG_DESK3",
    "AAPL": "TG_DESK4", "MSFT": "TG_DESK4", "NVDA": "TG_DESK4",
    "TSLA": "TG_DESK4", "SPY": "TG_DESK4", "QQQ": "TG_DESK4",
    "XAUUSD": "TG_DESK5", "XAGUSD": "TG_DESK5", "USOIL": "TG_DESK5", "UKOIL": "TG_DESK5",
    "SPX500": "TG_DESK6", "NAS100": "TG_DESK6", "US30": "TG_DESK6",
}

DESK_NAMES = {
    "TG_DESK1": "\U0001f30d Forex Majors",
    "TG_DESK2": "\U0001f500 Forex Crosses",
    "TG_DESK3": "\u20bf Crypto",
    "TG_DESK4": "\U0001f4c8 Stocks",
    "TG_DESK5": "\U0001f947 Commodities",
    "TG_DESK6": "\U0001f4ca Indices",
}

FOREX_SYMBOLS = {
    "EURUSD","GBPUSD","USDJPY","AUDUSD","USDCAD","NZDUSD","USDCHF",
    "EURGBP","EURJPY","GBPJPY"
}
CRYPTO_SYMBOLS = {
    "BTCUSD","ETHUSD","BTCUSDT","ETHUSDT","SOLUSD","SOLUSDT",
    "BNBUSD","BNBUSDT","XRPUSD","XRPUSDT"
}
COMMODITY_SYMBOLS = {"XAUUSD","XAGUSD","USOIL","UKOIL"}


def _detect_market(symbol: str) -> str:
    s = symbol.upper().replace("/", "").replace("-", "")
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
        return f"{price:,.2f}" if price > 1000 else f"{price:.4f}"
    else:
        return f"{price:,.2f}"


def _get_desk_chat_id(symbol: str) -> str | None:
    desk_key = DESK_MAP.get(symbol.upper().replace("/", "").replace("-", ""))
    if not desk_key:
        return None
    return getattr(settings, desk_key, None) or None


def format_signal_message(signal: ProcessedSignal, htf_bias: dict | None = None) -> str:
    market = _detect_market(signal.symbol)
    desk_key = DESK_MAP.get(signal.symbol.upper().replace("/", "").replace("-", ""))
    desk_name = DESK_NAMES.get(desk_key, "\U0001f4e1 NovaFX") if desk_key else "\U0001f4e1 NovaFX"

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

    # HTF bias block
    if htf_bias:
        bias_line = (
            f"\n\n{htf_bias['emoji']} <b>Confidence:</b> {htf_bias['strength']}"
            f"\n\U0001f4ca <b>1H Bias:</b> {htf_bias['h1_trend'].capitalize()}"
            f"  |  <b>4H Bias:</b> {htf_bias['h4_trend'].capitalize()}"
            f"\n<i>{htf_bias['label']}</i>"
        )
    else:
        bias_line = ""

    return (
        f"\u26a1 <b>NOVAFX SIGNAL</b>  |  {desk_name}\n\n"
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
        f"{bias_line}"
        f"\n\U0001f4c5 <i>{signal.timestamp.strftime('%d %b %Y  %H:%M UTC')}</i>\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u26a0\ufe0f <i>Risk max 1-2% per trade. Not financial advice.</i>"
    )


async def _post_to_channel(chat_id: str, text: str, token: str) -> bool:
    url = TELEGRAM_SEND_URL.format(token=token)
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return True
    except httpx.HTTPError:
        logger.error("Failed to post to chat_id %s", chat_id)
        return False


async def send_signal(signal: ProcessedSignal, htf_bias: dict | None = None) -> bool:
    if not settings.TELEGRAM_BOT_TOKEN:
        logger.warning("TELEGRAM_BOT_TOKEN not set - skipping")
        return False

    message = format_signal_message(signal, htf_bias=htf_bias)
    token = settings.TELEGRAM_BOT_TOKEN
    sent = False

    desk_chat_id = _get_desk_chat_id(signal.symbol)
    if desk_chat_id:
        ok = await _post_to_channel(desk_chat_id, message, token)
        if ok:
            logger.info("Signal sent to desk: %s %s [%s]",
                signal.action, signal.symbol,
                htf_bias["strength"] if htf_bias else "NO_BIAS")
            sent = True
    else:
        logger.warning("No desk mapping for symbol: %s", signal.symbol)

    portfolio_id = getattr(settings, "TG_PORTFOLIO", None)
    if portfolio_id:
        await _post_to_channel(portfolio_id, message, token)

    return sent


async def send_system_alert(message: str) -> bool:
    if not settings.TELEGRAM_BOT_TOKEN:
        return False
    system_id = getattr(settings, "TG_SYSTEM", None)
    if not system_id:
        return False
    return await _post_to_channel(system_id, message, settings.TELEGRAM_BOT_TOKEN)

"""
Live price fetcher for trade outcome monitoring.

Asset class routing:
- Crypto  -> CCXT / Binance REST (free, no auth)
- Forex   -> yfinance (free, 60s delay acceptable for TP/SL checks)
- Gold    -> yfinance GC=F
- Stocks  -> yfinance
- Indices -> yfinance
"""
import logging
from typing import Optional

import httpx
import yfinance as yf

logger = logging.getLogger(__name__)

# Map NovaFX symbols to yfinance tickers
YFINANCE_MAP = {
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X",    "USDCHF": "CHF=X",
    "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "XAUUSD": "GC=F",     "XAGUSD": "SI=F",
    "AAPL":   "AAPL",     "MSFT":   "MSFT",     "NVDA": "NVDA",
    "TSLA":   "TSLA",     "SPY":    "SPY",       "QQQ":  "QQQ",
    "SPX500": "^GSPC",    "NAS100": "^NDX",      "US30": "^DJI",
}

# Binance symbol map for crypto (free REST, no auth)
BINANCE_MAP = {
    "BTCUSDT": "BTCUSDT", "ETHUSDT": "ETHUSDT", "SOLUSDT": "SOLUSDT",
    "BNBUSDT": "BNBUSDT", "XRPUSDT": "XRPUSDT",
    "BTCUSD":  "BTCUSDT", "ETHUSD":  "ETHUSDT", "SOLUSD":  "SOLUSDT",
    "BNBUSD":  "BNBUSDT", "XRPUSD":  "XRPUSDT",
}

BINANCE_PRICE_URL = "https://api.binance.com/api/v3/ticker/price"


async def get_current_price(symbol: str) -> Optional[float]:
    """
    Fetch current market price for any NovaFX symbol.
    Returns None if price cannot be fetched.
    """
    s = symbol.upper().replace("/", "").replace("-", "")

    # Crypto via Binance REST (free, fast, no auth)
    if s in BINANCE_MAP:
        return await _get_binance_price(BINANCE_MAP[s])

    # Everything else via yfinance
    ticker = YFINANCE_MAP.get(s)
    if ticker:
        return _get_yfinance_price(ticker)

    logger.warning("No price source mapped for symbol: %s", symbol)
    return None


async def _get_binance_price(binance_symbol: str) -> Optional[float]:
    """Fetch price from Binance public API — no auth required."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                BINANCE_PRICE_URL,
                params={"symbol": binance_symbol}
            )
            resp.raise_for_status()
            data = resp.json()
            return float(data["price"])
    except Exception:
        logger.warning("Binance price fetch failed for %s", binance_symbol)
        return None


def _get_yfinance_price(ticker: str) -> Optional[float]:
    """Fetch latest price from yfinance — covers forex, gold, stocks, indices."""
    try:
        data = yf.download(ticker, period="1d", interval="1m",
                          progress=False, auto_adjust=True)
        if data.empty:
            return None
        return float(data["Close"].iloc[-1])
    except Exception:
        logger.warning("yfinance price fetch failed for %s", ticker)
        return None

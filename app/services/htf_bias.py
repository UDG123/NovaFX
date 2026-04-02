"""
Higher Timeframe Bias Filter for NovaFX.

Fetches 1H and 4H candles via yfinance (free, no auth, Railway-compatible).
Scores each signal as: STRONG | MODERATE | COUNTER-TREND
This does NOT block signals — it adds confidence context to the Telegram message.

Bias logic:
  4H trend  + 1H momentum + 15m signal all agree  -> STRONG
  2 of 3 agree                                     -> MODERATE
  Signal goes against 4H or 1H                     -> COUNTER-TREND
"""
import logging
from typing import Literal

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

BiasStrength = Literal["STRONG", "MODERATE", "COUNTER-TREND"]

# Map NovaFX symbols to yfinance tickers
YFINANCE_MAP = {
    # Forex Majors
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X", "USDCHF": "CHF=X",
    "NZDUSD": "NZDUSD=X",
    # Forex Crosses
    "EURGBP": "EURGBP=X", "EURJPY": "EURJPY=X", "GBPJPY": "GBPJPY=X",
    # Crypto
    "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD", "SOLUSDT": "SOL-USD",
    "BNBUSDT": "BNB-USD", "XRPUSDT": "XRP-USD",
    # Commodities
    "XAUUSD": "GC=F", "XAGUSD": "SI=F",
    # Stocks
    "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA",
    "TSLA": "TSLA", "SPY": "SPY", "QQQ": "QQQ",
    # Indices
    "SPX500": "^GSPC", "NAS100": "^NDX", "US30": "^DJI",
}


def _get_trend(df: pd.DataFrame) -> str:
    """Determine trend direction from EMA 20/50 cross on a candle DataFrame."""
    if df is None or len(df) < 51:
        return "neutral"
    close = df["Close"].dropna()
    ema20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
    if ema20 > ema50:
        return "bullish"
    elif ema20 < ema50:
        return "bearish"
    return "neutral"


def _fetch_candles(ticker: str, interval: str, period: str) -> pd.DataFrame | None:
    """Fetch OHLCV candles from yfinance. Returns None on failure."""
    try:
        df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=True)
        if df.empty:
            return None
        return df
    except Exception:
        logger.warning("yfinance fetch failed: %s %s", ticker, interval)
        return None


def get_htf_bias(symbol: str, signal_action: str) -> dict:
    """
    Returns HTF bias assessment for a given symbol and proposed signal direction.

    Returns:
        {
            "strength": "STRONG" | "MODERATE" | "COUNTER-TREND",
            "h1_trend": "bullish" | "bearish" | "neutral",
            "h4_trend": "bullish" | "bearish" | "neutral",
            "signal_action": "BUY" | "SELL",
            "agreements": int,  # 0-2 (how many HTFs agree with signal)
            "emoji": str,
            "label": str,
        }
    """
    ticker = YFINANCE_MAP.get(symbol.upper())
    if not ticker:
        logger.debug("No yfinance mapping for %s — skipping HTF bias", symbol)
        return _neutral_bias(signal_action)

    # Fetch 1H (last 30 days) and 4H (last 60 days)
    df_1h = _fetch_candles(ticker, "1h", "30d")
    df_4h = _fetch_candles(ticker, "1h", "60d")  # yfinance max for 1h is 730d; we downsample for 4H

    h1_trend = _get_trend(df_1h)

    # Downsample 1H to 4H
    if df_4h is not None and len(df_4h) >= 4:
        df_4h_resampled = df_4h["Close"].resample("4h").last().to_frame()
        df_4h_resampled.columns = ["Close"]
        h4_trend = _get_trend(df_4h_resampled)
    else:
        h4_trend = "neutral"

    signal_dir = "bullish" if signal_action == "BUY" else "bearish"

    agreements = sum([
        h1_trend == signal_dir,
        h4_trend == signal_dir,
    ])

    if agreements == 2:
        strength: BiasStrength = "STRONG"
        emoji = "\u2705"
        label = "Strong confluence \u2014 all timeframes aligned"
    elif agreements == 1:
        strength = "MODERATE"
        emoji = "\u26a0\ufe0f"
        label = "Moderate confluence \u2014 mixed higher timeframe signals"
    else:
        strength = "COUNTER-TREND"
        emoji = "\U0001f504"
        label = "Counter-trend \u2014 signal opposes higher timeframe bias"

    logger.info(
        "HTF bias %s %s: 1H=%s 4H=%s \u2192 %s (%d/2 agree)",
        signal_action, symbol, h1_trend, h4_trend, strength, agreements,
    )

    return {
        "strength": strength,
        "h1_trend": h1_trend,
        "h4_trend": h4_trend,
        "signal_action": signal_action,
        "agreements": agreements,
        "emoji": emoji,
        "label": label,
    }


def _neutral_bias(signal_action: str) -> dict:
    return {
        "strength": "MODERATE",
        "h1_trend": "neutral",
        "h4_trend": "neutral",
        "signal_action": signal_action,
        "agreements": 1,
        "emoji": "\u26a0\ufe0f",
        "label": "HTF data unavailable \u2014 proceed with caution",
    }

import logging
from typing import Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD

from app.data.fetcher import fetch_ohlcv
from app.models.signals import IncomingSignal

logger = logging.getLogger(__name__)

WATCHLIST = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "BTCUSD", "ETHUSD",
    "XAUUSD",
    "SPX500", "NAS100",
]


def strategy_ema_cross(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 21:
        return None
    ema9 = EMAIndicator(close=df["close"], window=9).ema_indicator()
    ema21 = EMAIndicator(close=df["close"], window=21).ema_indicator()

    prev_fast, curr_fast = ema9.iloc[-2], ema9.iloc[-1]
    prev_slow, curr_slow = ema21.iloc[-2], ema21.iloc[-1]

    if prev_fast <= prev_slow and curr_fast > curr_slow:
        action = "BUY"
    elif prev_fast >= prev_slow and curr_fast < curr_slow:
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol,
        action=action,
        price=float(df["close"].iloc[-1]),
        timeframe="15m",
        source="signal_engine",
        indicator="EMA 9/21 Cross",
    )


def strategy_rsi_extreme(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 15:
        return None
    rsi = RSIIndicator(close=df["close"], window=14).rsi()
    current_rsi = rsi.iloc[-1]

    if np.isnan(current_rsi):
        return None

    if current_rsi < 30:
        action = "BUY"
    elif current_rsi > 70:
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol,
        action=action,
        price=float(df["close"].iloc[-1]),
        timeframe="15m",
        source="signal_engine",
        indicator=f"RSI 14 Reversal ({current_rsi:.1f})",
    )


def strategy_macd_cross(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 35:
        return None
    macd_ind = MACD(close=df["close"])
    macd_line = macd_ind.macd()
    signal_line = macd_ind.macd_signal()

    prev_macd, curr_macd = macd_line.iloc[-2], macd_line.iloc[-1]
    prev_sig, curr_sig = signal_line.iloc[-2], signal_line.iloc[-1]

    if prev_macd <= prev_sig and curr_macd > curr_sig:
        action = "BUY"
    elif prev_macd >= prev_sig and curr_macd < curr_sig:
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol,
        action=action,
        price=float(df["close"].iloc[-1]),
        timeframe="15m",
        source="signal_engine",
        indicator="MACD Cross",
    )


STRATEGIES = [strategy_ema_cross, strategy_rsi_extreme, strategy_macd_cross]


async def run_signal_engine() -> list[IncomingSignal]:
    logger.info("Signal engine running - scanning %d symbols", len(WATCHLIST))
    signals: list[IncomingSignal] = []

    for symbol in WATCHLIST:
        df = await fetch_ohlcv(symbol)
        if df is None or df.empty:
            continue

        for strategy in STRATEGIES:
            try:
                signal = strategy(df, symbol)
                if signal:
                    signals.append(signal)
            except Exception:
                logger.exception("Strategy %s failed on %s", strategy.__name__, symbol)

    logger.info("Signal engine complete - %d signals generated", len(signals))
    return signals

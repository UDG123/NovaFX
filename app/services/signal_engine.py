import logging
from typing import Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

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


def strategy_bollinger_reversion(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 20:
        return None
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    close = float(df["close"].iloc[-1])

    if np.isnan(upper) or np.isnan(lower):
        return None

    if close <= lower:
        action = "BUY"
    elif close >= upper:
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol,
        action=action,
        price=close,
        timeframe="15m",
        source="signal_engine",
        indicator=f"BB Reversion (L={lower:.5g} U={upper:.5g})",
    )


STRATEGIES = [strategy_ema_cross, strategy_rsi_extreme, strategy_macd_cross, strategy_bollinger_reversion]

MIN_CONFLUENCE = 2


async def run_signal_engine() -> list[IncomingSignal]:
    logger.info("Signal engine running - scanning %d symbols", len(WATCHLIST))
    signals: list[IncomingSignal] = []

    for symbol in WATCHLIST:
        df = await fetch_ohlcv(symbol)
        if df is None or df.empty:
            continue

        # Collect each strategy's vote for this symbol
        raw: list[IncomingSignal] = []
        for strategy in STRATEGIES:
            try:
                signal = strategy(df, symbol)
                if signal:
                    raw.append(signal)
            except Exception:
                logger.exception("Strategy %s failed on %s", strategy.__name__, symbol)

        if not raw:
            continue

        # Count votes per direction
        buy_signals = [s for s in raw if s.action == "BUY"]
        sell_signals = [s for s in raw if s.action == "SELL"]

        if len(buy_signals) >= MIN_CONFLUENCE:
            direction, agreeing = "BUY", buy_signals
        elif len(sell_signals) >= MIN_CONFLUENCE:
            direction, agreeing = "SELL", sell_signals
        else:
            logger.info(
                "No confluence for %s (BUY=%d, SELL=%d) - skipping",
                symbol, len(buy_signals), len(sell_signals),
            )
            continue

        indicators = ", ".join(s.indicator for s in agreeing if s.indicator)
        signals.append(IncomingSignal(
            symbol=symbol,
            action=direction,
            price=float(df["close"].iloc[-1]),
            timeframe="15m",
            source="signal_engine",
            indicator=f"Confluence: {indicators}",
            confluence_count=len(agreeing),
        ))
        logger.info(
            "Confluence %s on %s (%d/%d strategies agree)",
            direction, symbol, len(agreeing), len(STRATEGIES),
        )

    logger.info("Signal engine complete - %d signals emitted", len(signals))
    return signals

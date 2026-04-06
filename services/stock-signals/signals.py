"""
NovaFX Stock Signal Generation.

RSI(14) + EMA(12/26) crossover with stock-appropriate parameters.
Pure numpy — no TA-Lib dependency.
"""

import logging
from typing import Optional

import numpy as np

from shared.models import AssetClass, Signal, SignalAction

logger = logging.getLogger("novafx.stocks.signals")

SL_PERCENT = 0.03   # 3% stop loss
TP_PERCENT = 0.05   # 5% take profit


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute RSI using Wilder smoothing."""
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - (100.0 / (1.0 + rs)))


def _ema(values: np.ndarray, period: int) -> float:
    """Compute latest EMA value."""
    if len(values) < period:
        return float(values[-1]) if len(values) > 0 else 0.0
    multiplier = 2.0 / (period + 1)
    ema = float(values[0])
    for val in values[1:]:
        ema = (float(val) - ema) * multiplier + ema
    return ema


def analyze_candles(
    symbol: str,
    candles: list[dict],
    data_source: str,
    data_confidence: str,
    data_stale: bool,
) -> Optional[Signal]:
    """
    Run RSI(14) + EMA(12/26) crossover on stock candle data.

    Returns Signal if conditions met, else None.
    """
    if len(candles) < 30:
        return None

    closes = np.array([c["close"] for c in candles], dtype=float)
    rsi = _rsi(closes)
    ema_fast = _ema(closes, 12)
    ema_slow = _ema(closes, 26)
    price = float(closes[-1])

    metadata = {
        "rsi": round(rsi, 2),
        "ema_fast": round(ema_fast, 4),
        "ema_slow": round(ema_slow, 4),
        "data_source": data_source,
        "data_confidence": data_confidence,
        "data_stale": data_stale,
    }

    # Buy: RSI < 35 AND ema_fast > ema_slow
    if rsi < 35 and ema_fast > ema_slow:
        confidence = min(0.5 + (35 - rsi) / 50, 0.95)
        return Signal(
            source=f"{data_source}-stocks",
            action=SignalAction.BUY,
            symbol=symbol,
            asset_class=AssetClass.STOCKS,
            confidence=round(confidence, 3),
            price=price,
            stop_loss=round(price * (1 - SL_PERCENT), 2),
            take_profit=[round(price * (1 + TP_PERCENT), 2)],
            timeframe="1h",
            strategy="RSI-EMA-Stocks",
            metadata=metadata,
        )

    # Sell: RSI > 65 AND ema_fast < ema_slow
    if rsi > 65 and ema_fast < ema_slow:
        confidence = min(0.5 + (rsi - 65) / 50, 0.95)
        return Signal(
            source=f"{data_source}-stocks",
            action=SignalAction.SELL,
            symbol=symbol,
            asset_class=AssetClass.STOCKS,
            confidence=round(confidence, 3),
            price=price,
            stop_loss=round(price * (1 + SL_PERCENT), 2),
            take_profit=[round(price * (1 - TP_PERCENT), 2)],
            timeframe="1h",
            strategy="RSI-EMA-Stocks",
            metadata=metadata,
        )

    return None

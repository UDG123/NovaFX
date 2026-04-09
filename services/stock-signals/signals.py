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


def _compute_atr(candles: list[dict], period: int = 14) -> float | None:
    """Compute ATR from candle dicts."""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return None
    return float(np.mean(trs[-period:]))


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

    if any(np.isnan(v) for v in [rsi, ema_fast, ema_slow]):
        return None

    # ATR-based stops with fallback
    atr = _compute_atr(candles)
    if atr and atr > 0:
        sl_distance = atr * 2.0
        tp_distance = atr * 4.0
    else:
        sl_distance = price * SL_PERCENT
        tp_distance = price * TP_PERCENT

    metadata = {
        "rsi": round(rsi, 2),
        "ema_fast": round(ema_fast, 4),
        "ema_slow": round(ema_slow, 4),
        "data_source": data_source,
        "data_confidence": data_confidence,
        "data_stale": data_stale,
    }

    # Buy: RSI < 40 AND ema_fast > ema_slow (lowered from 35 to improve signal flow)
    if rsi < 40 and ema_fast > ema_slow:
        confidence = min(0.4 + (40 - rsi) / 50, 0.95)
        return Signal(
            source=f"{data_source}-stocks",
            action=SignalAction.BUY,
            symbol=symbol,
            asset_class=AssetClass.STOCKS,
            confidence=round(confidence, 3),
            price=price,
            stop_loss=round(price - sl_distance, 2),
            take_profit=[round(price + tp_distance, 2)],
            timeframe="1h",
            strategy="RSI-EMA-Stocks",
            metadata=metadata,
        )

    # Sell: RSI > 60 AND ema_fast < ema_slow (lowered from 65 to improve signal flow)
    if rsi > 60 and ema_fast < ema_slow:
        confidence = min(0.4 + (rsi - 60) / 50, 0.95)
        return Signal(
            source=f"{data_source}-stocks",
            action=SignalAction.SELL,
            symbol=symbol,
            asset_class=AssetClass.STOCKS,
            confidence=round(confidence, 3),
            price=price,
            stop_loss=round(price + sl_distance, 2),
            take_profit=[round(price - tp_distance, 2)],
            timeframe="1h",
            strategy="RSI-EMA-Stocks",
            metadata=metadata,
        )

    return None

"""
NovaFX Forex Signal Generation.

Multi-timeframe RSI + MACD confluence (primary) with fallback to single-timeframe RSI.
No TA-Lib dependency — runs on any platform.
"""

import logging
from typing import Optional

import numpy as np

from shared.models import AssetClass, Signal, SignalAction
from signal_optimizer import multi_timeframe_signal

logger = logging.getLogger("novafx.forex.signals")

# JPY pairs use different pip sizing
JPY_PAIRS = {"USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY", "CAD/JPY", "NZD/JPY", "CHF/JPY"}

# Pip values
PIP_STANDARD = 0.0050   # 50 pips for most pairs
PIP_JPY = 0.50           # 50 pips for JPY pairs
TP_STANDARD = 0.0100     # 100 pips
TP_JPY = 1.00            # 100 pips for JPY


def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """Compute RSI using Wilder smoothing."""
    if len(closes) < period + 1:
        return 50.0

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder smoothing
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
    Run RSI(14) strategy on forex candle data (no EMA crossover required).

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
        sl_distance = atr * 1.5
        tp_distance = atr * 3.0
    else:
        is_jpy = symbol in JPY_PAIRS
        sl_distance = PIP_JPY if is_jpy else PIP_STANDARD
        tp_distance = TP_JPY if is_jpy else TP_STANDARD

    metadata = {
        "rsi": round(rsi, 2),
        "ema_fast": round(ema_fast, 6),
        "ema_slow": round(ema_slow, 6),
        "data_source": data_source,
        "data_confidence": data_confidence,
        "data_stale": data_stale,
    }

    # Buy: RSI < 40 (RSI-only, no EMA crossover required)
    if rsi < 40:
        confidence = min(0.4 + (40 - rsi) / 50, 0.95)
        return Signal(
            source=f"{data_source}-forex",
            action=SignalAction.BUY,
            symbol=symbol.replace("/", ""),
            asset_class=AssetClass.FOREX,
            confidence=round(confidence, 3),
            price=price,
            stop_loss=round(price - sl_distance, 6),
            take_profit=[round(price + tp_distance, 6)],
            timeframe="1h",
            strategy="RSI-EMA-Forex",
            metadata=metadata,
        )

    # Sell: RSI > 60 (RSI-only, no EMA crossover required)
    if rsi > 60:
        confidence = min(0.4 + (rsi - 60) / 50, 0.95)
        return Signal(
            source=f"{data_source}-forex",
            action=SignalAction.SELL,
            symbol=symbol.replace("/", ""),
            asset_class=AssetClass.FOREX,
            confidence=round(confidence, 3),
            price=price,
            stop_loss=round(price + sl_distance, 6),
            take_profit=[round(price - tp_distance, 6)],
            timeframe="1h",
            strategy="RSI-EMA-Forex",
            metadata=metadata,
        )

    return None


async def analyze_multi_timeframe(symbol: str) -> Optional[Signal]:
    """
    Multi-timeframe RSI + MACD confluence analysis.

    Uses 1h and 4h timeframes for stronger signal confirmation.
    Research shows this improves win rate from ~40% to ~60%.
    """
    try:
        result = await multi_timeframe_signal(symbol)
        if not result:
            return None

        action = SignalAction.BUY if result["direction"] == "BUY" else SignalAction.SELL

        return Signal(
            source="mtf-forex",
            action=action,
            symbol=symbol.replace("/", ""),
            asset_class=AssetClass.FOREX,
            confidence=round(result["confidence"] / 100, 3),  # Convert to 0-1 scale
            price=result["entry"],
            stop_loss=result["sl"],
            take_profit=[result["tp"]],
            timeframe="1h+4h",
            strategy="MTF-RSI-MACD",
            metadata={
                "rsi_1h": result["rsi_1h"],
                "rsi_4h": result["rsi_4h"],
                "macd_hist": result["macd_hist"],
                "atr": result["atr"],
            },
        )
    except Exception as e:
        logger.warning(f"Multi-timeframe analysis failed for {symbol}: {e}")
        return None

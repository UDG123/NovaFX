"""
Market regime detection for NovaFX.

Primary: ADX + Bollinger Band Width
Fallback: Hurst Exponent (R/S analysis)

Regimes:
  trending      — allow EMA, MACD
  mean_reverting — allow RSI, BB, RSI Divergence
  ranging        — suppress ALL signals
"""
import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RegimeType = Literal["trending", "mean_reverting", "ranging"]


# ---------------------------------------------------------------------------
# ADX-based regime detection (primary)
# ---------------------------------------------------------------------------


def detect_regime_adx(df: pd.DataFrame, lookback: int = 100) -> RegimeType:
    """Detect regime using ADX + Bollinger Band Width."""
    if df is None or len(df) < lookback // 2:
        return "ranging"

    subset = df.tail(lookback)

    try:
        from ta.trend import ADXIndicator
        from ta.volatility import BollingerBands

        adx_indicator = ADXIndicator(
            high=subset["high"], low=subset["low"], close=subset["close"], window=14
        )
        adx_value = adx_indicator.adx().iloc[-1]
        if np.isnan(adx_value):
            return "ranging"

        bb = BollingerBands(close=subset["close"], window=20, window_dev=2)
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        current_bbw = bb_width.iloc[-1]
        if np.isnan(current_bbw):
            return "ranging"
        bbw_percentile = (bb_width < current_bbw).sum() / len(bb_width) * 100

        if adx_value > 25 and bbw_percentile > 50:
            return "trending"
        elif adx_value < 20 and bbw_percentile < 30:
            return "mean_reverting"
        else:
            return "ranging"

    except Exception:
        logger.debug("ADX regime detection failed, falling back to Hurst")
        return _detect_regime_hurst(df, lookback)


# ---------------------------------------------------------------------------
# Hurst exponent fallback
# ---------------------------------------------------------------------------


def hurst_exponent(series: pd.Series, min_lags: int = 2, max_lags: int = 20) -> float:
    if len(series) < max_lags * 2:
        return 0.5

    series = series.dropna()
    lags = range(min_lags, max_lags)
    tau = []

    for lag in lags:
        diff = series.diff(lag).dropna()
        if len(diff) < 2:
            continue
        std = diff.std()
        if std > 0:
            tau.append(std)

    if len(tau) < 2:
        return 0.5

    log_lags = np.log(list(range(min_lags, min_lags + len(tau))))
    log_tau = np.log(tau)

    try:
        poly = np.polyfit(log_lags, log_tau, 1)
        return float(poly[0])
    except Exception:
        return 0.5


def _detect_regime_hurst(df: pd.DataFrame, lookback: int = 100) -> RegimeType:
    if df is None or len(df) < lookback // 2:
        return "ranging"

    closes = df["close"].tail(lookback)
    H = hurst_exponent(closes)

    if H > 0.55:
        return "trending"
    elif H < 0.45:
        return "mean_reverting"
    return "ranging"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_regime(df: pd.DataFrame, lookback: int = 100) -> RegimeType:
    """Detect market regime. ADX primary, Hurst fallback."""
    return detect_regime_adx(df, lookback)


REGIME_STRATEGY_MAP = {
    "trending": {"EMA 9/21 Cross", "MACD Cross", "Momentum Breakout", "Donchian Breakout", "MACD Trend"},
    "mean_reverting": {"RSI 14 Reversal", "BB Reversion", "RSI Divergence"},
    "ranging": set(),
}


def filter_signals_by_regime(signals: list, regime: RegimeType) -> list:
    """Keep only signals whose strategy matches the current regime."""
    allowed = REGIME_STRATEGY_MAP.get(regime, set())

    if not allowed:
        logger.info("Regime=ranging — suppressing all signals")
        return []

    filtered = []
    for signal in signals:
        indicator = signal.indicator or ""
        matched = any(strategy in indicator for strategy in allowed)
        if matched:
            filtered.append(signal)
        else:
            logger.debug(
                "Regime=%s — suppressing %s signal (%s)",
                regime, signal.action, indicator,
            )

    return filtered


# ---------------------------------------------------------------------------
# Chop detector
# ---------------------------------------------------------------------------


def is_choppy(df: pd.DataFrame) -> bool:
    """Detect choppy market using ADX + ATR + BB width. Returns True to suppress."""
    if df is None or len(df) < 50:
        return False

    close = df["close"].tail(50)
    high = df["high"].tail(50)
    low = df["low"].tail(50)

    chop_count = 0

    # 1. ADX < 20
    try:
        from ta.trend import ADXIndicator
        adx = ADXIndicator(high=high, low=low, close=close, window=14)
        adx_val = adx.adx().iloc[-1]
        if not np.isnan(adx_val) and adx_val < 20:
            chop_count += 1
    except Exception:
        pass

    # 2. ATR percentile < 30th
    try:
        tr = np.maximum(
            high.values[1:] - low.values[1:],
            np.maximum(
                np.abs(high.values[1:] - close.values[:-1]),
                np.abs(low.values[1:] - close.values[:-1])
            )
        )
        if len(tr) >= 14:
            atr_current = np.mean(tr[-14:])
            atrs = [np.mean(tr[i:i+14]) for i in range(len(tr)-14)]
            if atrs:
                atr_pct = (np.array(atrs) < atr_current).mean() * 100
                if atr_pct < 30:
                    chop_count += 1
    except Exception:
        pass

    # 3. Bollinger Band Width < 25th percentile
    try:
        from ta.volatility import BollingerBands
        bb = BollingerBands(close=close, window=20, window_dev=2)
        bb_width = (bb.bollinger_hband() - bb.bollinger_lband()) / bb.bollinger_mavg()
        bbw_current = bb_width.iloc[-1]
        if not np.isnan(bbw_current):
            bbw_pct = (bb_width < bbw_current).sum() / len(bb_width)
            if bbw_pct < 0.25:
                chop_count += 1
    except Exception:
        pass

    return chop_count >= 2

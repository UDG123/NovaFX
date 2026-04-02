"""
Hurst Exponent regime filter for NovaFX.

H > 0.55  → Trending market   — allow trend-following signals (EMA, MACD)
H < 0.45  → Mean-reverting    — allow mean-reversion signals (BB, RSI)
0.45-0.55 → Random walk       — suppress ALL signals, no edge
"""
import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

RegimeType = Literal["trending", "mean_reverting", "ranging"]


def hurst_exponent(series: pd.Series, min_lags: int = 2, max_lags: int = 20) -> float:
    """Calculate the Hurst Exponent using R/S analysis.

    Returns a float between 0 and 1:
      H > 0.55 -> trending
      H < 0.45 -> mean-reverting
      0.45-0.55 -> random walk (ranging)
    """
    if len(series) < max_lags * 2:
        return 0.5  # not enough data — treat as random walk

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


def detect_regime(df: pd.DataFrame, lookback: int = 100) -> RegimeType:
    """Detect the current market regime from OHLCV data.

    Uses the last `lookback` closes to compute Hurst exponent.
    """
    if df is None or len(df) < lookback // 2:
        return "ranging"  # not enough data — play it safe

    closes = df["close"].tail(lookback)
    H = hurst_exponent(closes)

    if H > 0.55:
        regime = "trending"
    elif H < 0.45:
        regime = "mean_reverting"
    else:
        regime = "ranging"

    logger.debug("Hurst=%.3f → regime=%s", H, regime)
    return regime


# Which strategies are valid in each regime
REGIME_STRATEGY_MAP = {
    "trending": {"EMA 9/21 Cross", "MACD Cross"},
    "mean_reverting": {"RSI 14 Reversal", "BB Reversion"},
    "ranging": set(),  # suppress all signals
}


def filter_signals_by_regime(
    signals: list,
    regime: RegimeType,
) -> list:
    """Keep only signals whose strategy matches the current regime.

    If regime is 'ranging', all signals are suppressed.
    """
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

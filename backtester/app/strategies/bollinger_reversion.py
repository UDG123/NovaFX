"""Bollinger Band Reversion strategy for the backtester.

BUY when price touches or crosses below the lower band.
SELL when price touches or crosses above the upper band.
Uses 20-period SMA with 2 standard deviation bands.
"""

from typing import Optional

import numpy as np
import pandas as pd
from ta.volatility import BollingerBands


def generate_signals(df: pd.DataFrame) -> list[dict]:
    """Scan OHLCV data and return entry/exit signal dicts for the backtest engine.

    Each signal: {"action": "BUY"|"SELL", "entry_idx": int, "exit_idx": int}
    Entry when price touches a band; exit when price reverts to the opposite band
    or the SMA (middle band).
    """
    if len(df) < 20:
        return []

    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    mid = bb.bollinger_mavg()

    signals: list[dict] = []
    i = 20  # start after indicator warm-up

    while i < len(df):
        close = df["close"].iloc[i]
        u = upper.iloc[i]
        l = lower.iloc[i]

        if np.isnan(u) or np.isnan(l):
            i += 1
            continue

        action: Optional[str] = None
        if close <= l:
            action = "BUY"
        elif close >= u:
            action = "SELL"

        if action is None:
            i += 1
            continue

        # Find exit: price reverts to middle band
        for j in range(i + 1, len(df)):
            exit_close = df["close"].iloc[j]
            m = mid.iloc[j]
            if np.isnan(m):
                continue
            if action == "BUY" and exit_close >= m:
                signals.append({"action": action, "entry_idx": i, "exit_idx": j})
                i = j + 1
                break
            elif action == "SELL" and exit_close <= m:
                signals.append({"action": action, "entry_idx": i, "exit_idx": j})
                i = j + 1
                break
        else:
            i += 1

    return signals

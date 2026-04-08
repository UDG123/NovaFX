"""Momentum Breakout: N-bar high/low breakout with EMA trend + RSI filter."""

import numpy as np
import pandas as pd
from typing import Any

from .base import BaseStrategy


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class MomentumBreakoutStrategy(BaseStrategy):
    name = "momentum_breakout"

    def get_default_params(self) -> dict[str, Any]:
        return {
            "lookback": 20,
            "ema_fast": 20,
            "ema_slow": 50,
            "rsi_overbought": 80,
            "rsi_oversold": 20,
        }

    def get_param_grid(self) -> dict[str, list[Any]]:
        return {
            "lookback": [10, 15, 20, 25, 30],
            "ema_fast": [15, 20, 25],
            "ema_slow": [40, 50, 60],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        lb = self.params.get("lookback", 20)
        ef = self.params.get("ema_fast", 20)
        es = self.params.get("ema_slow", 50)
        rsi_ob = self.params.get("rsi_overbought", 80)
        rsi_os = self.params.get("rsi_oversold", 20)

        ema_f = close.ewm(span=ef, adjust=False).mean()
        ema_s = close.ewm(span=es, adjust=False).mean()
        rsi = _rsi(close)

        high_n = high.rolling(lb).max().shift(1)
        low_n = low.rolling(lb).min().shift(1)

        long_entry = (close > high_n) & (ema_f > ema_s) & (rsi < rsi_ob)
        short_entry = (close < low_n) & (ema_f < ema_s) & (rsi > rsi_os)

        result = self._init_result(data)
        result.loc[long_entry, "signal"] = 1
        result.loc[short_entry, "signal"] = -1
        return self.apply_cooldown(self._fill_stops(data, result))

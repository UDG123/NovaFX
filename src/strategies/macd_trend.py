"""MACD Trend: MACD crossover with SMA trend filter. Best performer (100/100 MC)."""

import numpy as np
import pandas as pd
from typing import Any

from .base import BaseStrategy


class MACDTrendStrategy(BaseStrategy):
    name = "macd_trend"

    def get_default_params(self) -> dict[str, Any]:
        return {"macd_fast": 12, "macd_slow": 26, "macd_signal": 9, "trend_period": 50}

    def get_param_grid(self) -> dict[str, list[Any]]:
        return {
            "macd_fast": [8, 12, 16],
            "macd_slow": [21, 26, 30],
            "macd_signal": [7, 9, 12],
            "trend_period": [40, 50, 60],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        fast = self.params.get("macd_fast", 12)
        slow = self.params.get("macd_slow", 26)
        sig_p = self.params.get("macd_signal", 9)
        tp = self.params.get("trend_period", 50)

        ema_f = close.ewm(span=fast, adjust=False).mean()
        ema_s = close.ewm(span=slow, adjust=False).mean()
        macd = ema_f - ema_s
        signal = macd.ewm(span=sig_p, adjust=False).mean()
        sma = close.rolling(tp).mean()

        cross_up = (macd > signal) & (macd.shift(1) <= signal.shift(1))
        cross_down = (macd < signal) & (macd.shift(1) >= signal.shift(1))

        long_entry = cross_up & (close > sma)
        short_entry = cross_down & (close < sma)

        result = self._init_result(data)
        result.loc[long_entry, "signal"] = 1
        result.loc[short_entry, "signal"] = -1
        return self.apply_cooldown(self._fill_stops(data, result))

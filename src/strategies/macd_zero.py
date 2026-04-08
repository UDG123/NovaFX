"""MACD Zero-Line Cross: MACD signal-line cross with zero-line filter."""

import numpy as np
import pandas as pd
from typing import Any

from .base import BaseStrategy


class MACDZeroStrategy(BaseStrategy):
    name = "macd_zero"

    def get_default_params(self) -> dict[str, Any]:
        return {"fast": 12, "slow": 26, "signal": 9}

    def get_param_grid(self) -> dict[str, list[Any]]:
        return {
            "fast": [8, 12, 16],
            "slow": [21, 26, 30],
            "signal": [7, 9, 12],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        f = self.params.get("fast", 12)
        s = self.params.get("slow", 26)
        sig = self.params.get("signal", 9)

        ema_f = close.ewm(span=f, adjust=False).mean()
        ema_s = close.ewm(span=s, adjust=False).mean()
        macd = ema_f - ema_s
        signal_line = macd.ewm(span=sig, adjust=False).mean()

        cross_up = (macd > signal_line) & (macd.shift(1) <= signal_line.shift(1))
        cross_down = (macd < signal_line) & (macd.shift(1) >= signal_line.shift(1))

        # Zero-line filter: only BUY above zero, SELL below zero
        long_entry = cross_up & (macd >= 0)
        short_entry = cross_down & (macd <= 0)

        result = self._init_result(data)
        result.loc[long_entry, "signal"] = 1
        result.loc[short_entry, "signal"] = -1
        return self.apply_cooldown(self._fill_stops(data, result))

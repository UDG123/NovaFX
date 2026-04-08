"""Donchian Channel Breakout (Turtle-style): close above/below N-bar high/low."""

import numpy as np
import pandas as pd
from typing import Any

from .base import BaseStrategy


class DonchianBreakoutStrategy(BaseStrategy):
    name = "donchian_breakout"

    def get_default_params(self) -> dict[str, Any]:
        return {"entry_period": 20, "min_width": 0.002}

    def get_param_grid(self) -> dict[str, list[Any]]:
        return {
            "entry_period": [15, 20, 25, 30],
            "min_width": [0.001, 0.002, 0.003],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        high = data["high"]
        low = data["low"]

        ep = self.params.get("entry_period", 20)
        mw = self.params.get("min_width", 0.002)

        upper = high.rolling(ep).max().shift(1)
        lower = low.rolling(ep).min().shift(1)
        width = (upper - lower) / lower

        long_entry = (close > upper) & (width >= mw)
        short_entry = (close < lower) & (width >= mw)

        result = self._init_result(data)
        result.loc[long_entry, "signal"] = 1
        result.loc[short_entry, "signal"] = -1
        return self.apply_cooldown(self._fill_stops(data, result))

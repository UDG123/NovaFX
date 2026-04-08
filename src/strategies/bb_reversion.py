"""Bollinger Band Mean Reversion with trend gate. Blocked on ETH, SOL."""

import numpy as np
import pandas as pd
from typing import Any

from .base import BaseStrategy


class BBReversionStrategy(BaseStrategy):
    name = "bb_reversion"

    def get_default_params(self) -> dict[str, Any]:
        return {"bb_period": 20, "bb_std": 2.0}

    def get_param_grid(self) -> dict[str, list[Any]]:
        return {
            "bb_period": [15, 20, 25],
            "bb_std": [1.5, 2.0, 2.5],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        per = self.params.get("bb_period", 20)
        sd = self.params.get("bb_std", 2.0)

        mid = close.rolling(per).mean()
        std = close.rolling(per).std()
        upper = mid + sd * std
        lower = mid - sd * std
        bw = (upper - lower) / mid

        # Reject expanding bands (trending, not reverting)
        bw_prev = bw.shift(5)
        expanding = bw > bw_prev * 1.3
        # Reject strong slope
        slope = ((mid - mid.shift(5)) / mid).abs()
        trending = slope > 0.005

        long_entry = (close <= lower) & ~expanding & ~trending
        short_entry = (close >= upper) & ~expanding & ~trending

        result = self._init_result(data)
        result.loc[long_entry, "signal"] = 1
        result.loc[short_entry, "signal"] = -1
        return self.apply_cooldown(self._fill_stops(data, result))

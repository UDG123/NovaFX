"""EMA 9/21 Cross strategy with slope confirmation."""

import numpy as np
import pandas as pd
from typing import Any

from .base import BaseStrategy


class EMACrossStrategy(BaseStrategy):
    name = "ema_cross"

    def get_default_params(self) -> dict[str, Any]:
        return {"fast_period": 9, "slow_period": 21}

    def get_param_grid(self) -> dict[str, list[Any]]:
        return {
            "fast_period": [7, 9, 12],
            "slow_period": [18, 21, 26],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        fp = self.params.get("fast_period", 9)
        sp = self.params.get("slow_period", 21)

        ema_f = close.ewm(span=fp, adjust=False).mean()
        ema_s = close.ewm(span=sp, adjust=False).mean()

        # Crossover detection
        cross_up = (ema_f > ema_s) & (ema_f.shift(1) <= ema_s.shift(1))
        cross_down = (ema_f < ema_s) & (ema_f.shift(1) >= ema_s.shift(1))

        # Slope confirmation: slow EMA must move in signal direction
        slope = ema_s - ema_s.shift(3)
        long_entry = cross_up & (slope >= 0)
        short_entry = cross_down & (slope <= 0)

        result = self._init_result(data)
        result.loc[long_entry, "signal"] = 1
        result.loc[short_entry, "signal"] = -1
        return self.apply_cooldown(self._fill_stops(data, result))

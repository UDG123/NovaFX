"""RSI Adaptive: RSI with SMA50-based adaptive thresholds. Forex only."""

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


class RSIAdaptiveStrategy(BaseStrategy):
    name = "rsi_adaptive"

    def get_default_params(self) -> dict[str, Any]:
        return {"rsi_period": 14, "buy_base": 30, "sell_base": 70}

    def get_param_grid(self) -> dict[str, list[Any]]:
        return {
            "rsi_period": [10, 14, 20],
            "buy_base": [25, 30, 35],
            "sell_base": [65, 70, 75],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        rp = self.params.get("rsi_period", 14)
        bb = self.params.get("buy_base", 30)
        sb = self.params.get("sell_base", 70)

        rsi = _rsi(close, rp)
        sma50 = close.rolling(50).mean()

        # Adaptive thresholds based on trend context
        above_sma = close > sma50
        buy_thresh = pd.Series(bb, index=data.index)
        sell_thresh = pd.Series(sb, index=data.index)
        buy_thresh[above_sma] = bb + 10
        sell_thresh[above_sma] = sb + 10
        buy_thresh[~above_sma] = bb - 10
        sell_thresh[~above_sma] = sb - 10

        long_entry = rsi < buy_thresh
        short_entry = rsi > sell_thresh

        result = self._init_result(data)
        result.loc[long_entry, "signal"] = 1
        result.loc[short_entry, "signal"] = -1
        return self.apply_cooldown(self._fill_stops(data, result))

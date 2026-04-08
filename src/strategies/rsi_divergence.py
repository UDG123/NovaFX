"""RSI Divergence with SMA50 gate. FIXED: ranging regime only + directional filter."""

import numpy as np
import pandas as pd
from typing import Any

from .base import BaseStrategy, StrategyConfig


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1.0 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class RSIDivergenceStrategy(BaseStrategy):
    name = "rsi_divergence"

    def __init__(self, config: StrategyConfig | None = None):
        if config is None:
            config = StrategyConfig(
                name=self.name,
                params=self.get_default_params(),
                allowed_regimes=["ranging", "mean_reverting"],
            )
        super().__init__(config)

    def get_default_params(self) -> dict[str, Any]:
        return {"rsi_period": 14, "lookback": 30}

    def get_param_grid(self) -> dict[str, list[Any]]:
        return {
            "rsi_period": [10, 14, 21],
            "lookback": [20, 30, 40],
        }

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        close = data["close"]
        rp = self.params.get("rsi_period", 14)
        lb = self.params.get("lookback", 30)

        rsi = _rsi(close, rp)
        sma50 = close.rolling(50).mean()
        c = close.values
        rv = rsi.values

        result = self._init_result(data)

        # Scan for divergences bar-by-bar in the last section
        for i in range(lb + 4, len(c)):
            if np.any(np.isnan(rv[i - lb:i])):
                continue
            w = c[i - lb:i]
            rw = rv[i - lb:i]

            # Find swing lows/highs
            lows, highs = [], []
            for j in range(2, len(w) - 2):
                if w[j] < w[j-1] and w[j] < w[j-2] and w[j] < w[j+1] and w[j] < w[j+2]:
                    lows.append(j)
                if w[j] > w[j-1] and w[j] > w[j-2] and w[j] > w[j+1] and w[j] > w[j+2]:
                    highs.append(j)

            price = c[i]
            s50 = sma50.iloc[i]
            if np.isnan(s50):
                continue

            # Bullish divergence: lower low in price, higher low in RSI
            # Only BUY below SMA50
            if len(lows) >= 2 and price < s50:
                i1, i2 = lows[-2], lows[-1]
                if w[i2] < w[i1] and rw[i2] > rw[i1] and rw[i2] < 40:
                    result.iloc[i, result.columns.get_loc("signal")] = 1

            # Bearish divergence: higher high in price, lower high in RSI
            # Only SELL above SMA50
            if len(highs) >= 2 and price > s50:
                i1, i2 = highs[-2], highs[-1]
                if w[i2] > w[i1] and rw[i2] < rw[i1] and rw[i2] > 60:
                    result.iloc[i, result.columns.get_loc("signal")] = -1

        return self.apply_cooldown(self._fill_stops(data, result))

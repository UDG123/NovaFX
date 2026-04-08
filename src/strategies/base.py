"""Base strategy class defining the standard interface for all NovaFX strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class StrategyConfig:
    """Configuration for a strategy."""
    name: str
    params: dict[str, Any]
    allowed_regimes: list[str] = field(default_factory=lambda: ["trending", "ranging"])
    sl_atr_mult: float = 1.5
    tp_atr_mult: float = 3.0
    cooldown_bars: int = 0


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies.

    Subclasses must implement:
    - generate_signals(): vectorized signal generation returning a DataFrame
    - get_default_params(): default parameter values
    - get_param_grid(): parameter grid for optimization
    """

    name: str = "base"

    def __init__(self, config: StrategyConfig | None = None):
        if config is None:
            config = StrategyConfig(name=self.name, params=self.get_default_params())
        self.config = config
        self.params = config.params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals from OHLCV data.

        Args:
            data: DataFrame with columns: open, high, low, close, volume

        Returns:
            DataFrame with columns: signal (1/-1/0), entry_price, stop_loss,
            take_profit, confidence
        """

    @abstractmethod
    def get_default_params(self) -> dict[str, Any]:
        """Return default parameters for this strategy."""

    @abstractmethod
    def get_param_grid(self) -> dict[str, list[Any]]:
        """Return parameter grid for optimization."""

    def _init_result(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create empty result DataFrame with correct columns."""
        result = pd.DataFrame(index=data.index)
        result["signal"] = 0
        result["entry_price"] = data["close"]
        result["stop_loss"] = np.nan
        result["take_profit"] = np.nan
        result["confidence"] = 0.0
        return result

    def _fill_stops(self, data: pd.DataFrame, result: pd.DataFrame) -> pd.DataFrame:
        """Calculate ATR-based stops for all signal rows."""
        signal_mask = result["signal"] != 0
        if not signal_mask.any():
            return result

        # Vectorized ATR
        high = data["high"]
        low = data["low"]
        prev_close = data["close"].shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(14, min_periods=1).mean()

        close = data["close"]
        sl_mult = self.config.sl_atr_mult
        tp_mult = self.config.tp_atr_mult

        long_mask = signal_mask & (result["signal"] == 1)
        short_mask = signal_mask & (result["signal"] == -1)

        result.loc[long_mask, "stop_loss"] = close[long_mask] - sl_mult * atr[long_mask]
        result.loc[long_mask, "take_profit"] = close[long_mask] + tp_mult * atr[long_mask]
        result.loc[short_mask, "stop_loss"] = close[short_mask] + sl_mult * atr[short_mask]
        result.loc[short_mask, "take_profit"] = close[short_mask] - tp_mult * atr[short_mask]
        result.loc[signal_mask, "confidence"] = 1.0

        return result

    def apply_cooldown(self, result: pd.DataFrame) -> pd.DataFrame:
        """Suppress signals that fire within cooldown_bars of each other."""
        if self.config.cooldown_bars <= 0:
            return result
        out = result.copy()
        last_idx = -self.config.cooldown_bars - 1
        for i in range(len(out)):
            if out["signal"].iloc[i] != 0:
                if i - last_idx <= self.config.cooldown_bars:
                    out.iloc[i, out.columns.get_loc("signal")] = 0
                else:
                    last_idx = i
        return out

    def validate_regime(self, regime: str) -> bool:
        """Check if strategy is allowed in current regime."""
        return regime in self.config.allowed_regimes

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(params={self.params})"

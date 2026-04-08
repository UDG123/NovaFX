"""Position sizing: Kelly, volatility-target, regime-aware, correlation-aware, drawdown scaling."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


class SizingMethod(Enum):
    FIXED_FRACTIONAL = "fixed_fractional"
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    VOLATILITY_TARGET = "volatility_target"
    ADAPTIVE = "adaptive"


@dataclass
class SizingConfig:
    method: SizingMethod = SizingMethod.ADAPTIVE
    risk_per_trade: float = 0.02
    max_position_pct: float = 0.20
    kelly_fraction: float = 0.5
    min_kelly_trades: int = 30
    target_volatility: float = 0.15
    vol_lookback: int = 20
    regime_multipliers: dict[str, float] = field(default_factory=lambda: {
        "trending": 1.0, "bull": 1.0,
        "ranging": 0.8, "mean_reverting": 0.8,
        "volatile": 0.5, "bear": 0.7,
        "unknown": 0.7,
    })
    dd_threshold: float = 0.10
    dd_max_reduction: float = 0.50
    dd_max: float = 0.25
    max_correlated_exposure: float = 0.30
    correlation_threshold: float = 0.7


@dataclass
class SizingResult:
    position_size: float
    position_units: float
    dollar_amount: float
    base_size: float
    kelly_adjustment: float
    vol_adjustment: float
    regime_adjustment: float
    dd_adjustment: float
    correlation_adjustment: float
    capped: bool
    method_used: str
    notes: list[str]


class PositionSizer:
    """Intelligent position sizing with multiple adjustment layers."""

    def __init__(self, config: SizingConfig | None = None):
        self.config = config or SizingConfig()

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: float,
        strategy_stats: dict[str, Any] | None = None,
        current_drawdown: float = 0.0,
        regime: str = "unknown",
        asset_volatility: float | None = None,
        existing_positions: dict[str, float] | None = None,
        correlations: dict[str, float] | None = None,
        symbol: str = "",
    ) -> SizingResult:
        notes: list[str] = []

        # 1. Base size from risk per trade
        stop_dist = abs(entry_price - stop_loss) / entry_price if entry_price > 0 else 0.02
        if stop_dist == 0:
            stop_dist = 0.02
        base_size = self.config.risk_per_trade / stop_dist
        notes.append(f"Base: {base_size:.2%} (risk={self.config.risk_per_trade:.1%}, stop={stop_dist:.1%})")

        # 2. Kelly adjustment
        kelly_adj = 1.0
        if (self.config.method in (SizingMethod.KELLY, SizingMethod.HALF_KELLY, SizingMethod.ADAPTIVE)
                and strategy_stats and strategy_stats.get("n_trades", 0) >= self.config.min_kelly_trades):
            k = _kelly(strategy_stats)
            if k > 0:
                kelly_adj = min(k / base_size, 1.5)
                kelly_adj = max(kelly_adj, 0.3)
                notes.append(f"Kelly: {k:.2%} -> adj={kelly_adj:.2f}x")
            else:
                kelly_adj = 0.5
                notes.append("Kelly negative -> 0.5x")

        # 3. Volatility adjustment
        vol_adj = 1.0
        if (self.config.method in (SizingMethod.VOLATILITY_TARGET, SizingMethod.ADAPTIVE)
                and asset_volatility is not None and asset_volatility > 0):
            vol_adj = float(np.clip(self.config.target_volatility / asset_volatility, 0.3, 2.0))
            notes.append(f"Vol: {self.config.target_volatility:.0%}/{asset_volatility:.0%}={vol_adj:.2f}x")

        # 4. Regime adjustment
        regime_adj = self.config.regime_multipliers.get(regime, 0.7)
        if regime_adj != 1.0:
            notes.append(f"Regime '{regime}': {regime_adj:.1f}x")

        # 5. Drawdown scaling
        dd_adj = 1.0
        if current_drawdown >= self.config.dd_max:
            dd_adj = 0.0
            notes.append(f"DD={current_drawdown:.1%} >= max -> HALT")
        elif current_drawdown >= self.config.dd_threshold:
            dd_range = self.config.dd_max - self.config.dd_threshold
            progress = (current_drawdown - self.config.dd_threshold) / dd_range
            dd_adj = 1.0 - progress * self.config.dd_max_reduction
            notes.append(f"DD={current_drawdown:.1%}: {dd_adj:.2f}x")

        # 6. Correlation adjustment
        corr_adj = 1.0
        if existing_positions and correlations:
            corr_exp = sum(
                sz for sym, sz in existing_positions.items()
                if abs(correlations.get(sym, 0)) >= self.config.correlation_threshold
            )
            if corr_exp > 0:
                remaining = self.config.max_correlated_exposure - corr_exp
                if remaining <= 0:
                    corr_adj = 0.0
                    notes.append(f"Correlated exp {corr_exp:.1%} >= max -> BLOCK")
                else:
                    corr_adj = min(remaining / base_size, 1.0)
                    notes.append(f"Corr adj: {corr_adj:.2f}x")

        # Combine
        final = base_size * kelly_adj * vol_adj * regime_adj * dd_adj * corr_adj
        capped = False
        if final > self.config.max_position_pct:
            final = self.config.max_position_pct
            capped = True
            notes.append(f"Capped at {self.config.max_position_pct:.0%}")

        dollar = equity * final
        units = dollar / entry_price if entry_price > 0 else 0

        return SizingResult(
            position_size=final, position_units=units, dollar_amount=dollar,
            base_size=base_size, kelly_adjustment=kelly_adj, vol_adjustment=vol_adj,
            regime_adjustment=regime_adj, dd_adjustment=dd_adj,
            correlation_adjustment=corr_adj, capped=capped,
            method_used=self.config.method.value, notes=notes,
        )


def _kelly(stats: dict) -> float:
    """Kelly% = W - (1-W)/R, with fractional Kelly."""
    wr = stats.get("win_rate", 0.5)
    aw = stats.get("avg_win", 0.02)
    al = abs(stats.get("avg_loss", 0.01))
    if al == 0:
        return 0
    r = aw / al
    return max(0, (wr - (1 - wr) / r) * 0.5)


# Convenience functions

def fixed_fractional_size(equity: float, entry_price: float,
                          stop_loss: float, risk_pct: float = 0.02) -> float:
    sd = abs(entry_price - stop_loss) / entry_price if entry_price > 0 else 0.02
    return risk_pct / sd if sd > 0 else 0


def kelly_size(win_rate: float, avg_win: float, avg_loss: float,
               fraction: float = 0.5) -> float:
    if avg_loss == 0:
        return 0
    r = avg_win / abs(avg_loss)
    k = win_rate - (1 - win_rate) / r
    return max(0, k * fraction)


def volatility_adjusted_size(base_size: float, asset_vol: float,
                              target_vol: float = 0.15) -> float:
    if asset_vol == 0:
        return base_size
    return base_size * float(np.clip(target_vol / asset_vol, 0.3, 2.0))

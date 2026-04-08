"""Strategy registry and factory.

Usage:
    from src.strategies import get_strategy, STRATEGY_REGISTRY
    strategy = get_strategy('macd_trend')
    signals = strategy.generate_signals(data)
"""

from typing import Any, Type

from .base import BaseStrategy, StrategyConfig
from .ema_cross import EMACrossStrategy
from .momentum_breakout import MomentumBreakoutStrategy
from .donchian_breakout import DonchianBreakoutStrategy
from .macd_trend import MACDTrendStrategy
from .macd_zero import MACDZeroStrategy
from .bb_reversion import BBReversionStrategy
from .rsi_adaptive import RSIAdaptiveStrategy
from .rsi_divergence import RSIDivergenceStrategy

STRATEGY_REGISTRY: dict[str, Type[BaseStrategy]] = {
    "ema_cross": EMACrossStrategy,
    "momentum_breakout": MomentumBreakoutStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
    "macd_trend": MACDTrendStrategy,
    "macd_zero": MACDZeroStrategy,
    "bb_reversion": BBReversionStrategy,
    "rsi_adaptive": RSIAdaptiveStrategy,
    "rsi_divergence": RSIDivergenceStrategy,
}


def get_strategy(name: str,
                 params: dict[str, Any] | None = None,
                 config: StrategyConfig | None = None) -> BaseStrategy:
    """Factory: get strategy instance by name."""
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ValueError(f"Unknown strategy: {name}. Available: {available}")

    cls = STRATEGY_REGISTRY[name]

    if config is not None:
        return cls(config)
    if params is not None:
        defaults = cls(None).get_default_params()
        merged = {**defaults, **params}
        return cls(StrategyConfig(name=name, params=merged))
    return cls()


def list_strategies() -> dict[str, dict[str, Any]]:
    """List all strategies with defaults and param grids."""
    result = {}
    for name, cls in STRATEGY_REGISTRY.items():
        inst = cls()
        result[name] = {
            "default_params": inst.get_default_params(),
            "param_grid": inst.get_param_grid(),
            "allowed_regimes": inst.config.allowed_regimes,
        }
    return result


__all__ = [
    "BaseStrategy", "StrategyConfig",
    "EMACrossStrategy", "MomentumBreakoutStrategy", "DonchianBreakoutStrategy",
    "MACDTrendStrategy", "MACDZeroStrategy", "BBReversionStrategy",
    "RSIAdaptiveStrategy", "RSIDivergenceStrategy",
    "STRATEGY_REGISTRY", "get_strategy", "list_strategies",
]

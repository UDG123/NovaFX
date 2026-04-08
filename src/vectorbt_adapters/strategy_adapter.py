"""Adapt src/strategies classes for VectorBT's vectorized backtesting."""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from src.strategies import get_strategy, STRATEGY_REGISTRY
from src.execution.volume_handler import VolumeHandler


@dataclass
class VBTBacktestResult:
    """Results from VectorBT backtest."""
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    portfolio: vbt.Portfolio


class VBTStrategyAdapter:
    """Runs src/strategies through VectorBT.

    Usage:
        adapter = VBTStrategyAdapter('macd_trend', data, 'BTC-USD')
        result = adapter.run_backtest()
    """

    def __init__(self, strategy_name: str, data: pd.DataFrame, symbol: str = "UNKNOWN"):
        if strategy_name not in STRATEGY_REGISTRY:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        self.strategy_name = strategy_name
        self.data = data
        self.symbol = symbol
        self.volume_handler = VolumeHandler(symbol)
        self.freq = "1h"
        if isinstance(data.index, pd.DatetimeIndex) and len(data) > 1:
            delta = data.index[1] - data.index[0]
            self.freq = "1h" if delta <= pd.Timedelta(hours=1) else "1d"

    def generate_signals(self, params: dict[str, Any] | None = None) -> dict[str, pd.Series]:
        """Generate entry/exit signals via strategy module."""
        strategy = get_strategy(self.strategy_name, params=params)
        signals_df = strategy.generate_signals(self.data)
        sig = signals_df["signal"]
        return {
            "long_entries": sig == 1,
            "long_exits": sig == -1,
            "short_entries": sig == -1,
            "short_exits": sig == 1,
        }

    def get_param_grid(self) -> dict[str, list]:
        return get_strategy(self.strategy_name).get_param_grid()

    def get_default_params(self) -> dict[str, Any]:
        return get_strategy(self.strategy_name).get_default_params()

    def run_backtest(self, params: dict[str, Any] | None = None,
                     sl_atr: float = 1.5, tp_atr: float = 3.0,
                     fees: float = 0.001, slippage: float | pd.Series | None = None,
                     init_cash: float = 10000) -> VBTBacktestResult:
        """Run VectorBT backtest with ATR stops and volume-aware slippage."""
        signals = self.generate_signals(params)

        atr = vbt.ATR.run(
            self.data["high"], self.data["low"], self.data["close"], window=14
        ).atr

        sl_stop = sl_atr * atr / self.data["close"]
        tp_stop = tp_atr * atr / self.data["close"]

        if slippage is None:
            slippage = self.volume_handler.get_slippage(
                trade_size_usd=init_cash * 0.02, data=self.data,
            )

        pf = vbt.Portfolio.from_signals(
            close=self.data["close"],
            entries=signals["long_entries"],
            exits=signals["long_exits"],
            short_entries=signals["short_entries"],
            short_exits=signals["short_exits"],
            sl_stop=sl_stop,
            tp_stop=tp_stop,
            fees=fees,
            slippage=slippage,
            init_cash=init_cash,
            freq=self.freq,
            upon_opposite_entry="close",
        )

        trades = pf.trades
        n = int(trades.count())

        return VBTBacktestResult(
            total_return=float(pf.total_return()),
            sharpe_ratio=float(pf.sharpe_ratio()) if not np.isnan(pf.sharpe_ratio()) else 0.0,
            sortino_ratio=float(pf.sortino_ratio()) if not np.isnan(pf.sortino_ratio()) else 0.0,
            max_drawdown=float(pf.max_drawdown()),
            win_rate=float(trades.win_rate()) if n > 0 else 0.0,
            profit_factor=float(trades.profit_factor()) if n > 0 else 0.0,
            num_trades=n,
            avg_trade_return=float(trades.returns.mean()) if n > 0 else 0.0,
            portfolio=pf,
        )


def vbt_backtest(strategy_name: str, data: pd.DataFrame, symbol: str,
                 params: dict[str, Any] | None = None, **kwargs) -> VBTBacktestResult:
    """Convenience: quick VBT backtest."""
    return VBTStrategyAdapter(strategy_name, data, symbol).run_backtest(params=params, **kwargs)

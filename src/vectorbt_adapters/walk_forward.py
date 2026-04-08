"""Walk-forward optimization using VectorBT speed."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .param_optimizer import VBTParamOptimizer
from .strategy_adapter import VBTStrategyAdapter


@dataclass
class WFWindow:
    window_idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    best_params: dict[str, Any]
    is_sharpe: float
    oos_sharpe: float
    oos_return: float
    oos_trades: int


@dataclass
class WFResults:
    strategy_name: str
    symbol: str
    windows: list[WFWindow]
    oos_sharpe: float
    oos_return: float
    efficiency_ratio: float
    param_stability: dict[str, float]

    def summary(self) -> str:
        lines = [
            f"Walk-Forward: {self.strategy_name} on {self.symbol}",
            "=" * 50,
            f"Windows: {len(self.windows)}",
            f"OOS Sharpe: {self.oos_sharpe:.2f}",
            f"OOS Return: {self.oos_return:.2%}",
            f"Efficiency: {self.efficiency_ratio:.2f}",
            "", "Param Stability (CV):",
        ]
        for p, cv in self.param_stability.items():
            lines.append(f"  {p}: {cv:.3f}")
        return "\n".join(lines)


class VBTWalkForward:
    """Walk-forward optimization with VBT speed.

    Usage:
        wf = VBTWalkForward('macd_trend', data, 'BTC-USD')
        results = wf.run(n_windows=10)
    """

    def __init__(self, strategy_name: str, data: pd.DataFrame, symbol: str):
        self.strategy_name = strategy_name
        self.data = data
        self.symbol = symbol

    def run(self, n_windows: int = 10, train_pct: float = 0.7,
            purge_bars: int = 24, metric: str = "sharpe_ratio",
            min_trades: int = 5, verbose: bool = True) -> WFResults:

        window_size = len(self.data) // n_windows
        train_size = int(window_size * train_pct)
        windows: list[WFWindow] = []

        for i in range(n_windows):
            start = i * window_size
            train_end = start + train_size
            test_start = train_end + purge_bars
            test_end = min(start + window_size, len(self.data))

            if test_end <= test_start + 20:
                continue

            train_df = self.data.iloc[start:train_end]
            test_df = self.data.iloc[test_start:test_end]

            if verbose:
                ts = train_df.index[0]
                te = test_df.index[-1]
                t_str = ts.date() if hasattr(ts, "date") else ts
                e_str = te.date() if hasattr(te, "date") else te
                print(f"\nWindow {i+1}/{n_windows}: train {t_str} -> test {e_str}")

            opt = VBTParamOptimizer(self.strategy_name, train_df, self.symbol)
            try:
                opt_result = opt.optimize(metric=metric, min_trades=min_trades, verbose=False)
            except ValueError:
                if verbose:
                    print("  Skipping: no valid results")
                continue

            best_params = opt_result.best_params
            is_sharpe = opt_result.best_metric_value

            adapter = VBTStrategyAdapter(self.strategy_name, test_df, self.symbol)
            oos = adapter.run_backtest(params=best_params)

            if verbose:
                print(f"  IS Sharpe: {is_sharpe:.2f}  OOS Sharpe: {oos.sharpe_ratio:.2f}  "
                      f"OOS Return: {oos.total_return:.2%}")

            windows.append(WFWindow(
                window_idx=i,
                train_start=train_df.index[0],
                train_end=train_df.index[-1],
                test_start=test_df.index[0],
                test_end=test_df.index[-1],
                best_params=best_params,
                is_sharpe=is_sharpe,
                oos_sharpe=oos.sharpe_ratio,
                oos_return=oos.total_return,
                oos_trades=oos.num_trades,
            ))

        if not windows:
            raise ValueError("No valid windows")

        avg_is = np.mean([w.is_sharpe for w in windows])
        avg_oos = np.mean([w.oos_sharpe for w in windows])
        total_ret = float(np.prod([1 + w.oos_return for w in windows]) - 1)
        eff = avg_oos / avg_is if avg_is > 0 else 0.0

        stability = {}
        all_params = [w.best_params for w in windows]
        if len(all_params) >= 2:
            for key in all_params[0]:
                vals = [p[key] for p in all_params]
                if all(isinstance(v, (int, float)) for v in vals):
                    m = np.mean(vals)
                    stability[key] = float(np.std(vals) / m) if m != 0 else 0.0

        result = WFResults(
            strategy_name=self.strategy_name, symbol=self.symbol,
            windows=windows, oos_sharpe=avg_oos, oos_return=total_ret,
            efficiency_ratio=eff, param_stability=stability,
        )

        if verbose:
            print("\n" + result.summary())

        return result

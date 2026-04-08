"""Fast parameter optimization using VectorBT."""

from dataclasses import dataclass
from itertools import product
from typing import Any

import pandas as pd

from .strategy_adapter import VBTStrategyAdapter


METRICS = [
    "total_return", "sharpe_ratio", "sortino_ratio",
    "max_drawdown", "win_rate", "profit_factor", "num_trades",
]


@dataclass
class OptimizationResult:
    strategy_name: str
    symbol: str
    best_params: dict[str, Any]
    best_metric_value: float
    metric_name: str
    all_results: pd.DataFrame
    n_tested: int
    n_valid: int


class VBTParamOptimizer:
    """Fast parameter sweep over strategy param grid.

    Usage:
        opt = VBTParamOptimizer('macd_trend', data, 'BTC-USD')
        result = opt.optimize(metric='sharpe_ratio')
    """

    def __init__(self, strategy_name: str, data: pd.DataFrame, symbol: str):
        self.strategy_name = strategy_name
        self.data = data
        self.symbol = symbol
        self.adapter = VBTStrategyAdapter(strategy_name, data, symbol)
        self.results: pd.DataFrame | None = None

    def optimize(self, param_grid: dict[str, list] | None = None,
                 metric: str = "sharpe_ratio", min_trades: int = 10,
                 verbose: bool = True, **backtest_kwargs) -> OptimizationResult:
        if param_grid is None:
            param_grid = self.adapter.get_param_grid()
        if metric not in METRICS:
            raise ValueError(f"Unknown metric: {metric}. Available: {METRICS}")

        param_names = list(param_grid.keys())
        combos = list(product(*param_grid.values()))

        if verbose:
            print(f"Testing {len(combos)} combinations...")

        results = []
        valid = 0

        for i, combo in enumerate(combos):
            params = dict(zip(param_names, combo))
            if not self._valid_combo(params):
                continue
            try:
                r = self.adapter.run_backtest(params=params, **backtest_kwargs)
                if r.num_trades < min_trades:
                    continue
                valid += 1
                results.append({
                    **params,
                    "total_return": r.total_return,
                    "sharpe_ratio": r.sharpe_ratio,
                    "sortino_ratio": r.sortino_ratio,
                    "max_drawdown": r.max_drawdown,
                    "win_rate": r.win_rate,
                    "profit_factor": r.profit_factor,
                    "num_trades": r.num_trades,
                })
            except Exception:
                continue

            if verbose and (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(combos)} tested, {valid} valid")

        if not results:
            raise ValueError("No valid results. Try reducing min_trades.")

        self.results = pd.DataFrame(results)
        ascending = metric == "max_drawdown"
        self.results = self.results.sort_values(metric, ascending=ascending)

        best = self.results.iloc[0]
        best_params = {k: best[k] for k in param_names}

        if verbose:
            print(f"\nBest {metric}: {best[metric]:.4f}")
            print(f"Best params: {best_params}")

        return OptimizationResult(
            strategy_name=self.strategy_name, symbol=self.symbol,
            best_params=best_params, best_metric_value=float(best[metric]),
            metric_name=metric, all_results=self.results,
            n_tested=len(combos), n_valid=valid,
        )

    def _valid_combo(self, p: dict) -> bool:
        if "ema_fast" in p and "ema_slow" in p and p["ema_fast"] >= p["ema_slow"]:
            return False
        if "macd_fast" in p and "macd_slow" in p and p["macd_fast"] >= p["macd_slow"]:
            return False
        if "fast" in p and "slow" in p and p["fast"] >= p["slow"]:
            return False
        return True

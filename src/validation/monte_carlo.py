"""
Monte Carlo simulation for NovaFX strategy validation.

Three simulation types:
  1. Reshuffle: Same trades, random order — tests if edge depends on sequence
  2. Skip: Remove random X% of trades — simulates missed fills / slippage
  3. Parameter perturbation: Randomize params ±X% — tests param sensitivity

Reports percentile distributions for equity, drawdown, Sharpe, and win rate.
"""
import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MCResults:
    """Monte Carlo simulation results."""
    final_equity_dist: np.ndarray
    max_drawdown_dist: np.ndarray
    sharpe_dist: np.ndarray
    win_rate_dist: np.ndarray
    n_simulations: int
    simulation_type: str

    def percentile(self, metric: str, pct: float) -> float:
        """Get percentile for a given metric."""
        arr = getattr(self, f"{metric}_dist", None)
        if arr is None or len(arr) == 0:
            return 0.0
        return float(np.percentile(arr, pct))

    def summary(self) -> dict:
        """Return 5th, 25th, 50th, 75th, 95th percentiles for key metrics."""
        pcts = [5, 25, 50, 75, 95]
        result = {}
        for metric in ["final_equity", "max_drawdown", "sharpe", "win_rate"]:
            arr = getattr(self, f"{metric}_dist", None)
            if arr is not None and len(arr) > 0:
                result[metric] = {
                    f"p{p}": float(np.percentile(arr, p)) for p in pcts
                }
                result[metric]["mean"] = float(np.mean(arr))
                result[metric]["std"] = float(np.std(arr))
            else:
                result[metric] = {f"p{p}": 0.0 for p in pcts}
        return result

    def probability_of_loss(self) -> float:
        """Probability that final equity is negative."""
        if len(self.final_equity_dist) == 0:
            return 0.0
        return float(np.mean(self.final_equity_dist < 0))

    def probability_of_drawdown_exceeding(self, threshold: float) -> float:
        """Probability of max drawdown exceeding threshold (e.g. -30%)."""
        if len(self.max_drawdown_dist) == 0:
            return 0.0
        return float(np.mean(self.max_drawdown_dist < threshold))


def _compute_equity_stats(returns: np.ndarray, initial_equity: float = 10000.0
                          ) -> tuple[float, float, float, float]:
    """Compute final equity, max drawdown %, Sharpe, and win rate from returns array."""
    if len(returns) == 0:
        return initial_equity, 0.0, 0.0, 0.0

    # Equity curve
    equity = initial_equity * np.cumprod(1 + returns / 100)
    final_eq = equity[-1] - initial_equity  # PnL in dollars

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak * 100  # Drawdown as negative %
    max_dd = float(dd.min())

    # Sharpe (annualized from trade-level returns)
    if len(returns) >= 2 and np.std(returns) > 0:
        sharpe = float(np.mean(returns) / np.std(returns) * np.sqrt(len(returns)))
    else:
        sharpe = 0.0

    # Win rate
    win_rate = float(np.sum(returns > 0) / len(returns) * 100)

    return final_eq, max_dd, sharpe, win_rate


class MonteCarloSimulator:
    """Monte Carlo simulator for strategy validation."""

    def __init__(self, n_simulations: int = 1000, random_seed: int = 42,
                 initial_equity: float = 10000.0):
        self.n_simulations = n_simulations
        self.rng = np.random.RandomState(random_seed)
        self.initial_equity = initial_equity

    def reshuffle_trades(self, trade_returns: np.ndarray) -> MCResults:
        """Randomly reorder trades to test if edge depends on trade sequence.

        If performance is similar regardless of order, the strategy has a
        genuine statistical edge rather than luck-dependent sequencing.
        """
        returns = np.asarray(trade_returns, dtype=float)
        n = len(returns)
        if n == 0:
            return self._empty_result("reshuffle")

        final_eqs = np.zeros(self.n_simulations)
        max_dds = np.zeros(self.n_simulations)
        sharpes = np.zeros(self.n_simulations)
        win_rates = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            shuffled = self.rng.permutation(returns)
            feq, mdd, sh, wr = _compute_equity_stats(shuffled, self.initial_equity)
            final_eqs[i] = feq
            max_dds[i] = mdd
            sharpes[i] = sh
            win_rates[i] = wr

        return MCResults(
            final_equity_dist=final_eqs,
            max_drawdown_dist=max_dds,
            sharpe_dist=sharpes,
            win_rate_dist=win_rates,
            n_simulations=self.n_simulations,
            simulation_type="reshuffle",
        )

    def skip_trades(self, trade_returns: np.ndarray,
                    skip_pct: float = 0.05) -> MCResults:
        """Randomly skip X% of trades to simulate missed fills.

        Tests robustness: if strategy survives missing 5% of trades,
        it's resilient to real-world execution gaps.
        """
        returns = np.asarray(trade_returns, dtype=float)
        n = len(returns)
        if n == 0:
            return self._empty_result("skip")

        n_skip = max(1, int(n * skip_pct))

        final_eqs = np.zeros(self.n_simulations)
        max_dds = np.zeros(self.n_simulations)
        sharpes = np.zeros(self.n_simulations)
        win_rates = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            # Randomly select indices to remove
            skip_idx = self.rng.choice(n, size=n_skip, replace=False)
            mask = np.ones(n, dtype=bool)
            mask[skip_idx] = False
            subset = returns[mask]
            feq, mdd, sh, wr = _compute_equity_stats(subset, self.initial_equity)
            final_eqs[i] = feq
            max_dds[i] = mdd
            sharpes[i] = sh
            win_rates[i] = wr

        return MCResults(
            final_equity_dist=final_eqs,
            max_drawdown_dist=max_dds,
            sharpe_dist=sharpes,
            win_rate_dist=win_rates,
            n_simulations=self.n_simulations,
            simulation_type=f"skip_{skip_pct:.0%}",
        )

    def parameter_perturbation(self, strategy_fn, data,
                               base_params: dict,
                               perturbation_pct: float = 0.10,
                               sl_mult: float = 1.5,
                               tp_mult: float = 3.0) -> MCResults:
        """Slightly randomize parameters each run to test sensitivity.

        For each simulation:
          1. Perturb each numeric param by ±perturbation_pct
          2. Run strategy with perturbed params
          3. Record equity stats

        strategy_fn: callable(data, params) -> list of trade PnL %
        """
        final_eqs = np.zeros(self.n_simulations)
        max_dds = np.zeros(self.n_simulations)
        sharpes = np.zeros(self.n_simulations)
        win_rates = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            perturbed = {}
            for k, v in base_params.items():
                if isinstance(v, (int, float)):
                    noise = self.rng.uniform(-perturbation_pct, perturbation_pct)
                    new_val = v * (1 + noise)
                    perturbed[k] = int(round(new_val)) if isinstance(v, int) else round(new_val, 6)
                else:
                    perturbed[k] = v

            try:
                trade_pnls = strategy_fn(data, perturbed, sl_mult, tp_mult)
                returns = np.array(trade_pnls, dtype=float)
            except Exception:
                returns = np.array([])

            feq, mdd, sh, wr = _compute_equity_stats(returns, self.initial_equity)
            final_eqs[i] = feq
            max_dds[i] = mdd
            sharpes[i] = sh
            win_rates[i] = wr

        return MCResults(
            final_equity_dist=final_eqs,
            max_drawdown_dist=max_dds,
            sharpe_dist=sharpes,
            win_rate_dist=win_rates,
            n_simulations=self.n_simulations,
            simulation_type=f"perturbation_±{perturbation_pct:.0%}",
        )

    def bootstrap_confidence(self, trade_returns: np.ndarray,
                             sample_pct: float = 0.80) -> MCResults:
        """Bootstrap resampling with replacement.

        Draws sample_pct of trades with replacement, computes stats.
        Tests stability of performance metrics.
        """
        returns = np.asarray(trade_returns, dtype=float)
        n = len(returns)
        if n == 0:
            return self._empty_result("bootstrap")

        sample_size = max(1, int(n * sample_pct))

        final_eqs = np.zeros(self.n_simulations)
        max_dds = np.zeros(self.n_simulations)
        sharpes = np.zeros(self.n_simulations)
        win_rates = np.zeros(self.n_simulations)

        for i in range(self.n_simulations):
            idx = self.rng.choice(n, size=sample_size, replace=True)
            sample = returns[idx]
            feq, mdd, sh, wr = _compute_equity_stats(sample, self.initial_equity)
            final_eqs[i] = feq
            max_dds[i] = mdd
            sharpes[i] = sh
            win_rates[i] = wr

        return MCResults(
            final_equity_dist=final_eqs,
            max_drawdown_dist=max_dds,
            sharpe_dist=sharpes,
            win_rate_dist=win_rates,
            n_simulations=self.n_simulations,
            simulation_type=f"bootstrap_{sample_pct:.0%}",
        )

    def _empty_result(self, sim_type: str) -> MCResults:
        return MCResults(
            final_equity_dist=np.array([]),
            max_drawdown_dist=np.array([]),
            sharpe_dist=np.array([]),
            win_rate_dist=np.array([]),
            n_simulations=0,
            simulation_type=sim_type,
        )

"""
Monte Carlo validation for NovaFX strategies.

Runs 4 MC simulation types on backtest trade results:
  1. Reshuffle (random trade order)
  2. Skip 5% (simulate missed fills)
  3. Bootstrap (resample with replacement)
  4. Parameter perturbation ±10%

Outputs percentile tables, robustness scores, and DD warnings.

Usage:
    python scripts/monte_carlo_validation.py              # All crypto
    python scripts/monte_carlo_validation.py --quick       # BTC only, 1 strategy
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest_harness import fetch_ohlcv
from src.optimization.walk_forward import _run_parameterized, _simulate, PARAM_GRIDS
from src.validation.monte_carlo import MonteCarloSimulator, MCResults, _compute_equity_stats

ASSETS = {
    "BTC-USD": ("BTC-USD", "crypto", 1.5, 3.5),
    "ETH-USD": ("ETH-USD", "crypto", 1.5, 3.5),
    "SOL-USD": ("SOL-USD", "crypto", 1.5, 3.5),
    "XRP-USD": ("XRP-USD", "crypto", 1.5, 3.5),
}

STRATEGIES = ["ema_cross", "macd_zero", "macd_trend", "donchian_breakout", "momentum_breakout"]

# Default params per strategy (from WFO best-of-median)
DEFAULT_PARAMS = {
    "ema_cross": {"fast_period": 9, "slow_period": 21},
    "macd_zero": {"fast": 12, "slow": 26, "signal": 9},
    "macd_trend": {"fast": 12, "slow": 26, "signal": 9, "trend_period": 50},
    "donchian_breakout": {"entry_period": 20, "min_width": 0.001},
    "momentum_breakout": {"lookback": 20, "ema_fast": 20, "ema_slow": 50},
}


def _strategy_fn_for_perturbation(strategy: str):
    """Return a callable(data, params, sl_mult, tp_mult) -> list[float]."""
    def fn(data, params, sl_mult, tp_mult):
        sigs = _run_parameterized(data, strategy, params)
        trades = _simulate(data, sigs, sl_mult, tp_mult)
        return [t.pnl_pct for t in trades]
    return fn


def compute_robustness_score(
    original_pnls: np.ndarray,
    reshuffle: MCResults,
    skip: MCResults,
    bootstrap: MCResults,
    perturbation: MCResults | None = None,
) -> tuple[float, dict]:
    """Compute 0-100 robustness score.

    Components (each 0-25 points):
      1. DD stability:    95th pct DD < 2x original DD
      2. Profit consistency: >70% of simulations profitable
      3. Sharpe stability:   median Sharpe within 50% of original
      4. Param sensitivity:  perturbation median PnL within 50% of original
    """
    orig_eq, orig_dd, orig_sharpe, orig_wr = _compute_equity_stats(original_pnls)
    details = {}

    # 1. DD stability (25 pts): compare 95th pct DD to original
    p95_dd = reshuffle.percentile("max_drawdown", 5)  # 5th pct of DD = worst 95th
    dd_ratio = abs(p95_dd / orig_dd) if orig_dd != 0 else 1.0
    if dd_ratio <= 1.5:
        dd_score = 25.0
    elif dd_ratio <= 2.0:
        dd_score = 20.0
    elif dd_ratio <= 3.0:
        dd_score = 10.0
    else:
        dd_score = 0.0
    details["dd_stability"] = {"score": dd_score, "ratio": dd_ratio, "p95_dd": p95_dd, "orig_dd": orig_dd}

    # 2. Profit consistency (25 pts): % of simulations with positive PnL
    prob_profit = 1.0 - reshuffle.probability_of_loss()
    if prob_profit >= 0.85:
        profit_score = 25.0
    elif prob_profit >= 0.70:
        profit_score = 20.0
    elif prob_profit >= 0.50:
        profit_score = 12.0
    else:
        profit_score = 0.0
    details["profit_consistency"] = {"score": profit_score, "prob_profit": prob_profit}

    # 3. Sharpe stability (25 pts): bootstrap median Sharpe vs original
    median_sharpe = bootstrap.percentile("sharpe", 50)
    sharpe_ratio = median_sharpe / orig_sharpe if orig_sharpe != 0 else 0.0
    if sharpe_ratio >= 0.75:
        sharpe_score = 25.0
    elif sharpe_ratio >= 0.50:
        sharpe_score = 15.0
    elif sharpe_ratio >= 0.25:
        sharpe_score = 5.0
    else:
        sharpe_score = 0.0
    details["sharpe_stability"] = {"score": sharpe_score, "ratio": sharpe_ratio, "median": median_sharpe}

    # 4. Param sensitivity (25 pts): perturbation or skip median PnL
    if perturbation is not None and len(perturbation.final_equity_dist) > 0:
        median_pert_pnl = perturbation.percentile("final_equity", 50)
        pnl_ratio = median_pert_pnl / orig_eq if orig_eq != 0 else 0.0
    else:
        # Fall back to skip test
        median_skip_pnl = skip.percentile("final_equity", 50)
        pnl_ratio = median_skip_pnl / orig_eq if orig_eq != 0 else 0.0

    if pnl_ratio >= 0.80:
        param_score = 25.0
    elif pnl_ratio >= 0.50:
        param_score = 15.0
    elif pnl_ratio >= 0.25:
        param_score = 5.0
    else:
        param_score = 0.0
    details["param_sensitivity"] = {"score": param_score, "pnl_ratio": pnl_ratio}

    total = dd_score + profit_score + sharpe_score + param_score
    return total, details


def print_mc_table(name: str, result: MCResults, original_stats: dict):
    """Print percentile table matching the user's requested format."""
    s = result.summary()

    print(f"\n    [{result.simulation_type}] ({result.n_simulations} simulations)")
    print(f"    {'Metric':<20} {'5th':>10} {'25th':>10} {'50th':>10} {'75th':>10} {'95th':>10} {'Original':>10}")
    print(f"    {'-' * 80}")

    for metric, label, fmt, orig_key in [
        ("final_equity", "Final Equity ($)", "${:,.0f}", "pnl_dollars"),
        ("max_drawdown", "Max Drawdown", "{:.1f}%", "max_dd"),
        ("sharpe", "Sharpe Ratio", "{:.3f}", "sharpe"),
        ("win_rate", "Win Rate", "{:.1f}%", "win_rate"),
    ]:
        m = s.get(metric, {})
        orig_val = original_stats.get(orig_key, 0)
        vals = [m.get(f"p{p}", 0) for p in [5, 25, 50, 75, 95]]

        if metric == "final_equity":
            formatted = [f"${v + 10000:,.0f}" for v in vals]
            orig_str = f"${orig_val + 10000:,.0f}"
        elif metric == "max_drawdown":
            formatted = [f"{v:.1f}%" for v in vals]
            orig_str = f"{orig_val:.1f}%"
        elif metric == "sharpe":
            formatted = [f"{v:.3f}" for v in vals]
            orig_str = f"{orig_val:.3f}"
        else:
            formatted = [f"{v:.1f}%" for v in vals]
            orig_str = f"{orig_val:.1f}%"

        print(f"    {label:<20} {formatted[0]:>10} {formatted[1]:>10} {formatted[2]:>10} "
              f"{formatted[3]:>10} {formatted[4]:>10} {orig_str:>10}")


def main():
    quick = "--quick" in sys.argv
    n_sims = 500 if quick else 1000

    if quick:
        assets = {"BTC-USD": ASSETS["BTC-USD"]}
        strategies = ["macd_trend"]
    else:
        assets = ASSETS
        strategies = STRATEGIES

    print("=" * 80)
    print(f"  NOVAFX MONTE CARLO VALIDATION")
    print(f"  {n_sims} simulations | Skip=5% | Perturbation=±10%")
    print("=" * 80)

    mc = MonteCarloSimulator(n_simulations=n_sims, random_seed=42)
    all_scores = []

    for symbol, (ticker, ac, sl_m, tp_m) in assets.items():
        print(f"\n{'═' * 80}")
        print(f"  {symbol}")
        print(f"{'═' * 80}")

        try:
            df = fetch_ohlcv(ticker, "1h", "730d")
        except Exception:
            try:
                df = fetch_ohlcv(ticker, "1h", "90d")
            except Exception as e:
                print(f"  [SKIP] {e}")
                continue

        if df is None or len(df) < 300:
            print(f"  [SKIP] only {len(df) if df is not None else 0} bars")
            continue
        print(f"  {len(df)} bars loaded")

        for strat in strategies:
            params = DEFAULT_PARAMS.get(strat, {})
            if not params:
                continue

            print(f"\n  Strategy: {strat}")
            print(f"  Params: {params}")

            # Generate base trade results
            sigs = _run_parameterized(df, strat, params)
            trades = _simulate(df, sigs, sl_m, tp_m)
            pnls = np.array([t.pnl_pct for t in trades])

            if len(pnls) < 10:
                print(f"    [SKIP] only {len(pnls)} trades (need 10+)")
                continue

            # Original stats
            orig_eq, orig_dd, orig_sharpe, orig_wr = _compute_equity_stats(pnls)
            orig_stats = {
                "pnl_dollars": orig_eq,
                "max_dd": orig_dd,
                "sharpe": orig_sharpe,
                "win_rate": orig_wr,
            }
            print(f"    Original: {len(pnls)} trades | PnL=${orig_eq + 10000:,.0f} | "
                  f"DD={orig_dd:.1f}% | Sharpe={orig_sharpe:.3f} | WR={orig_wr:.1f}%")

            # Run MC simulations
            r_reshuffle = mc.reshuffle_trades(pnls)
            r_skip = mc.skip_trades(pnls, skip_pct=0.05)
            r_bootstrap = mc.bootstrap_confidence(pnls)

            # Parameter perturbation
            strat_fn = _strategy_fn_for_perturbation(strat)
            r_perturb = mc.parameter_perturbation(strat_fn, df, params, 0.10, sl_m, tp_m)

            # Print tables
            print_mc_table(symbol, r_reshuffle, orig_stats)
            print_mc_table(symbol, r_skip, orig_stats)
            print_mc_table(symbol, r_bootstrap, orig_stats)
            print_mc_table(symbol, r_perturb, orig_stats)

            # Robustness score
            score, details = compute_robustness_score(pnls, r_reshuffle, r_skip, r_bootstrap, r_perturb)
            tag = "ROBUST" if score >= 60 else "MARGINAL" if score >= 40 else "WEAK"

            print(f"\n    ROBUSTNESS SCORE: {score:.0f}/100 [{tag}]")
            for k, v in details.items():
                print(f"      {k}: {v['score']:.0f}/25 pts")

            # DD warning
            p95_dd = r_reshuffle.percentile("max_drawdown", 5)
            if abs(p95_dd) > abs(orig_dd) * 2:
                print(f"    ⚠ WARNING: 95th pct DD ({p95_dd:.1f}%) > 2x backtest DD ({orig_dd:.1f}%)")

            # P(loss)
            p_loss = r_reshuffle.probability_of_loss()
            print(f"    P(loss): {p_loss:.1%}")

            all_scores.append({
                "symbol": symbol,
                "strategy": strat,
                "n_trades": len(pnls),
                "orig_pnl": orig_eq,
                "orig_sharpe": orig_sharpe,
                "score": score,
                "tag": tag,
                "p_loss": p_loss,
                "p95_dd": p95_dd,
            })

            time.sleep(0.1)

    # Summary
    if all_scores:
        print(f"\n{'=' * 80}")
        print(f"  MONTE CARLO SUMMARY")
        print(f"{'=' * 80}")
        print(f"  {'Symbol':<10} {'Strategy':<22} {'#Tr':>4} {'Orig$':>8} {'Score':>6} {'Tag':>8} {'P(loss)':>8} {'95%DD':>7}")
        print(f"  {'-' * 78}")
        for r in sorted(all_scores, key=lambda x: x["score"], reverse=True):
            print(f"  {r['symbol']:<10} {r['strategy']:<22} {r['n_trades']:>4} "
                  f"${r['orig_pnl'] + 10000:>7,.0f} {r['score']:>5.0f} {r['tag']:>8} "
                  f"{r['p_loss']:>7.1%} {r['p95_dd']:>6.1f}%")

        robust = sum(1 for r in all_scores if r["score"] >= 60)
        total = len(all_scores)
        print(f"\n  Production-ready (score >= 60): {robust}/{total}")

    print()


if __name__ == "__main__":
    main()

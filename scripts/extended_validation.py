#!/usr/bin/env python3
"""Extended validation: 12-18 month backtests + Monte Carlo across all combos.

Usage:
    python scripts/extended_validation.py
    python scripts/extended_validation.py --range 730d --sims 500
"""
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_harness import fetch_ohlcv
from src.strategies import STRATEGY_REGISTRY, get_strategy
from src.vectorbt_adapters import VBTStrategyAdapter
from src.vectorbt_adapters.data_utils import prepare_vbt_data
from src.validation.monte_carlo import MonteCarloSimulator, _compute_equity_stats
from config.strategy_blacklist import is_blacklisted

ASSETS = {
    "BTC-USD": ("BTC-USD", "crypto"),
    "ETH-USD": ("ETH-USD", "crypto"),
    "SOL-USD": ("SOL-USD", "crypto"),
    "XRP-USD": ("XRP-USD", "crypto"),
    "EURUSD": ("EURUSD=X", "forex"),
    "GBPUSD": ("GBPUSD=X", "forex"),
    "USDJPY": ("JPY=X", "forex"),
    "AAPL": ("AAPL", "stocks"),
    "SPY": ("SPY", "stocks"),
    "QQQ": ("QQQ", "stocks"),
    "XAUUSD": ("GC=F", "commodities"),
    "XAGUSD": ("SI=F", "commodities"),
}

STRATEGIES = list(STRATEGY_REGISTRY.keys())


def run_combo(strategy_name: str, symbol: str, df: pd.DataFrame,
              mc: MonteCarloSimulator, n_sims: int) -> dict[str, Any] | None:
    """Run backtest + MC for one strategy-asset combo."""
    try:
        adapter = VBTStrategyAdapter(strategy_name, df, symbol)
        result = adapter.run_backtest()
    except Exception as e:
        return {"error": str(e)}

    n = result.num_trades
    if n < 3:
        return {"n_trades": n, "skipped": "too_few_trades"}

    entry = {
        "strategy": strategy_name,
        "symbol": symbol,
        "n_trades": n,
        "total_return": round(result.total_return, 6),
        "sharpe": round(result.sharpe_ratio, 4),
        "sortino": round(result.sortino_ratio, 4),
        "max_drawdown": round(result.max_drawdown, 6),
        "win_rate": round(result.win_rate, 4),
        "profit_factor": round(result.profit_factor, 4),
        "statistically_significant": n >= 50,
    }

    # Monte Carlo (only if enough trades for meaningful simulation)
    if n >= 20:
        # Extract trade PnLs from portfolio
        trades = result.portfolio.trades.records_readable
        if len(trades) > 0 and "Return" in trades.columns:
            pnls = trades["Return"].values * 100
        else:
            pnls = np.array([result.total_return / max(n, 1) * 100] * n)

        mc_reshuffle = mc.reshuffle_trades(pnls)
        mc_skip = mc.skip_trades(pnls, skip_pct=0.05)

        summary = mc_reshuffle.summary()
        entry["mc_ran"] = True
        entry["mc_p_loss"] = round(mc_reshuffle.probability_of_loss(), 4)
        entry["mc_sharpe_p5"] = round(summary["sharpe"]["p5"], 4)
        entry["mc_sharpe_p50"] = round(summary["sharpe"]["p50"], 4)
        entry["mc_sharpe_p95"] = round(summary["sharpe"]["p95"], 4)
        entry["mc_return_p5"] = round(summary["final_equity"]["p5"], 2)
        entry["mc_return_p50"] = round(summary["final_equity"]["p50"], 2)
        entry["mc_return_p95"] = round(summary["final_equity"]["p95"], 2)
        entry["mc_dd_p5"] = round(summary["max_drawdown"]["p5"], 4)

        # Robustness score (simplified: p_loss < 30% and sharpe_p5 > 0)
        robust_score = 0
        if mc_reshuffle.probability_of_loss() < 0.3:
            robust_score += 35
        if summary["sharpe"]["p5"] > -1.0:
            robust_score += 35
        skip_summary = mc_skip.summary()
        if skip_summary["final_equity"]["p50"] > 0:
            robust_score += 30
        entry["mc_robust_score"] = robust_score
        entry["robust"] = robust_score >= 70
    else:
        entry["mc_ran"] = False
        entry["robust"] = False

    return entry


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--range", default="730d")
    parser.add_argument("--sims", type=int, default=500)
    parser.add_argument("--output", default="results/extended_validation.json")
    args = parser.parse_args()

    print("=" * 70)
    print("  NOVAFX EXTENDED VALIDATION")
    print(f"  Range: {args.range} | MC sims: {args.sims}")
    print("=" * 70)

    mc = MonteCarloSimulator(n_simulations=args.sims, random_seed=42)
    all_results: list[dict] = []
    skipped = 0
    errors = 0

    for symbol, (ticker, ac) in ASSETS.items():
        print(f"\n{'═' * 60}")
        print(f"  {symbol} ({ac})")

        try:
            df = fetch_ohlcv(ticker, "1h", args.range)
        except Exception:
            try:
                df = fetch_ohlcv(ticker, "1h", "90d")
            except Exception as e:
                print(f"  [SKIP] fetch failed: {e}")
                continue

        if df is None or len(df) < 300:
            print(f"  [SKIP] {len(df) if df is not None else 0} bars")
            continue

        df = df.set_index("timestamp")
        df = prepare_vbt_data(df, symbol)
        print(f"  {len(df)} bars")

        for strat in STRATEGIES:
            if is_blacklisted(strat, symbol):
                print(f"    {strat}: BLACKLISTED — skip")
                skipped += 1
                continue

            print(f"    {strat}", end=" ", flush=True)
            entry = run_combo(strat, symbol, df, mc, args.sims)

            if entry is None:
                errors += 1
                print("[ERROR]")
                continue

            if "error" in entry:
                errors += 1
                print(f"[ERROR: {entry['error'][:40]}]")
                continue

            if "skipped" in entry:
                skipped += 1
                print(f"[{entry['n_trades']} trades — skip]")
                continue

            tag = ""
            if entry.get("robust"):
                tag = "ROBUST"
            elif entry.get("statistically_significant"):
                tag = "SIG"
            print(f"-> {entry['n_trades']}tr Sh={entry['sharpe']:.2f} "
                  f"Ret={entry['total_return']:.2%} {tag}")
            all_results.append(entry)

        time.sleep(0.3)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "meta": {
                "range": args.range,
                "mc_sims": args.sims,
                "n_assets": len(ASSETS),
                "n_strategies": len(STRATEGIES),
                "n_results": len(all_results),
                "n_skipped": skipped,
                "n_errors": errors,
            },
            "results": all_results,
        }, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  DONE: {len(all_results)} combos validated, {skipped} skipped, {errors} errors")
    print(f"  Saved to {output_path}")

    sig = sum(1 for r in all_results if r.get("statistically_significant"))
    rob = sum(1 for r in all_results if r.get("robust"))
    print(f"  Statistically significant (N>=50): {sig}")
    print(f"  Robust (MC>=70): {rob}")


if __name__ == "__main__":
    main()

"""
Run Walk-Forward Optimization across strategies and assets.

Usage:
    python scripts/run_walk_forward.py                  # All strategies, crypto only
    python scripts/run_walk_forward.py --strategy ema_cross --all-assets
    python scripts/run_walk_forward.py --quick           # One strategy, one asset
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest_harness import fetch_ohlcv
from src.optimization.walk_forward import (
    WalkForwardOptimizer, WalkForwardResults, PARAM_GRIDS,
)

# Assets for WFO — use longest-history assets
ASSETS = {
    "crypto": {
        "BTC-USD": "BTC-USD",
        "ETH-USD": "ETH-USD",
        "SOL-USD": "SOL-USD",
        "XRP-USD": "XRP-USD",
    },
    "forex": {
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X",
    },
    "stocks": {
        "SPY": "SPY",
        "QQQ": "QQQ",
        "AAPL": "AAPL",
    },
}

# ATR multipliers per asset class
SL_TP = {
    "crypto": (1.5, 3.5),
    "forex": (1.5, 2.0),
    "stocks": (2.0, 3.0),
}

STRATEGIES_TO_RUN = [
    "ema_cross", "rsi_adaptive", "macd_zero", "bb_reversion",
    "momentum_breakout", "donchian_breakout", "macd_trend",
]


def print_wfo_result(symbol: str, strategy: str, r: WalkForwardResults):
    """Print one WFO result block."""
    eff_tag = "OK" if r.efficiency_ratio > 0.5 else "OVERFIT" if r.efficiency_ratio > 0 else "NEG"

    print(f"\n    {strategy}")
    print(f"    {'─' * 50}")
    print(f"    Windows: {len(r.windows)}  |  OOS trades: {r.total_oos_trades}")
    print(f"    OOS Sharpe: {r.oos_sharpe:.3f}  |  OOS PnL: {r.oos_pnl:+.2f}%  |  OOS WR: {r.oos_win_rate:.1f}%")
    print(f"    Efficiency ratio: {r.efficiency_ratio:.3f}  [{eff_tag}]")

    # Per-window detail
    if r.windows:
        print(f"    {'Win':>4} {'IS_Sh':>7} {'OOS_Sh':>7} {'IS_PnL':>8} {'OOS_PnL':>8} {'OOS_WR':>7} {'#Tr':>4}  Params")
        for w in r.windows:
            params_str = ", ".join(f"{k}={v}" for k, v in w.best_params.items())
            print(f"    {w.window_id:>4} {w.is_sharpe:>7.3f} {w.oos_sharpe:>7.3f} "
                  f"{w.is_pnl:>+7.2f}% {w.oos_pnl:>+7.2f}% {w.oos_win_rate:>6.1f}% {w.n_oos_trades:>4}  {params_str}")

    # Param stability
    if r.param_stability:
        print(f"    Param stability:")
        for k, v in r.param_stability.items():
            vals = v["values"]
            print(f"      {k}: unique={v['n_unique']}/{len(vals)}  "
                  f"cv={v['cv']:.3f}  values={vals}")


def main():
    quick = "--quick" in sys.argv
    all_assets = "--all-assets" in sys.argv
    strat_filter = None
    for i, a in enumerate(sys.argv):
        if a == "--strategy" and i + 1 < len(sys.argv):
            strat_filter = sys.argv[i + 1]

    strategies = [strat_filter] if strat_filter else STRATEGIES_TO_RUN
    asset_classes = ASSETS if all_assets else {"crypto": ASSETS["crypto"]}

    if quick:
        asset_classes = {"crypto": {"BTC-USD": "BTC-USD"}}
        strategies = ["ema_cross"]

    print("=" * 70)
    print("  NOVAFX WALK-FORWARD OPTIMIZATION")
    print(f"  Train=2000 bars | Test=500 bars | Step=500 | Embargo=50")
    print(f"  Strategies: {', '.join(strategies)}")
    print("=" * 70)

    wfo = WalkForwardOptimizer(train_bars=2000, test_bars=500, step_bars=500, embargo_bars=50)

    summary_rows = []

    for ac, syms in asset_classes.items():
        sl_m, tp_m = SL_TP.get(ac, (1.5, 3.0))

        for symbol, ticker in syms.items():
            print(f"\n{'═' * 70}")
            print(f"  {symbol} ({ac})")
            print(f"{'═' * 70}")

            try:
                # Use max range for hourly data (~2 years)
                df = fetch_ohlcv(ticker, "1h", "730d")
            except Exception:
                try:
                    df = fetch_ohlcv(ticker, "1h", "90d")
                except Exception as e:
                    print(f"  [SKIP] fetch failed: {e}")
                    continue

            if df is None or len(df) < 2600:
                bars = len(df) if df is not None else 0
                print(f"  [SKIP] only {bars} bars (need 2600+)")
                continue
            print(f"  {len(df)} bars loaded")

            for strat in strategies:
                if strat not in PARAM_GRIDS:
                    print(f"\n    {strat}: no param grid, skipping")
                    continue

                try:
                    result = wfo.run(df, strat, metric="sharpe", sl_mult=sl_m, tp_mult=tp_m)
                    print_wfo_result(symbol, strat, result)
                    summary_rows.append({
                        "symbol": symbol,
                        "strategy": strat,
                        "oos_sharpe": result.oos_sharpe,
                        "oos_pnl": result.oos_pnl,
                        "oos_wr": result.oos_win_rate,
                        "efficiency": result.efficiency_ratio,
                        "n_trades": result.total_oos_trades,
                        "n_windows": len(result.windows),
                    })
                except Exception as e:
                    print(f"\n    {strat}: [ERROR] {e}")

            time.sleep(0.3)

    # Summary table
    if summary_rows:
        print(f"\n{'=' * 70}")
        print(f"  WALK-FORWARD SUMMARY")
        print(f"{'=' * 70}")
        print(f"  {'Symbol':<10} {'Strategy':<22} {'OOS_Sh':>7} {'OOS_PnL':>8} {'WR%':>6} {'Eff':>6} {'#Tr':>4} {'Flag':>8}")
        print(f"  {'-' * 72}")
        for r in sorted(summary_rows, key=lambda x: x["oos_pnl"], reverse=True):
            flag = "OK" if r["efficiency"] > 0.5 else "OVERFIT" if r["efficiency"] > 0 else "NEG"
            print(f"  {r['symbol']:<10} {r['strategy']:<22} {r['oos_sharpe']:>7.3f} "
                  f"{r['oos_pnl']:>+7.2f}% {r['oos_wr']:>5.1f}% {r['efficiency']:>5.2f} "
                  f"{r['n_trades']:>4} {flag:>8}")

        # Flag alerts
        overfit = [r for r in summary_rows if 0 < r["efficiency"] < 0.5]
        if overfit:
            print(f"\n  ⚠ OVERFITTING WARNING: {len(overfit)} strategy-asset combos have efficiency < 0.5")
            for r in overfit:
                print(f"    {r['symbol']}/{r['strategy']}: eff={r['efficiency']:.3f}")

    print()


if __name__ == "__main__":
    main()

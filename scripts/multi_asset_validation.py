#!/usr/bin/env python3
"""Multi-asset validation with walk-forward optimization.

Generates asset-specific parameter configs in config/asset_params.json.

Usage:
    python scripts/multi_asset_validation.py --data-dir data/extended
"""
import json
import sys
import argparse
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.strategies import STRATEGY_REGISTRY
from src.vectorbt_adapters import VBTStrategyAdapter, VBTParamOptimizer, VBTWalkForward
from src.vectorbt_adapters.data_utils import prepare_vbt_data
from src.execution.volume_handler import classify_asset
from config.strategy_blacklist import is_blacklisted


@dataclass
class AssetConfig:
    asset: str
    asset_class: str
    strategies: dict[str, dict[str, Any]]
    best_strategy: str
    best_sharpe: float
    wf_efficiency: float
    mc_score: float
    n_bars: int
    date_range: str


def load_csv(filepath: Path) -> pd.DataFrame:
    """Load CSV with timestamp index."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower()
    for col in ["timestamp", "datetime", "date"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break
    return df.sort_index()


def run_asset(asset: str, data: pd.DataFrame, strategies: list[str],
              n_wf: int, verbose: bool) -> AssetConfig | None:
    """Full optimization pipeline for one asset."""
    ac = classify_asset(asset)
    if verbose:
        print(f"\n{'=' * 55}")
        print(f"  {asset} ({ac}) — {len(data)} bars")

    results: dict[str, dict] = {}

    for strat in strategies:
        if is_blacklisted(strat, asset):
            if verbose:
                print(f"    {strat}: BLACKLISTED")
            continue

        if verbose:
            print(f"    {strat}", end=" ", flush=True)

        try:
            # 1. Param optimization
            opt = VBTParamOptimizer(strat, data, asset)
            opt_r = opt.optimize(metric="sharpe_ratio", min_trades=5, verbose=False)
            best_params = opt_r.best_params

            # 2. Walk-forward
            wf = VBTWalkForward(strat, data, asset)
            wf_r = wf.run(n_windows=n_wf, verbose=False)

            # 3. Backtest with best params
            adapter = VBTStrategyAdapter(strat, data, asset)
            bt = adapter.run_backtest(params=best_params)

            results[strat] = {
                "params": {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                           for k, v in best_params.items()},
                "sharpe": round(float(opt_r.best_metric_value), 4),
                "wf_efficiency": round(float(wf_r.efficiency_ratio), 4),
                "oos_sharpe": round(float(wf_r.oos_sharpe), 4),
                "n_trades": int(bt.num_trades),
                "total_return": round(float(bt.total_return), 6),
                "max_drawdown": round(float(bt.max_drawdown), 6),
                "win_rate": round(float(bt.win_rate), 4),
                "param_stability": {k: round(float(v), 4) for k, v in wf_r.param_stability.items()},
            }

            if verbose:
                print(f"Sh={results[strat]['sharpe']:.2f} WF={results[strat]['wf_efficiency']:.2f} "
                      f"n={bt.num_trades}")
        except Exception as e:
            if verbose:
                print(f"[{str(e)[:40]}]")

    if not results:
        return None

    def score(s):
        r = results[s]
        return 0.5 * min(r["sharpe"], 3) / 3 + 0.3 * min(r["wf_efficiency"], 1) + 0.2 * r["win_rate"]

    best = max(results, key=score)
    b = results[best]

    if verbose:
        print(f"  -> Best: {best} (Sh={b['sharpe']:.2f})")

    return AssetConfig(
        asset=asset, asset_class=ac, strategies=results,
        best_strategy=best, best_sharpe=b["sharpe"],
        wf_efficiency=b["wf_efficiency"], mc_score=0,
        n_bars=len(data),
        date_range=f"{data.index[0]} to {data.index[-1]}",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/extended")
    parser.add_argument("--output", default="config/asset_params.json")
    parser.add_argument("--wf-windows", type=int, default=6)
    parser.add_argument("--strategies", nargs="+")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    strategies = args.strategies or list(STRATEGY_REGISTRY.keys())

    # Discover CSV files
    csvs = sorted(data_dir.glob("*_1h.csv"))
    if not csvs:
        print(f"No CSV files in {data_dir}. Run fetch_historical_data.py first.")
        return

    print(f"Multi-Asset Validation: {len(csvs)} files, {len(strategies)} strategies")

    configs: list[AssetConfig] = []
    for csv_path in csvs:
        asset = csv_path.stem.replace("_1h", "").upper().replace("_", "-")
        try:
            df = load_csv(csv_path)
            df = prepare_vbt_data(df, asset)
            if len(df) < 500:
                print(f"\n  {asset}: only {len(df)} bars, skip")
                continue
            cfg = run_asset(asset, df, strategies, args.wf_windows, verbose=True)
            if cfg:
                configs.append(cfg)
        except Exception as e:
            print(f"\n  {asset}: ERROR {e}")

    # Save
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated": datetime.now().isoformat(),
        "n_assets": len(configs),
        "assets": {c.asset: asdict(c) for c in configs},
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"\n{'=' * 55}")
    print(f"  {len(configs)} assets validated -> {out}")
    for c in sorted(configs, key=lambda x: x.best_sharpe, reverse=True):
        print(f"    {c.asset:12} {c.best_strategy:20} Sh={c.best_sharpe:.2f} WF={c.wf_efficiency:.2f}")


if __name__ == "__main__":
    main()

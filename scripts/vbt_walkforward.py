#!/usr/bin/env python3
"""VectorBT walk-forward optimization CLI.

Usage:
    python scripts/vbt_walkforward.py --strategy macd_trend --symbol BTC-USD
    python scripts/vbt_walkforward.py --strategy donchian_breakout --symbol ETH-USD --windows 8
"""
import sys
import argparse
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_harness import fetch_ohlcv
from src.vectorbt_adapters import VBTWalkForward
from src.vectorbt_adapters.data_utils import prepare_vbt_data


def main():
    parser = argparse.ArgumentParser(description="VBT Walk-Forward Optimization")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--train-pct", type=float, default=0.7)
    parser.add_argument("--metric", default="sharpe_ratio")
    parser.add_argument("--range", default="730d")
    parser.add_argument("--output", help="Output CSV path")
    args = parser.parse_args()

    yf_ticker = args.symbol
    print(f"Fetching {yf_ticker} ({args.range})...")
    df = fetch_ohlcv(yf_ticker, "1h", args.range)
    df = df.set_index("timestamp")
    df = prepare_vbt_data(df, args.symbol)
    print(f"Loaded {len(df)} bars")

    wf = VBTWalkForward(args.strategy, df, args.symbol)
    results = wf.run(
        n_windows=args.windows,
        train_pct=args.train_pct,
        metric=args.metric,
    )

    if args.output:
        rows = [{
            "window": w.window_idx,
            "is_sharpe": w.is_sharpe,
            "oos_sharpe": w.oos_sharpe,
            "oos_return": w.oos_return,
            **w.best_params,
        } for w in results.windows]
        pd.DataFrame(rows).to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

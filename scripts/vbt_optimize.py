#!/usr/bin/env python3
"""VectorBT parameter optimization CLI.

Usage:
    python scripts/vbt_optimize.py --strategy macd_trend --symbol BTC-USD
    python scripts/vbt_optimize.py --strategy momentum_breakout --symbol ETH-USD --metric sortino_ratio
"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_harness import fetch_ohlcv
from src.vectorbt_adapters import VBTParamOptimizer
from src.vectorbt_adapters.data_utils import prepare_vbt_data
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="VBT Parameter Optimization")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--metric", default="sharpe_ratio")
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--range", default="730d", help="Yahoo Finance range")
    parser.add_argument("--output", help="Output CSV path")
    args = parser.parse_args()

    # Fetch data
    yf_ticker = args.symbol
    print(f"Fetching {yf_ticker} ({args.range})...")
    df = fetch_ohlcv(yf_ticker, "1h", args.range)
    # Convert to DatetimeIndex for VBT
    df = df.set_index("timestamp")
    df = prepare_vbt_data(df, args.symbol)
    print(f"Loaded {len(df)} bars: {df.index[0]} to {df.index[-1]}")

    # Optimize
    optimizer = VBTParamOptimizer(args.strategy, df, args.symbol)
    result = optimizer.optimize(metric=args.metric, min_trades=args.min_trades)

    print(f"\nTop 10 Combinations:")
    print(result.all_results.head(10).to_string())

    if args.output:
        result.all_results.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()

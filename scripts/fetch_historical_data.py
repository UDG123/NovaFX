#!/usr/bin/env python3
"""Fetch historical OHLCV data for multiple assets.

Usage:
    python scripts/fetch_historical_data.py --preset all --period 18mo
    python scripts/fetch_historical_data.py --assets BTC-USD ETH-USD --period 2y
"""
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_harness import fetch_ohlcv

PRESETS = {
    "crypto": [
        ("BTC-USD", "BTC-USD"), ("ETH-USD", "ETH-USD"), ("SOL-USD", "SOL-USD"),
        ("XRP-USD", "XRP-USD"), ("BNB-USD", "BNB-USD"), ("AVAX-USD", "AVAX-USD"),
        ("LINK-USD", "LINK-USD"), ("DOT-USD", "DOT-USD"), ("ATOM-USD", "ATOM-USD"),
    ],
    "forex_major": [
        ("EURUSD", "EURUSD=X"), ("GBPUSD", "GBPUSD=X"), ("USDJPY", "JPY=X"),
        ("AUDUSD", "AUDUSD=X"), ("USDCAD", "CAD=X"), ("USDCHF", "CHF=X"),
    ],
    "forex_minor": [
        ("EURGBP", "EURGBP=X"), ("EURJPY", "EURJPY=X"), ("GBPJPY", "GBPJPY=X"),
    ],
    "indices": [
        ("SPY", "SPY"), ("QQQ", "QQQ"), ("DIA", "DIA"),
    ],
}
PRESETS["all"] = PRESETS["crypto"] + PRESETS["forex_major"] + PRESETS["forex_minor"] + PRESETS["indices"]


def main():
    parser = argparse.ArgumentParser(description="Fetch Historical Data")
    parser.add_argument("--assets", nargs="+", help="Specific symbols")
    parser.add_argument("--preset", choices=list(PRESETS.keys()))
    parser.add_argument("--period", default="730d", help="Yahoo range (e.g. 730d, 90d)")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--output-dir", default="data/extended")
    args = parser.parse_args()

    if args.preset:
        assets = PRESETS[args.preset]
    elif args.assets:
        assets = [(a, a) for a in args.assets]
    else:
        assets = PRESETS["all"]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(assets)} assets ({args.period}, {args.interval})")
    success = 0

    for name, ticker in assets:
        print(f"  {name} ({ticker})", end=" ", flush=True)
        try:
            df = fetch_ohlcv(ticker, args.interval, args.period)
            if df is None or len(df) < 100:
                print(f"[{len(df) if df is not None else 0} bars — skip]")
                continue
            clean = name.replace("-", "_").replace("/", "_").lower()
            filepath = output_dir / f"{clean}_{args.interval}.csv"
            df.to_csv(filepath, index=False)
            print(f"-> {len(df)} bars -> {filepath.name}")
            success += 1
        except Exception as e:
            print(f"[ERROR: {e}]")
        time.sleep(0.5)

    print(f"\nFetched {success}/{len(assets)} assets to {output_dir}")


if __name__ == "__main__":
    main()

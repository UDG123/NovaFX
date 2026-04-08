#!/usr/bin/env python3
"""Fetch historical OHLCV data from Alpaca Markets.

Usage:
    python scripts/fetch_data_alpaca.py --crypto BTC/USD ETH/USD --days 365
    python scripts/fetch_data_alpaca.py --stocks SPY QQQ AAPL --days 180
    python scripts/fetch_data_alpaca.py --preset crypto --days 365
"""
import os
import sys
import argparse
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

PRESETS = {
    "crypto": ["BTC/USD", "ETH/USD", "SOL/USD", "AVAX/USD", "LINK/USD", "DOT/USD"],
    "crypto_small": ["BTC/USD", "ETH/USD"],
    "stocks_index": ["SPY", "QQQ", "DIA", "IWM"],
    "stocks_tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META"],
    "stocks_mixed": ["SPY", "QQQ", "AAPL", "MSFT", "TSLA", "JPM"],
}


def fetch_crypto(symbols: list[str], days: int, output_dir: Path) -> dict:
    """Fetch crypto OHLCV from Alpaca (no API keys needed for crypto)."""
    from alpaca.data.historical import CryptoHistoricalDataClient
    from alpaca.data.requests import CryptoBarsRequest
    from alpaca.data.timeframe import TimeFrame

    client = CryptoHistoricalDataClient()
    end = datetime.now()
    start = end - timedelta(days=days)
    results = {}

    for symbol in symbols:
        print(f"  {symbol}", end=" ", flush=True)
        try:
            request = CryptoBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Hour,
                start=start, end=end,
            )
            bars = client.get_crypto_bars(request)
            df = bars.df
            if df.empty:
                print("NO DATA")
                continue
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level="symbol", drop=True)
            df = df[["open", "high", "low", "close", "volume"]]
            clean = symbol.replace("/", "_").lower()
            filepath = output_dir / f"{clean}_1h.csv"
            df.to_csv(filepath)
            results[symbol] = df
            print(f"-> {len(df)} bars")
        except Exception as e:
            print(f"ERROR: {e}")

    return results


def fetch_stocks(symbols: list[str], days: int, output_dir: Path) -> dict:
    """Fetch stock OHLCV from Alpaca (API keys required)."""
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key = os.environ.get("APCA_API_KEY_ID")
    api_secret = os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not api_secret:
        raise ValueError(
            "Set APCA_API_KEY_ID and APCA_API_SECRET_KEY.\n"
            "Get free keys at: https://app.alpaca.markets/signup"
        )

    client = StockHistoricalDataClient(api_key, api_secret)
    end = datetime.now()
    start = end - timedelta(days=days)
    results = {}

    for symbol in symbols:
        print(f"  {symbol}", end=" ", flush=True)
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Hour,
                start=start, end=end,
            )
            bars = client.get_stock_bars(request)
            df = bars.df
            if df.empty:
                print("NO DATA")
                continue
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level="symbol", drop=True)
            df = df[["open", "high", "low", "close", "volume"]]
            filepath = output_dir / f"{symbol.lower()}_1h.csv"
            df.to_csv(filepath)
            results[symbol] = df
            print(f"-> {len(df)} bars")
        except Exception as e:
            print(f"ERROR: {e}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Fetch Data from Alpaca")
    parser.add_argument("--crypto", nargs="+")
    parser.add_argument("--stocks", nargs="+")
    parser.add_argument("--preset", choices=list(PRESETS.keys()))
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--output-dir", default="data/extended")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    crypto_syms: list[str] = []
    stock_syms: list[str] = []

    if args.preset:
        for s in PRESETS[args.preset]:
            (crypto_syms if "/" in s else stock_syms).append(s)
    if args.crypto:
        crypto_syms.extend(args.crypto)
    if args.stocks:
        stock_syms.extend(args.stocks)

    print(f"Alpaca Data Fetcher: {args.days} days -> {output_dir}")

    if crypto_syms:
        print(f"\nCRYPTO ({len(crypto_syms)}):")
        fetch_crypto(crypto_syms, args.days, output_dir)

    if stock_syms:
        print(f"\nSTOCKS ({len(stock_syms)}):")
        try:
            fetch_stocks(stock_syms, args.days, output_dir)
        except ValueError as e:
            print(f"  {e}")

    print(f"\nDone. Files in {output_dir}/")


if __name__ == "__main__":
    main()

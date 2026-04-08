"""
Train HMM regime models for each asset in the NovaFX universe.

Uses 2 years of hourly data from Yahoo Finance Chart API.
Saves serialized models to models/regime/.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest_harness import fetch_ohlcv
from src.regime.hmm_regime import HMMRegimeDetector

# Full symbol universe — maps NovaFX name to Yahoo Finance ticker
UNIVERSE = {
    # Forex
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "CAD=X",
    "USDCHF": "CHF=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    # Crypto
    "BTC-USD": "BTC-USD",
    "ETH-USD": "ETH-USD",
    "SOL-USD": "SOL-USD",
    "BNB-USD": "BNB-USD",
    "XRP-USD": "XRP-USD",
    # Stocks
    "AAPL": "AAPL",
    "MSFT": "MSFT",
    "NVDA": "NVDA",
    "TSLA": "TSLA",
    "SPY": "SPY",
    "QQQ": "QQQ",
    # Commodities
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
}


def train_single(symbol: str, ticker: str) -> dict | None:
    """Fetch data and train one HMM model."""
    print(f"  {symbol} ({ticker})", end=" ", flush=True)

    try:
        # Yahoo hourly data max range is ~730d (2 years)
        df = fetch_ohlcv(ticker, interval="1h", range_="730d")
    except Exception as e:
        # Fallback to shorter range for assets with limited hourly history
        try:
            df = fetch_ohlcv(ticker, interval="1h", range_="90d")
        except Exception:
            print(f"[SKIP] fetch failed: {e}")
            return None

    if df is None or len(df) < 200:
        print(f"[SKIP] only {len(df) if df is not None else 0} bars")
        return None

    print(f"({len(df)} bars)", end=" ", flush=True)

    detector = HMMRegimeDetector(n_states=3, n_iter=200, vol_window=20, use_adx=True)
    try:
        stats = detector.train(df, symbol=symbol)
    except Exception as e:
        print(f"[FAIL] {e}")
        return None

    path = detector.save_model()
    print(f"-> saved to {path.name}")

    return stats


def main():
    print("=" * 70)
    print("  NOVAFX HMM REGIME MODEL TRAINING")
    print("  Training 3-state Gaussian HMM per asset")
    print("=" * 70)

    results = {}
    failed = []

    for symbol, ticker in UNIVERSE.items():
        stats = train_single(symbol, ticker)
        if stats:
            results[symbol] = stats
        else:
            failed.append(symbol)
        time.sleep(0.5)  # Rate limit

    # Report
    print(f"\n{'=' * 70}")
    print(f"  TRAINING RESULTS — {len(results)}/{len(UNIVERSE)} models trained")
    print(f"{'=' * 70}")

    print(f"\n  {'Symbol':<12} {'Bull%':>6} {'Bear%':>6} {'Range%':>7} "
          f"{'Bull Ret':>9} {'Bear Ret':>9} {'Bull Vol':>9} {'Bear Vol':>9}")
    print(f"  {'-' * 75}")

    for symbol, stats in results.items():
        bull = stats.get("bull", {})
        bear = stats.get("bear", {})
        rang = stats.get("ranging", {})

        print(f"  {symbol:<12} "
              f"{bull.get('time_pct', 0):>5.1f}% "
              f"{bear.get('time_pct', 0):>5.1f}% "
              f"{rang.get('time_pct', 0):>6.1f}% "
              f"{bull.get('mean_return', 0):>+8.5f} "
              f"{bear.get('mean_return', 0):>+8.5f} "
              f"{bull.get('volatility', 0):>8.5f} "
              f"{bear.get('volatility', 0):>8.5f}")

    if failed:
        print(f"\n  Failed: {', '.join(failed)}")

    # Verify all models load correctly
    print(f"\n  Verification: loading all saved models...")
    model_dir = Path(__file__).resolve().parent.parent / "models" / "regime"
    for pkl in sorted(model_dir.glob("*.pkl")):
        det = HMMRegimeDetector()
        det.load_model(pkl)
        print(f"    {pkl.name} -> {det.symbol} ({det.n_states} states)")

    print(f"\n  Done. Models saved to models/regime/")


if __name__ == "__main__":
    main()

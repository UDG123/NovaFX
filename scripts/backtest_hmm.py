"""
NovaFX Backtest with HMM Regime Filter — A/B comparison.

Runs the same signal pipeline twice:
  A) Current simple regime filter (ADX + Hurst)
  B) HMM regime filter (trained Gaussian HMM)

Reports side-by-side: signal count, win rate, PnL per regime.
"""
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest_harness import (
    fetch_ohlcv, calc_ema, calc_atr, calc_adx, calc_hurst,
    ALL_STRATEGIES, REGIME_ALLOWED, ATR_MULT, PCT_FALLBACK,
    detect_regime, is_choppy, check_volume, simulate_trade,
)
from src.regime.hmm_regime import HMMRegimeDetector, HMM_REGIME_ALLOWED

# Yahoo tickers per asset class
SYMBOLS = {
    "forex": {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
        "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X", "USDCHF": "CHF=X",
        "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X",
    },
    "stocks": {
        "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA",
        "TSLA": "TSLA", "SPY": "SPY", "QQQ": "QQQ",
    },
    "crypto": {
        "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD",
        "SOL-USD": "SOL-USD", "XRP-USD": "XRP-USD",
    },
    "commodities": {
        "XAUUSD": "GC=F", "XAGUSD": "SI=F",
    },
}

# Map backtest symbol names to model file names
SYMBOL_TO_MODEL = {
    "EURUSD": "EURUSD", "GBPUSD": "GBPUSD", "USDJPY": "USDJPY",
    "AUDUSD": "AUDUSD", "USDCAD": "USDCAD", "USDCHF": "USDCHF",
    "NZDUSD": "NZDUSD", "EURGBP": "EURGBP",
    "BTC-USD": "BTC_USD", "ETH-USD": "ETH_USD",
    "SOL-USD": "SOL_USD", "XRP-USD": "XRP_USD",
    "AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA",
    "TSLA": "TSLA", "SPY": "SPY", "QQQ": "QQQ",
    "XAUUSD": "XAUUSD", "XAGUSD": "XAGUSD",
}


@dataclass
class RunStats:
    """Stats for one backtest variant (simple or HMM)."""
    emitted: int = 0
    wins: int = 0
    losses: int = 0
    opens: int = 0
    pnl: float = 0.0
    regime_counts: dict = field(default_factory=lambda: defaultdict(int))
    regime_wins: dict = field(default_factory=lambda: defaultdict(int))
    regime_losses: dict = field(default_factory=lambda: defaultdict(int))
    regime_pnl: dict = field(default_factory=lambda: defaultdict(float))
    strat_emit: dict = field(default_factory=lambda: defaultdict(int))
    transitions: int = 0


def _load_hmm(symbol: str) -> HMMRegimeDetector | None:
    """Try to load a pre-trained HMM model for the given symbol."""
    model_name = SYMBOL_TO_MODEL.get(symbol, symbol.replace("-", "_"))
    model_path = Path(__file__).resolve().parent.parent / "models" / "regime" / f"{model_name}.pkl"
    if not model_path.exists():
        return None
    det = HMMRegimeDetector()
    det.load_model(model_path)
    return det


def run_variant(df: pd.DataFrame, asset_class: str, regime_mode: str,
                hmm: HMMRegimeDetector | None, stats: RunStats) -> None:
    """Run backtest pipeline with either 'simple' or 'hmm' regime filter."""
    W = 250
    if len(df) < W + 20:
        return

    H = df["high"].values.astype(float)
    L = df["low"].values.astype(float)
    C = df["close"].values.astype(float)
    last_signal_bar = -999
    prev_regime = None

    for start in range(0, len(df) - W - 20, 4):
        current_bar = start + W - 1
        if current_bar - last_signal_bar < 16:
            continue

        window = df.iloc[start:start + W].copy().reset_index(drop=True)

        # Run strategies
        hits = []
        for name, fn in ALL_STRATEGIES:
            try:
                sig = fn(window)
                if sig:
                    hits.append((name, sig))
            except Exception:
                pass

        if not hits:
            continue

        # Regime detection — branch on mode
        if regime_mode == "hmm" and hmm is not None:
            regime, confidence = hmm.predict(window)
            allowed = HMM_REGIME_ALLOWED.get(regime, set())
            # For HMM: also allow trend-following in bear regime (short-biased)
            # The trend alignment gate later handles direction filtering
        else:
            regime = detect_regime(window)
            allowed = REGIME_ALLOWED.get(regime, set())
            confidence = 1.0

        # Track regime transitions
        if prev_regime is not None and regime != prev_regime:
            stats.transitions += 1
        prev_regime = regime

        stats.regime_counts[regime] += 1
        filtered = [(n, s) for n, s in hits if n in allowed]
        if not filtered:
            continue

        # Chop filter (shared)
        if is_choppy(window):
            continue

        # Volume filter (shared)
        if not check_volume(window, asset_class=asset_class):
            continue

        # Confluence + direction
        buys = [x for x in filtered if x[1] == "BUY"]
        sells = [x for x in filtered if x[1] == "SELL"]
        if len(buys) >= 1:
            direction = "BUY"
            strat_name = buys[0][0]
        elif len(sells) >= 1:
            direction = "SELL"
            strat_name = sells[0][0]
        else:
            continue

        # HMM-specific: in bear regime, only allow SELL; in bull, only BUY
        if regime_mode == "hmm" and hmm is not None:
            if regime == "bull" and direction == "SELL":
                continue
            if regime == "bear" and direction == "BUY":
                continue

        # Trend alignment gate (shared)
        ema20 = calc_ema(window["close"], 20).iloc[-1]
        ema50 = calc_ema(window["close"], 50).iloc[-1] if len(window) >= 50 else ema20
        if not (np.isnan(ema20) or np.isnan(ema50)):
            if direction == "BUY" and ema20 < ema50:
                continue
            if direction == "SELL" and ema20 > ema50:
                continue

        # SL/TP
        price = C[current_bar]
        atr_val = calc_atr(window["high"], window["low"], window["close"]).iloc[-1]
        mult = ATR_MULT.get(asset_class, ATR_MULT["forex"])
        fb = PCT_FALLBACK.get(asset_class, PCT_FALLBACK["forex"])
        atr_ok = not np.isnan(atr_val) and atr_val > 0

        if atr_ok:
            sl_d = atr_val * mult["sl"]
            tp_d = atr_val * mult["tp"]
        else:
            sl_d = price * fb["sl"]
            tp_d = price * fb["tp"]

        sl = price - sl_d if direction == "BUY" else price + sl_d
        tp1 = price + tp_d if direction == "BUY" else price - tp_d

        stats.emitted += 1
        stats.strat_emit[strat_name] += 1
        last_signal_bar = current_bar

        # Simulate trade
        idx = current_bar
        if idx + 20 < len(df):
            outcome, pnl = simulate_trade(H, L, C, idx, direction, sl, tp1)
            stats.pnl += pnl
            stats.regime_pnl[regime] += pnl
            if outcome == "TP1":
                stats.wins += 1
                stats.regime_wins[regime] += 1
            elif outcome == "SL":
                stats.losses += 1
                stats.regime_losses[regime] += 1
            else:
                stats.opens += 1


def main():
    print("=" * 80)
    print("  NOVAFX BACKTEST — HMM vs SIMPLE REGIME FILTER COMPARISON")
    print("  90d | 1H | Yahoo Finance API")
    print("=" * 80)

    simple_all = RunStats()
    hmm_all = RunStats()

    per_symbol_simple: dict[str, RunStats] = {}
    per_symbol_hmm: dict[str, RunStats] = {}

    for ac, syms in SYMBOLS.items():
        print(f"\n[{ac.upper()}]")
        for symbol, ticker in syms.items():
            print(f"  {symbol}", end=" ", flush=True)
            try:
                df = fetch_ohlcv(ticker, "1h", "90d")
            except Exception as e:
                print(f"[SKIP: {e}]")
                continue

            if df is None or len(df) < 270:
                print(f"[{len(df) if df is not None else 0} bars — skip]")
                continue
            print(f"({len(df)} bars)", end=" ", flush=True)

            # Load HMM model
            hmm = _load_hmm(symbol)
            hmm_tag = "HMM" if hmm else "no-model"

            # Run both variants
            s_stats = RunStats()
            h_stats = RunStats()

            run_variant(df, ac, "simple", None, s_stats)
            run_variant(df, ac, "hmm", hmm, h_stats)

            per_symbol_simple[symbol] = s_stats
            per_symbol_hmm[symbol] = h_stats

            # Accumulate totals
            for attr in ["emitted", "wins", "losses", "opens", "pnl", "transitions"]:
                setattr(simple_all, attr, getattr(simple_all, attr) + getattr(s_stats, attr))
                setattr(hmm_all, attr, getattr(hmm_all, attr) + getattr(h_stats, attr))
            for r in s_stats.regime_counts:
                simple_all.regime_counts[r] += s_stats.regime_counts[r]
                simple_all.regime_wins[r] += s_stats.regime_wins.get(r, 0)
                simple_all.regime_losses[r] += s_stats.regime_losses.get(r, 0)
                simple_all.regime_pnl[r] += s_stats.regime_pnl.get(r, 0)
            for r in h_stats.regime_counts:
                hmm_all.regime_counts[r] += h_stats.regime_counts[r]
                hmm_all.regime_wins[r] += h_stats.regime_wins.get(r, 0)
                hmm_all.regime_losses[r] += h_stats.regime_losses.get(r, 0)
                hmm_all.regime_pnl[r] += h_stats.regime_pnl.get(r, 0)

            s_wr = s_stats.wins / (s_stats.wins + s_stats.losses) * 100 if s_stats.wins + s_stats.losses else 0
            h_wr = h_stats.wins / (h_stats.wins + h_stats.losses) * 100 if h_stats.wins + h_stats.losses else 0
            print(f"-> S:{s_stats.emitted}sig/{s_wr:.0f}%WR/{s_stats.pnl:+.1f}%  "
                  f"H({hmm_tag}):{h_stats.emitted}sig/{h_wr:.0f}%WR/{h_stats.pnl:+.1f}%")

            time.sleep(0.3)

    # Final comparison report
    print(f"\n{'=' * 80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    def _wr(s: RunStats) -> float:
        t = s.wins + s.losses
        return s.wins / t * 100 if t else 0

    print(f"\n  {'Metric':<25} {'Simple':>12} {'HMM':>12} {'Delta':>10}")
    print(f"  {'-' * 60}")
    print(f"  {'Signals emitted':<25} {simple_all.emitted:>12} {hmm_all.emitted:>12} "
          f"{hmm_all.emitted - simple_all.emitted:>+10}")
    print(f"  {'Wins':<25} {simple_all.wins:>12} {hmm_all.wins:>12} "
          f"{hmm_all.wins - simple_all.wins:>+10}")
    print(f"  {'Losses':<25} {simple_all.losses:>12} {hmm_all.losses:>12} "
          f"{hmm_all.losses - simple_all.losses:>+10}")
    print(f"  {'Win Rate':<25} {_wr(simple_all):>11.1f}% {_wr(hmm_all):>11.1f}% "
          f"{_wr(hmm_all) - _wr(simple_all):>+9.1f}%")
    print(f"  {'Total PnL':<25} {simple_all.pnl:>+11.2f}% {hmm_all.pnl:>+11.2f}% "
          f"{hmm_all.pnl - simple_all.pnl:>+9.2f}%")
    print(f"  {'Regime transitions':<25} {simple_all.transitions:>12} {hmm_all.transitions:>12}")

    # Per-regime breakdown
    print(f"\n  REGIME BREAKDOWN — Simple")
    print(f"  {'Regime':<18} {'Signals':>8} {'Wins':>6} {'Losses':>7} {'WR%':>7} {'PnL%':>9}")
    print(f"  {'-' * 55}")
    for r in sorted(simple_all.regime_counts.keys()):
        cnt = simple_all.regime_counts[r]
        w = simple_all.regime_wins.get(r, 0)
        l = simple_all.regime_losses.get(r, 0)
        wr = w / (w + l) * 100 if w + l else 0
        p = simple_all.regime_pnl.get(r, 0)
        print(f"  {r:<18} {cnt:>8} {w:>6} {l:>7} {wr:>6.1f}% {p:>+8.2f}%")

    print(f"\n  REGIME BREAKDOWN — HMM")
    print(f"  {'Regime':<18} {'Signals':>8} {'Wins':>6} {'Losses':>7} {'WR%':>7} {'PnL%':>9}")
    print(f"  {'-' * 55}")
    for r in sorted(hmm_all.regime_counts.keys()):
        cnt = hmm_all.regime_counts[r]
        w = hmm_all.regime_wins.get(r, 0)
        l = hmm_all.regime_losses.get(r, 0)
        wr = w / (w + l) * 100 if w + l else 0
        p = hmm_all.regime_pnl.get(r, 0)
        print(f"  {r:<18} {cnt:>8} {w:>6} {l:>7} {wr:>6.1f}% {p:>+8.2f}%")

    # Per-symbol comparison
    print(f"\n  PER-SYMBOL COMPARISON")
    print(f"  {'Symbol':<12} {'S.Emit':>6} {'S.WR%':>6} {'S.PnL':>8} "
          f"{'H.Emit':>6} {'H.WR%':>6} {'H.PnL':>8}  Winner")
    print(f"  {'-' * 70}")
    for symbol in per_symbol_simple:
        s = per_symbol_simple[symbol]
        h = per_symbol_hmm.get(symbol, RunStats())
        s_wr = _wr(s)
        h_wr = _wr(h)
        winner = "HMM" if h.pnl > s.pnl else "Simple" if s.pnl > h.pnl else "Tie"
        print(f"  {symbol:<12} {s.emitted:>6} {s_wr:>5.1f}% {s.pnl:>+7.2f}% "
              f"{h.emitted:>6} {h_wr:>5.1f}% {h.pnl:>+7.2f}%  {winner}")

    hmm_wins = sum(1 for sym in per_symbol_simple
                   if per_symbol_hmm.get(sym, RunStats()).pnl > per_symbol_simple[sym].pnl)
    total = len(per_symbol_simple)
    print(f"\n  HMM wins {hmm_wins}/{total} symbols by PnL")
    print()


if __name__ == "__main__":
    main()

"""
NovaFX Backtest — Signal Scoring A/B Comparison.

Runs the pipeline twice:
  A) Unscored: current binary emit/block filters
  B) Scored: weighted confidence scoring with threshold gate

Reports: signal count, win rate, PnL, confidence distribution.
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
    fetch_ohlcv, calc_ema, calc_atr, calc_rsi,
    ALL_STRATEGIES, REGIME_ALLOWED, ATR_MULT, PCT_FALLBACK,
    detect_regime, is_choppy, check_volume, simulate_trade,
)
from src.signals.signal_scorer import SignalScorer

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


@dataclass
class RunStats:
    emitted: int = 0
    wins: int = 0
    losses: int = 0
    opens: int = 0
    pnl: float = 0.0
    confidences: list = field(default_factory=list)
    win_confs: list = field(default_factory=list)
    loss_confs: list = field(default_factory=list)
    sized_pnl: float = 0.0  # PnL with confidence-based sizing


def _vol_ratio(df: pd.DataFrame, lookback: int = 20) -> float:
    """Current volume / 20-bar average volume."""
    if "volume" not in df.columns or len(df) < lookback + 1:
        return 1.0
    vols = df["volume"].tail(lookback + 1)
    avg = vols.iloc[:-1].mean()
    if avg == 0:
        return 1.0
    return float(vols.iloc[-1] / avg)


def _get_hour(df: pd.DataFrame, idx: int) -> int | None:
    """Get UTC hour at index if timestamp available."""
    if "timestamp" in df.columns:
        ts = df["timestamp"].iloc[idx]
        if hasattr(ts, "hour"):
            return ts.hour
    return None


def run_pipeline(df: pd.DataFrame, asset_class: str, scorer: SignalScorer | None,
                 stats: RunStats) -> None:
    """Run backtest pipeline. If scorer is None, run unscored (current behavior)."""
    W = 250
    if len(df) < W + 20:
        return

    H = df["high"].values.astype(float)
    L = df["low"].values.astype(float)
    C = df["close"].values.astype(float)
    last_signal_bar = -999

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

        # Regime filter
        regime = detect_regime(window)
        allowed = REGIME_ALLOWED.get(regime, set())
        filtered = [(n, s) for n, s in hits if n in allowed]
        if not filtered:
            continue

        # Chop filter
        if is_choppy(window):
            continue

        # Volume filter
        if not check_volume(window, asset_class=asset_class):
            continue

        # Direction selection
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

        # Trend alignment gate (still applied in both modes)
        ema20 = calc_ema(window["close"], 20).iloc[-1]
        ema50 = calc_ema(window["close"], 50).iloc[-1] if len(window) >= 50 else ema20
        if not (np.isnan(ema20) or np.isnan(ema50)):
            if direction == "BUY" and ema20 < ema50:
                continue
            if direction == "SELL" and ema20 > ema50:
                continue

        # Compute SL/TP
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

        # Signal scoring gate
        if scorer is not None:
            vol_r = _vol_ratio(window)
            hour = _get_hour(df, current_bar)

            n_agree = len(buys) if direction == "BUY" else len(sells)
            ctx = {
                "direction": direction,
                "df": window,
                "regime": regime,
                "regime_allowed": allowed,
                "strategy": strat_name,
                "asset_class": asset_class,
                "atr_val": float(atr_val) if atr_ok else 0.0,
                "sl_dist": sl_d,
                "tp_dist": tp_d,
                "hour_utc": hour,
                "volume_ratio": vol_r,
                "n_strategies_agree": n_agree,
            }
            should_trade, confidence = scorer.should_emit(ctx)
            if not should_trade:
                continue
            size_mult = scorer.position_size_multiplier(confidence)
        else:
            confidence = 1.0
            size_mult = 1.0

        sl = price - sl_d if direction == "BUY" else price + sl_d
        tp1 = price + tp_d if direction == "BUY" else price - tp_d

        stats.emitted += 1
        stats.confidences.append(confidence)
        last_signal_bar = current_bar

        # Simulate trade
        idx = current_bar
        if idx + 20 < len(df):
            outcome, pnl = simulate_trade(H, L, C, idx, direction, sl, tp1)
            stats.pnl += pnl
            stats.sized_pnl += pnl * size_mult
            if outcome == "TP1":
                stats.wins += 1
                stats.win_confs.append(confidence)
                if scorer is not None:
                    scorer.record_outcome(ctx, confidence, won=True)
            elif outcome == "SL":
                stats.losses += 1
                stats.loss_confs.append(confidence)
                if scorer is not None:
                    scorer.record_outcome(ctx, confidence, won=False)
            else:
                stats.opens += 1


def main():
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
    print("=" * 80)
    print(f"  NOVAFX BACKTEST — SIGNAL SCORING A/B COMPARISON")
    print(f"  90d | 1H | Threshold={threshold}")
    print("=" * 80)

    scorer = SignalScorer(threshold=threshold)

    unscored_all = RunStats()
    scored_all = RunStats()
    per_sym_u: dict[str, RunStats] = {}
    per_sym_s: dict[str, RunStats] = {}

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

            u_stats = RunStats()
            s_stats = RunStats()

            run_pipeline(df, ac, None, u_stats)
            run_pipeline(df, ac, scorer, s_stats)

            per_sym_u[symbol] = u_stats
            per_sym_s[symbol] = s_stats

            for attr in ["emitted", "wins", "losses", "opens", "pnl", "sized_pnl"]:
                setattr(unscored_all, attr, getattr(unscored_all, attr) + getattr(u_stats, attr))
                setattr(scored_all, attr, getattr(scored_all, attr) + getattr(s_stats, attr))
            unscored_all.confidences.extend(u_stats.confidences)
            scored_all.confidences.extend(s_stats.confidences)
            scored_all.win_confs.extend(s_stats.win_confs)
            scored_all.loss_confs.extend(s_stats.loss_confs)

            u_wr = u_stats.wins / (u_stats.wins + u_stats.losses) * 100 if u_stats.wins + u_stats.losses else 0
            s_wr = s_stats.wins / (s_stats.wins + s_stats.losses) * 100 if s_stats.wins + s_stats.losses else 0
            print(f"-> U:{u_stats.emitted}sig/{u_wr:.0f}%WR/{u_stats.pnl:+.1f}%  "
                  f"S:{s_stats.emitted}sig/{s_wr:.0f}%WR/{s_stats.pnl:+.1f}%")

            time.sleep(0.3)

    # Report
    def _wr(s: RunStats) -> float:
        t = s.wins + s.losses
        return s.wins / t * 100 if t else 0

    print(f"\n{'=' * 80}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n  {'Metric':<28} {'Unscored':>12} {'Scored':>12} {'Delta':>10}")
    print(f"  {'-' * 63}")
    print(f"  {'Signals emitted':<28} {unscored_all.emitted:>12} {scored_all.emitted:>12} "
          f"{scored_all.emitted - unscored_all.emitted:>+10}")
    print(f"  {'Wins':<28} {unscored_all.wins:>12} {scored_all.wins:>12} "
          f"{scored_all.wins - unscored_all.wins:>+10}")
    print(f"  {'Losses':<28} {unscored_all.losses:>12} {scored_all.losses:>12} "
          f"{scored_all.losses - unscored_all.losses:>+10}")
    print(f"  {'Win Rate':<28} {_wr(unscored_all):>11.1f}% {_wr(scored_all):>11.1f}% "
          f"{_wr(scored_all) - _wr(unscored_all):>+9.1f}%")
    print(f"  {'Total PnL (flat size)':<28} {unscored_all.pnl:>+11.2f}% {scored_all.pnl:>+11.2f}% "
          f"{scored_all.pnl - unscored_all.pnl:>+9.2f}%")
    print(f"  {'PnL (confidence-sized)':<28} {'—':>12} {scored_all.sized_pnl:>+11.2f}%")

    # Confidence distribution
    if scored_all.confidences:
        confs = np.array(scored_all.confidences)
        print(f"\n  CONFIDENCE DISTRIBUTION (scored signals)")
        print(f"  {'Range':<20} {'Count':>6} {'%':>6}")
        print(f"  {'-' * 35}")
        for lo, hi, label in [(0.65, 0.70, "0.65-0.70"), (0.70, 0.75, "0.70-0.75"),
                               (0.75, 0.80, "0.75-0.80"), (0.80, 0.85, "0.80-0.85"),
                               (0.85, 0.90, "0.85-0.90"), (0.90, 1.01, "0.90-1.00")]:
            cnt = int(np.sum((confs >= lo) & (confs < hi)))
            pct = cnt / len(confs) * 100 if len(confs) else 0
            print(f"  {label:<20} {cnt:>6} {pct:>5.1f}%")
        print(f"  {'Total':<20} {len(confs):>6}")
        print(f"  Mean: {np.mean(confs):.4f}  Median: {np.median(confs):.4f}")

    # Win confidence vs loss confidence
    if scored_all.win_confs and scored_all.loss_confs:
        print(f"\n  CONFIDENCE BY OUTCOME")
        print(f"  Avg winner confidence:  {np.mean(scored_all.win_confs):.4f}")
        print(f"  Avg loser confidence:   {np.mean(scored_all.loss_confs):.4f}")
        print(f"  Delta:                  {np.mean(scored_all.win_confs) - np.mean(scored_all.loss_confs):+.4f}")

    # Learned weights
    print(f"\n  FINAL LEARNED WEIGHTS")
    for k, v in sorted(scorer.weights.items()):
        default = 1.0 / len(scorer.weights)
        delta = v - default
        print(f"  {k:<22} {v:.4f}  ({delta:+.4f} from uniform)")

    # Per-symbol comparison
    print(f"\n  PER-SYMBOL COMPARISON")
    print(f"  {'Symbol':<12} {'U.Emit':>6} {'U.WR%':>6} {'U.PnL':>8} "
          f"{'S.Emit':>6} {'S.WR%':>6} {'S.PnL':>8}  Winner")
    print(f"  {'-' * 70}")
    for symbol in per_sym_u:
        u = per_sym_u[symbol]
        s = per_sym_s.get(symbol, RunStats())
        u_wr = _wr(u)
        s_wr = _wr(s)
        winner = "Scored" if s.pnl > u.pnl else "Unscored" if u.pnl > s.pnl else "Tie"
        print(f"  {symbol:<12} {u.emitted:>6} {u_wr:>5.1f}% {u.pnl:>+7.2f}% "
              f"{s.emitted:>6} {s_wr:>5.1f}% {s.pnl:>+7.2f}%  {winner}")

    scored_wins = sum(1 for sym in per_sym_u
                      if per_sym_s.get(sym, RunStats()).pnl > per_sym_u[sym].pnl)
    total = len(per_sym_u)
    print(f"\n  Scored wins {scored_wins}/{total} symbols by PnL")
    print()


if __name__ == "__main__":
    main()

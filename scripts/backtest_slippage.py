"""
NovaFX Backtest — Slippage Impact Analysis.

Runs backtests with and without slippage to measure execution cost impact
on R:R, win rate, and PnL.
"""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.backtest_harness import (
    fetch_ohlcv, calc_ema, calc_atr, calc_rsi,
    ALL_STRATEGIES, REGIME_ALLOWED, ATR_MULT, PCT_FALLBACK,
    detect_regime, is_choppy, check_volume,
)
from src.execution.slippage_model import SlippageModel, SlippageStats, fix_forex_volume

SYMBOLS = {
    "crypto": {
        "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD",
        "SOL-USD": "SOL-USD", "XRP-USD": "XRP-USD",
    },
    "forex": {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",
        "USDJPY": "JPY=X", "USDCAD": "CAD=X",
    },
    "commodities": {
        "XAUUSD": "GC=F", "XAGUSD": "SI=F",
    },
}


@dataclass
class TradeLog:
    wins: int = 0
    losses: int = 0
    opens: int = 0
    pnl: float = 0.0
    win_pnls: list = field(default_factory=list)
    loss_pnls: list = field(default_factory=list)
    intended_rrs: list = field(default_factory=list)
    actual_rrs: list = field(default_factory=list)


def simulate_with_slippage(highs, lows, closes, idx, direction, sl, tp1,
                           actual_entry, bars=20):
    """Simulate trade using actual (slipped) entry, original SL/TP levels."""
    entry = actual_entry
    for j in range(idx + 1, min(idx + bars + 1, len(closes))):
        if direction == "BUY":
            if lows[j] <= sl:
                return "SL", (sl - entry) / entry * 100
            if highs[j] >= tp1:
                return "TP1", (tp1 - entry) / entry * 100
        else:
            if highs[j] >= sl:
                return "SL", (entry - sl) / entry * 100
            if lows[j] <= tp1:
                return "TP1", (entry - tp1) / entry * 100
    last = closes[min(idx + bars, len(closes) - 1)]
    pnl = ((last - entry) / entry if direction == "BUY" else (entry - last) / entry) * 100
    return "OPEN", pnl


def run_backtest(df: pd.DataFrame, asset_class: str, symbol: str,
                 use_slippage: bool, trade_log: TradeLog,
                 slippage_stats: SlippageStats | None = None):
    """Run backtest pipeline with optional slippage."""
    W = 250
    if len(df) < W + 20:
        return

    H = df["high"].values.astype(float)
    L = df["low"].values.astype(float)
    C = df["close"].values.astype(float)
    V = df["volume"].values.astype(float) if "volume" in df.columns else np.ones(len(df))
    last_signal_bar = -999

    slip_model = SlippageModel(symbol=symbol) if use_slippage else None

    for start in range(0, len(df) - W - 20, 4):
        current_bar = start + W - 1
        if current_bar - last_signal_bar < 16:
            continue

        window = df.iloc[start:start + W].copy().reset_index(drop=True)

        # Strategy pipeline (same as harness)
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

        regime = detect_regime(window)
        allowed = REGIME_ALLOWED.get(regime, set())
        filtered = [(n, s) for n, s in hits if n in allowed]
        if not filtered:
            continue
        if is_choppy(window):
            continue
        if not check_volume(window, asset_class=asset_class):
            continue

        buys = [x for x in filtered if x[1] == "BUY"]
        sells = [x for x in filtered if x[1] == "SELL"]
        if buys:
            direction = "BUY"
        elif sells:
            direction = "SELL"
        else:
            continue

        ema20 = calc_ema(window["close"], 20).iloc[-1]
        ema50 = calc_ema(window["close"], 50).iloc[-1] if len(window) >= 50 else ema20
        if not (np.isnan(ema20) or np.isnan(ema50)):
            if direction == "BUY" and ema20 < ema50:
                continue
            if direction == "SELL" and ema20 > ema50:
                continue

        # SL/TP computation
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

        # Intended R:R
        intended_rr = tp_d / sl_d if sl_d > 0 else 0
        trade_log.intended_rrs.append(intended_rr)

        # Apply slippage
        if use_slippage and slip_model is not None:
            # Use fix_forex_volume to get reliable volume across all asset types
            fixed_vol = fix_forex_volume(
                df["volume"].iloc[max(0, current_bar - 20):current_bar],
                df["close"].iloc[max(0, current_bar - 20):current_bar],
                symbol,
            )
            avg_vol_usd = float(fixed_vol.mean()) * price
            vol = float(atr_val / price) if atr_ok else 0.01

            # Get hour
            hour = None
            if "timestamp" in df.columns:
                ts = df["timestamp"].iloc[current_bar]
                if hasattr(ts, "hour"):
                    hour = ts.hour

            actual_entry, record = slip_model.calculate_and_apply(
                price, direction,
                trade_size_usd=1000.0,
                avg_volume_usd=max(avg_vol_usd, 1.0),
                volatility=vol,
                hour_utc=hour,
            )

            # Actual R:R after slippage
            if direction == "BUY":
                actual_tp_dist = tp1 - actual_entry
                actual_sl_dist = actual_entry - sl
            else:
                actual_tp_dist = actual_entry - tp1
                actual_sl_dist = sl - actual_entry
            actual_rr = actual_tp_dist / actual_sl_dist if actual_sl_dist > 0 else 0
            trade_log.actual_rrs.append(actual_rr)
        else:
            actual_entry = price
            trade_log.actual_rrs.append(intended_rr)

        last_signal_bar = current_bar

        # Simulate trade
        idx = current_bar
        if idx + 20 < len(df):
            if use_slippage:
                outcome, pnl = simulate_with_slippage(H, L, C, idx, direction, sl, tp1, actual_entry)
            else:
                # Standard sim (entry = close[idx])
                entry = C[idx]
                outcome = "OPEN"
                pnl = 0.0
                for j in range(idx + 1, min(idx + 21, len(C))):
                    if direction == "BUY":
                        if L[j] <= sl:
                            outcome, pnl = "SL", (sl - entry) / entry * 100
                            break
                        if H[j] >= tp1:
                            outcome, pnl = "TP1", (tp1 - entry) / entry * 100
                            break
                    else:
                        if H[j] >= sl:
                            outcome, pnl = "SL", (entry - sl) / entry * 100
                            break
                        if L[j] <= tp1:
                            outcome, pnl = "TP1", (entry - tp1) / entry * 100
                            break
                else:
                    last_c = C[min(idx + 20, len(C) - 1)]
                    pnl = ((last_c - entry) / entry if direction == "BUY" else (entry - last_c) / entry) * 100

            trade_log.pnl += pnl
            if outcome == "TP1":
                trade_log.wins += 1
                trade_log.win_pnls.append(pnl)
            elif outcome == "SL":
                trade_log.losses += 1
                trade_log.loss_pnls.append(pnl)
            else:
                trade_log.opens += 1

    if use_slippage and slip_model is not None:
        # Copy stats
        if slippage_stats is not None:
            for r in slip_model.stats.records:
                slippage_stats.add(r)


def main():
    print("=" * 80)
    print("  NOVAFX BACKTEST — SLIPPAGE IMPACT ANALYSIS")
    print("=" * 80)

    global_no_slip = TradeLog()
    global_slip = TradeLog()
    global_slip_stats = SlippageStats()
    per_sym = {}

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
                print(f"[{len(df) if df is not None else 0} bars]")
                continue
            print(f"({len(df)} bars)", end=" ", flush=True)

            no_slip = TradeLog()
            with_slip = TradeLog()
            slip_stats = SlippageStats()

            run_backtest(df, ac, symbol, False, no_slip)
            run_backtest(df, ac, symbol, True, with_slip, slip_stats)

            per_sym[symbol] = (no_slip, with_slip, slip_stats)

            # Accumulate globals
            for attr in ["wins", "losses", "opens", "pnl"]:
                setattr(global_no_slip, attr, getattr(global_no_slip, attr) + getattr(no_slip, attr))
                setattr(global_slip, attr, getattr(global_slip, attr) + getattr(with_slip, attr))
            global_no_slip.intended_rrs.extend(no_slip.intended_rrs)
            global_slip.actual_rrs.extend(with_slip.actual_rrs)
            for r in slip_stats.records:
                global_slip_stats.add(r)

            ns_wr = no_slip.wins / (no_slip.wins + no_slip.losses) * 100 if no_slip.wins + no_slip.losses else 0
            ws_wr = with_slip.wins / (with_slip.wins + with_slip.losses) * 100 if with_slip.wins + with_slip.losses else 0
            n_tot = no_slip.wins + no_slip.losses + no_slip.opens
            print(f"-> {n_tot} trades | No-slip: {ns_wr:.0f}%WR/{no_slip.pnl:+.1f}%  "
                  f"Slipped: {ws_wr:.0f}%WR/{with_slip.pnl:+.1f}%")
            time.sleep(0.3)

    # Summary
    def _wr(t): return t.wins / (t.wins + t.losses) * 100 if t.wins + t.losses else 0
    def _rr(t):
        if t.win_pnls and t.loss_pnls:
            return abs(np.mean(t.win_pnls)) / abs(np.mean(t.loss_pnls))
        return 0

    print(f"\n{'=' * 80}")
    print(f"  SUMMARY")
    print(f"{'=' * 80}")

    print(f"\n  {'Metric':<28} {'No Slippage':>14} {'With Slippage':>14} {'Delta':>10}")
    print(f"  {'-' * 68}")
    ns_tot = global_no_slip.wins + global_no_slip.losses + global_no_slip.opens
    ws_tot = global_slip.wins + global_slip.losses + global_slip.opens
    print(f"  {'Trades':<28} {ns_tot:>14} {ws_tot:>14}")
    print(f"  {'Wins':<28} {global_no_slip.wins:>14} {global_slip.wins:>14} "
          f"{global_slip.wins - global_no_slip.wins:>+10}")
    print(f"  {'Losses':<28} {global_no_slip.losses:>14} {global_slip.losses:>14} "
          f"{global_slip.losses - global_no_slip.losses:>+10}")
    print(f"  {'Win Rate':<28} {_wr(global_no_slip):>13.1f}% {_wr(global_slip):>13.1f}% "
          f"{_wr(global_slip) - _wr(global_no_slip):>+9.1f}%")
    print(f"  {'Total PnL':<28} {global_no_slip.pnl:>+13.2f}% {global_slip.pnl:>+13.2f}% "
          f"{global_slip.pnl - global_no_slip.pnl:>+9.2f}%")
    print(f"  {'Actual R:R':<28} {_rr(global_no_slip):>13.2f} {_rr(global_slip):>13.2f}")

    # R:R comparison
    if global_no_slip.intended_rrs and global_slip.actual_rrs:
        avg_intended = np.mean(global_no_slip.intended_rrs)
        avg_actual = np.mean(global_slip.actual_rrs)
        print(f"\n  R:R Analysis:")
        print(f"    Intended avg R:R:  {avg_intended:.3f}")
        print(f"    Actual avg R:R:    {avg_actual:.3f}")
        print(f"    R:R compression:   {(1 - avg_actual / avg_intended) * 100:.2f}%")

    # Slippage report
    print(f"\n  SLIPPAGE STATISTICS")
    print(f"  {'-' * 50}")
    print(f"  Total trades with slippage: {global_slip_stats.n_trades}")
    print(f"  Avg slippage per trade:     {global_slip_stats.avg_slippage_pct:.4f}%")
    print(f"  Max slippage:               {global_slip_stats.max_slippage_pct:.4f}%")
    print(f"  Total slippage cost:        {global_slip_stats.total_cost_pct:.4f}%")

    # Slippage by hour
    if global_slip_stats.by_hour:
        print(f"\n  Slippage by hour (UTC):")
        print(f"  {'Hour':>6} {'Trades':>7} {'Avg Slip':>10}")
        for h in sorted(global_slip_stats.by_hour):
            d = global_slip_stats.by_hour[h]
            avg = d["total"] / d["count"] if d["count"] else 0
            print(f"  {h:>4}:00 {d['count']:>7} {avg:>9.4f}%")

    # Per-symbol table
    print(f"\n  PER-SYMBOL IMPACT")
    print(f"  {'Symbol':<10} {'NS_WR':>6} {'S_WR':>6} {'NS_PnL':>8} {'S_PnL':>8} {'AvgSlip':>8} {'RR_loss':>8}")
    print(f"  {'-' * 58}")
    for sym, (ns, ws, ss) in per_sym.items():
        ns_w = _wr(ns)
        ws_w = _wr(ws)
        avg_s = ss.avg_slippage_pct
        rr_loss = ""
        if ns.intended_rrs and ws.actual_rrs:
            i = np.mean(ns.intended_rrs)
            a = np.mean(ws.actual_rrs)
            rr_loss = f"{(1 - a / i) * 100:.1f}%" if i > 0 else "—"
        print(f"  {sym:<10} {ns_w:>5.1f}% {ws_w:>5.1f}% {ns.pnl:>+7.2f}% {ws.pnl:>+7.2f}% "
              f"{avg_s:>7.4f}% {rr_loss:>8}")

    print()


if __name__ == "__main__":
    main()

"""
Diagnostic: BTC-USD and BNB-USD signal analysis.
No strategy changes — observation only.
"""
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, ".")
from scripts.backtest_harness import (
    fetch_ohlcv, calc_atr, calc_adx, calc_rsi, calc_ema, calc_macd,
    calc_bollinger, calc_hurst, detect_regime, is_choppy, check_volume,
    ALL_STRATEGIES, REGIME_ALLOWED, ATR_MULT, PCT_FALLBACK,
    simulate_trade,
)

DIAG_SYMBOLS = {
    "BTC-USD": {"asset_class": "crypto", "ticker": "BTC-USD"},
    "BNB-USD": {"asset_class": "crypto", "ticker": "BNB-USD"},
}


def run_diagnostic(name, ticker, ac):
    print(f"\n{'='*70}")
    print(f"  DIAGNOSTIC: {name}")
    print(f"{'='*70}")

    df = fetch_ohlcv(ticker, "1h", "90d")
    if df is None or len(df) < 270:
        print(f"  [ERROR] Only {len(df) if df is not None else 0} bars — insufficient")
        return
    print(f"  Bars: {len(df)}")

    H = df["high"].values.astype(float)
    L = df["low"].values.astype(float)
    C = df["close"].values.astype(float)
    W = 250

    # Accumulators
    regime_counts = defaultdict(int)
    strat_raw = defaultdict(int)
    strat_emit = defaultdict(int)
    strat_wins = defaultdict(int)
    strat_losses = defaultdict(int)
    strat_pnl = defaultdict(float)
    strat_directions = defaultdict(lambda: defaultdict(int))  # strat -> {BUY: n, SELL: n}
    atr_pcts = []
    signal_hours = []
    signal_atr_pcts = []  # ATR% at time of signal
    choppy_count = 0
    volume_blocked = 0
    regime_blocked = 0
    total_raw = 0
    total_emitted = 0
    total_wins = 0
    total_losses = 0
    total_opens = 0
    total_pnl = 0.0
    win_pnls = []
    loss_pnls = []

    # Per-regime win rates
    regime_wins = defaultdict(int)
    regime_losses = defaultdict(int)

    # Per-direction stats
    buy_wins = 0
    buy_losses = 0
    sell_wins = 0
    sell_losses = 0

    last_signal_bar = -999  # Cooldown: min 16 bars between signals
    trend_blocked = 0

    for start in range(0, len(df) - W - 20, 4):
        current_bar = start + W - 1
        if current_bar - last_signal_bar < 16:
            continue

        window = df.iloc[start:start + W].copy().reset_index(drop=True)

        # Collect ATR as % of price at every window
        atr_val = calc_atr(window["high"], window["low"], window["close"]).iloc[-1]
        price = C[start + W - 1]
        if not np.isnan(atr_val) and price > 0:
            atr_pcts.append(atr_val / price * 100)

        # Run strategies
        hits = []
        for sname, fn in ALL_STRATEGIES:
            try:
                sig = fn(window)
                if sig:
                    hits.append((sname, sig))
                    strat_raw[sname] += 1
            except Exception:
                pass

        if not hits:
            continue
        total_raw += len(hits)

        # Regime
        regime = detect_regime(window)
        regime_counts[regime] += 1
        allowed = REGIME_ALLOWED.get(regime, set())
        filtered = [(n, s) for n, s in hits if n in allowed]
        regime_blocked += len(hits) - len(filtered)
        if not filtered:
            continue

        # Choppy
        if is_choppy(window):
            choppy_count += len(filtered)
            continue

        # Volume
        if not check_volume(window, asset_class=ac):
            volume_blocked += len(filtered)
            continue

        # Direction selection (matching run_full_backtest logic)
        buys = [x for x in filtered if x[1] == "BUY"]
        sells = [x for x in filtered if x[1] == "SELL"]
        if len(buys) >= 1:
            direction = "BUY"
            non_div = [x for x in buys if x[0] != "rsi_divergence"]
            strat_name = non_div[0][0] if non_div else buys[0][0]
        elif len(sells) >= 1:
            direction = "SELL"
            non_div = [x for x in sells if x[0] != "rsi_divergence"]
            strat_name = non_div[0][0] if non_div else sells[0][0]
        else:
            continue

        # Trend alignment gate: reject BUY if EMA20 < EMA50, SELL if EMA20 > EMA50
        ema20 = calc_ema(window["close"], 20).iloc[-1]
        ema50 = calc_ema(window["close"], 50).iloc[-1] if len(window) >= 50 else ema20
        if not (np.isnan(ema20) or np.isnan(ema50)):
            if direction == "BUY" and ema20 < ema50:
                trend_blocked += 1
                continue
            if direction == "SELL" and ema20 > ema50:
                trend_blocked += 1
                continue

        # Record strategy direction
        strat_directions[strat_name][direction] += 1

        # SL/TP
        mult = ATR_MULT.get(ac, ATR_MULT["forex"])
        fb = PCT_FALLBACK.get(ac, PCT_FALLBACK["forex"])
        atr_ok = not np.isnan(atr_val) and atr_val > 0
        if atr_ok:
            sl_d = atr_val * mult["sl"]
            tp_d = atr_val * mult["tp"]
        else:
            sl_d = price * fb["sl"]
            tp_d = price * fb["tp"]

        sl = price - sl_d if direction == "BUY" else price + sl_d
        tp1 = price + tp_d if direction == "BUY" else price - tp_d

        total_emitted += 1
        strat_emit[strat_name] += 1
        last_signal_bar = current_bar

        # Record signal hour
        ts_idx = start + W - 1
        if "timestamp" in df.columns:
            ts = df["timestamp"].iloc[ts_idx]
            if hasattr(ts, "hour"):
                signal_hours.append(ts.hour)
        signal_atr_pcts.append(atr_val / price * 100 if atr_ok else 0)

        # Simulate
        idx = start + W - 1
        if idx + 20 < len(df):
            outcome, pnl = simulate_trade(H, L, C, idx, direction, sl, tp1)
            total_pnl += pnl
            strat_pnl[strat_name] += pnl

            if outcome == "TP1":
                total_wins += 1
                win_pnls.append(pnl)
                strat_wins[strat_name] += 1
                regime_wins[regime] += 1
                if direction == "BUY":
                    buy_wins += 1
                else:
                    sell_wins += 1
            elif outcome == "SL":
                total_losses += 1
                loss_pnls.append(pnl)
                strat_losses[strat_name] += 1
                regime_losses[regime] += 1
                if direction == "BUY":
                    buy_losses += 1
                else:
                    sell_losses += 1
            else:
                total_opens += 1

    # ── REPORT ──
    tot = total_wins + total_losses + total_opens
    wr = total_wins / tot * 100 if tot else 0

    print(f"\n  SUMMARY: {tot} trades, {wr:.1f}% WR, {total_pnl:+.2f}% PnL")
    print(f"  Raw signals: {total_raw}, Emitted: {total_emitted}")

    # 1. Regime distribution
    print(f"\n  1. REGIME DISTRIBUTION (at signal windows)")
    total_regimes = sum(regime_counts.values())
    if total_regimes:
        for r in ["trending", "mean_reverting", "ranging"]:
            cnt = regime_counts[r]
            pct = cnt / total_regimes * 100
            rw = regime_wins[r]
            rl = regime_losses[r]
            rt = rw + rl
            rwr = rw / rt * 100 if rt else 0
            print(f"    {r:<16} {cnt:>4} ({pct:>5.1f}%)  WR in regime: {rwr:.1f}% ({rw}W/{rl}L)")
    print(f"    Regime blocked: {regime_blocked}")
    print(f"    Choppy blocked: {choppy_count}")
    print(f"    Volume blocked: {volume_blocked}")
    print(f"    Trend gate blocked: {trend_blocked}")

    # 2. Strategy breakdown
    print(f"\n  2. STRATEGY BREAKDOWN (emitted signals)")
    print(f"    {'Strategy':<18} {'Raw':>5} {'Emit':>5} {'Win':>4} {'Loss':>4} {'WR%':>6} {'PnL%':>8}  Direction")
    print(f"    {'-'*75}")
    for sn in sorted(strat_emit.keys()):
        raw = strat_raw.get(sn, 0)
        emi = strat_emit[sn]
        w = strat_wins.get(sn, 0)
        l = strat_losses.get(sn, 0)
        t = w + l
        wr_s = w / t * 100 if t else 0
        p = strat_pnl.get(sn, 0)
        dirs = strat_directions.get(sn, {})
        dir_str = f"BUY={dirs.get('BUY',0)} SELL={dirs.get('SELL',0)}"
        print(f"    {sn:<18} {raw:>5} {emi:>5} {w:>4} {l:>4} {wr_s:>5.1f}% {p:>+7.2f}%  {dir_str}")

    # Direction breakdown
    print(f"\n    Direction summary:")
    bt = buy_wins + buy_losses
    st_ = sell_wins + sell_losses
    print(f"      BUY:  {buy_wins}W / {buy_losses}L = {buy_wins/bt*100 if bt else 0:.1f}% WR")
    print(f"      SELL: {sell_wins}W / {sell_losses}L = {sell_wins/st_*100 if st_ else 0:.1f}% WR")

    # 3. ATR analysis
    print(f"\n  3. ATR AS % OF PRICE")
    if atr_pcts:
        print(f"    All windows:  min={min(atr_pcts):.3f}%  avg={np.mean(atr_pcts):.3f}%  "
              f"median={np.median(atr_pcts):.3f}%  max={max(atr_pcts):.3f}%")
    if signal_atr_pcts:
        print(f"    At signals:   min={min(signal_atr_pcts):.3f}%  avg={np.mean(signal_atr_pcts):.3f}%  "
              f"median={np.median(signal_atr_pcts):.3f}%  max={max(signal_atr_pcts):.3f}%")
        # SL/TP distances implied
        mult = ATR_MULT.get(ac, ATR_MULT["forex"])
        avg_atr = np.mean(signal_atr_pcts)
        print(f"    Implied SL distance: {avg_atr * mult['sl']:.3f}% (ATR × {mult['sl']})")
        print(f"    Implied TP distance: {avg_atr * mult['tp']:.3f}% (ATR × {mult['tp']})")
        if win_pnls:
            print(f"    Avg win PnL:  {np.mean(win_pnls):+.3f}%")
        if loss_pnls:
            print(f"    Avg loss PnL: {np.mean(loss_pnls):+.3f}%")
        if win_pnls and loss_pnls:
            rr = abs(np.mean(win_pnls)) / abs(np.mean(loss_pnls))
            print(f"    Actual R:R:   1:{rr:.2f}")

    # 4. Signal timing
    print(f"\n  4. SIGNAL TIMING (UTC hours)")
    if signal_hours:
        hour_counts = defaultdict(int)
        for h in signal_hours:
            hour_counts[h] += 1
        # Group into sessions
        asia = sum(hour_counts[h] for h in range(0, 8))
        london = sum(hour_counts[h] for h in range(8, 16))
        ny = sum(hour_counts[h] for h in range(16, 24))
        total_h = len(signal_hours)
        print(f"    Asia (00-08):   {asia:>3} ({asia/total_h*100:.0f}%)")
        print(f"    London (08-16): {london:>3} ({london/total_h*100:.0f}%)")
        print(f"    New York (16-24): {ny:>3} ({ny/total_h*100:.0f}%)")
        print(f"    Top 5 hours: ", end="")
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(", ".join(f"{h:02d}:00={c}" for h, c in sorted_hours))

        # Clustering check: are >50% in any 4-hour block?
        for block_start in range(0, 24, 4):
            block = sum(hour_counts[h] for h in range(block_start, block_start + 4))
            if block / total_h > 0.4:
                print(f"    ⚠️  CLUSTER: {block_start:02d}-{block_start+4:02d} UTC has {block/total_h*100:.0f}% of signals")
    else:
        print(f"    [No timestamp data available]")

    # 5. ADX distribution at signal points
    print(f"\n  5. ADDITIONAL OBSERVATIONS")
    if atr_pcts:
        high_vol = sum(1 for a in signal_atr_pcts if a > np.percentile(atr_pcts, 75))
        low_vol = sum(1 for a in signal_atr_pcts if a < np.percentile(atr_pcts, 25))
        print(f"    Signals during high-vol (>75th pct): {high_vol}/{total_emitted} ({high_vol/total_emitted*100:.0f}%)")
        print(f"    Signals during low-vol (<25th pct):  {low_vol}/{total_emitted} ({low_vol/total_emitted*100:.0f}%)")

    # Consecutive loss streaks
    if loss_pnls:
        max_streak = 0
        cur_streak = 0
        # Reconstruct outcome sequence
        outcomes = []
        # Re-run to get sequence (simplified — just count from pnls)
        print(f"    Max single loss: {min(loss_pnls):.3f}%")
        print(f"    Max single win:  {max(win_pnls):.3f}%" if win_pnls else "")


if __name__ == "__main__":
    for name, info in DIAG_SYMBOLS.items():
        try:
            run_diagnostic(name, info["ticker"], info["asset_class"])
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
        time.sleep(0.5)
    print()

"""
Full 20-symbol backtest with detailed reporting.
Uses backtest_harness.py functions.
"""
import sys
import time
import numpy as np
from dataclasses import dataclass, field

sys.path.insert(0, ".")
from scripts.backtest_harness import (
    fetch_ohlcv, calc_atr, calc_adx, calc_ema, calc_rsi, calc_macd,
    calc_bollinger, calc_hurst, detect_regime, is_choppy, check_volume,
    ALL_STRATEGIES, REGIME_ALLOWED, ATR_MULT, PCT_FALLBACK,
)

SYMBOLS = {
    "forex": {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "USDJPY=X",
              "AUDUSD": "AUDUSD=X", "USDCAD": "USDCAD=X"},
    "stock": {"AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA", "AMZN": "AMZN",
              "TSLA": "TSLA", "META": "META", "GOOGL": "GOOGL"},
    "crypto": {"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
               "BNB": "BNB-USD", "XRP": "XRP-USD"},
}

@dataclass
class SymbolResult:
    symbol: str
    asset_class: str
    raw: int = 0
    regime_f: int = 0
    choppy_f: int = 0
    volume_f: int = 0
    emitted: int = 0
    wins: int = 0
    losses: int = 0
    opens: int = 0
    pnl: float = 0.0
    win_pnls: list = field(default_factory=list)
    loss_pnls: list = field(default_factory=list)
    strat_raw: dict = field(default_factory=dict)
    strat_emit: dict = field(default_factory=dict)
    strat_wins: dict = field(default_factory=dict)
    strat_losses: dict = field(default_factory=dict)
    strat_pnl: dict = field(default_factory=dict)
    regimes: dict = field(default_factory=lambda: {"trending": 0, "mean_reverting": 0, "ranging": 0})
    atr_sls: list = field(default_factory=list)


def simulate(H, L, C, idx, direction, sl, tp1, bars=20):
    entry = C[idx]
    for j in range(idx + 1, min(idx + bars + 1, len(C))):
        if direction == "BUY":
            if L[j] <= sl: return "SL", (sl - entry) / entry * 100
            if H[j] >= tp1: return "TP1", (tp1 - entry) / entry * 100
        else:
            if H[j] >= sl: return "SL", (entry - sl) / entry * 100
            if L[j] <= tp1: return "TP1", (entry - tp1) / entry * 100
    last = C[min(idx + bars, len(C) - 1)]
    return "OPEN", ((last - entry) / entry if direction == "BUY" else (entry - last) / entry) * 100


def run_symbol(nova_sym, yf_ticker, ac):
    r = SymbolResult(symbol=nova_sym, asset_class=ac)
    try:
        df = fetch_ohlcv(yf_ticker, "1h", "90d")
    except Exception as e:
        print(f"    [ERROR: {e}]")
        return r
    if df is None or len(df) < 270:
        print(f"    {len(df) if df is not None else 0} bars [insufficient]")
        return r

    print(f"    {len(df)} bars", end="", flush=True)
    H = df["high"].values.astype(float)
    L = df["low"].values.astype(float)
    C = df["close"].values.astype(float)
    W = 250

    for start in range(0, len(df) - W - 20, 4):
        window = df.iloc[start:start + W].copy().reset_index(drop=True)

        hits = []
        for name, fn in ALL_STRATEGIES:
            try:
                sig = fn(window)
                if sig:
                    hits.append((name, sig))
                    r.strat_raw[name] = r.strat_raw.get(name, 0) + 1
            except Exception:
                pass
        if not hits:
            continue
        r.raw += len(hits)

        regime = detect_regime(window)
        r.regimes[regime] = r.regimes.get(regime, 0) + 1
        allowed = REGIME_ALLOWED.get(regime, set())
        filtered = [(n, s) for n, s in hits if n in allowed]
        r.regime_f += len(hits) - len(filtered)
        if not filtered:
            continue

        if is_choppy(window):
            r.choppy_f += len(filtered)
            continue

        if not check_volume(window, asset_class=ac):
            r.volume_f += len(filtered)
            continue

        # Prefer non-divergence strategies when both fire
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

        price = C[start + W - 1]
        atr_val = calc_atr(window["high"], window["low"], window["close"]).iloc[-1]
        atr_ok = not np.isnan(atr_val) and atr_val > 0
        mult = ATR_MULT.get(ac, ATR_MULT.get("forex", {"sl": 1.5, "tp": 2.0}))
        fb = PCT_FALLBACK.get(ac, PCT_FALLBACK.get("forex", {"sl": 0.003, "tp": 0.006}))

        if atr_ok:
            sl_d = atr_val * mult["sl"]; tp_d = atr_val * mult["tp"]
            r.atr_sls.append(sl_d / price * 100)
        else:
            sl_d = price * fb["sl"]; tp_d = price * fb["tp"]

        sl = price - sl_d if direction == "BUY" else price + sl_d
        tp1 = price + tp_d if direction == "BUY" else price - tp_d

        r.emitted += 1
        r.strat_emit[strat_name] = r.strat_emit.get(strat_name, 0) + 1

        idx = start + W - 1
        if idx + 20 < len(df):
            outcome, pnl = simulate(H, L, C, idx, direction, sl, tp1)
            r.pnl += pnl
            r.strat_pnl[strat_name] = r.strat_pnl.get(strat_name, 0) + pnl
            if outcome == "TP1":
                r.wins += 1; r.win_pnls.append(pnl)
                r.strat_wins[strat_name] = r.strat_wins.get(strat_name, 0) + 1
            elif outcome == "SL":
                r.losses += 1; r.loss_pnls.append(pnl)
                r.strat_losses[strat_name] = r.strat_losses.get(strat_name, 0) + 1
            else:
                r.opens += 1

    tot = r.wins + r.losses + r.opens
    wr = r.wins / tot * 100 if tot else 0
    print(f" -> {tot} trades, {wr:.0f}% WR, {r.pnl:+.2f}% PnL")
    return r


def main():
    print("=" * 75)
    print("  NOVAFX FULL BACKTEST — 17 symbols")
    print("  90d | 1H | Yahoo Finance API | Tuned filters")
    print("=" * 75)

    results: list[SymbolResult] = []
    for ac, syms in SYMBOLS.items():
        print(f"\n[{ac.upper()}]")
        for sym, tick in syms.items():
            print(f"  {sym} ({tick})", end=" ", flush=True)
            r = run_symbol(sym, tick, ac)
            results.append(r)
            time.sleep(0.4)

    # ── 1. Filter funnel per asset class ──
    print(f"\n{'='*75}")
    print("  1. FILTER FUNNEL PER ASSET CLASS")
    print(f"{'='*75}")
    for ac in SYMBOLS:
        rs = [r for r in results if r.asset_class == ac]
        raw = sum(r.raw for r in rs)
        reg = sum(r.regime_f for r in rs)
        chp = sum(r.choppy_f for r in rs)
        vol = sum(r.volume_f for r in rs)
        emi = sum(r.emitted for r in rs)
        after_regime = raw - reg
        after_choppy = after_regime - chp
        after_volume = after_choppy - vol
        print(f"\n  {ac.upper()}: Raw({raw}) → -regime({reg}) = {after_regime}"
              f" → -choppy({chp}) = {after_choppy}"
              f" → -volume({vol}) = {after_volume}"
              f" → Emitted: {emi}")
        # Over-filter alert
        if raw > 0 and reg / raw > 0.6:
            print(f"    ⚠️  OVER-FILTER: regime blocks {reg/raw*100:.0f}% of raw")
        if after_regime > 0 and chp / after_regime > 0.6:
            print(f"    ⚠️  OVER-FILTER: choppy blocks {chp/after_regime*100:.0f}% of post-regime")
        if after_choppy > 0 and vol / after_choppy > 0.6:
            print(f"    ⚠️  OVER-FILTER: volume blocks {vol/after_choppy*100:.0f}% of post-choppy")

    # ── 2. Per-strategy breakdown ──
    print(f"\n{'='*75}")
    print("  2. PER-STRATEGY BREAKDOWN")
    print(f"{'='*75}")
    all_strats = set()
    for r in results:
        all_strats.update(r.strat_raw.keys())
    hdr = f"  {'Strategy':<20} {'Raw':>5} {'Emit':>5} {'Win':>4} {'Loss':>4} {'WR%':>6} {'AvgRR':>7}"
    print(hdr)
    print("  " + "-" * 55)
    for sn in sorted(all_strats):
        raw = sum(r.strat_raw.get(sn, 0) for r in results)
        emi = sum(r.strat_emit.get(sn, 0) for r in results)
        wins = sum(r.strat_wins.get(sn, 0) for r in results)
        losses = sum(r.strat_losses.get(sn, 0) for r in results)
        tot = wins + losses
        wr = wins / tot * 100 if tot else 0
        # Avg RR from strategy-level PnL
        pnl_total = sum(r.strat_pnl.get(sn, 0) for r in results)
        avg_rr_str = "—"
        if tot > 0:
            # Collect all win/loss pnls for this strategy (approximate)
            avg_rr_str = f"{pnl_total/tot:+.2f}%"
        print(f"  {sn:<20} {raw:>5} {emi:>5} {wins:>4} {losses:>4} {wr:>5.1f}% {avg_rr_str:>7}")
        if tot > 5 and wr < 30:
            print(f"    ⚠️  WIN RATE BELOW 30%")

    # ── 3. Per-asset-class summary ──
    print(f"\n{'='*75}")
    print("  3. PER-ASSET-CLASS SUMMARY")
    print(f"{'='*75}")
    hdr = f"  {'Class':<10} {'Emit':>5} {'Win':>4} {'Loss':>4} {'WR%':>6} {'AvgRR':>7} {'PnL%':>8}"
    print(hdr)
    print("  " + "-" * 50)
    for ac in SYMBOLS:
        rs = [r for r in results if r.asset_class == ac]
        emi = sum(r.emitted for r in rs)
        wins = sum(r.wins for r in rs)
        losses = sum(r.losses for r in rs)
        pnl = sum(r.pnl for r in rs)
        tot = wins + losses
        wr = wins / tot * 100 if tot else 0
        aw = np.mean([p for r in rs for p in r.win_pnls]) if any(r.win_pnls for r in rs) else 0
        al = abs(np.mean([p for r in rs for p in r.loss_pnls])) if any(r.loss_pnls for r in rs) else 1
        rr = aw / al if al > 0 else 0
        print(f"  {ac:<10} {emi:>5} {wins:>4} {losses:>4} {wr:>5.1f}% {f'1:{rr:.1f}':>7} {pnl:>+7.2f}%")

    # ── 4. Top 5 / Bottom 5 by win rate ──
    print(f"\n{'='*75}")
    print("  4. TOP 5 / BOTTOM 5 BY WIN RATE (min 3 trades)")
    print(f"{'='*75}")
    ranked = []
    for r in results:
        tot = r.wins + r.losses
        if tot >= 3:
            ranked.append((r.symbol, r.asset_class, r.wins / tot * 100, tot, r.pnl))
    ranked.sort(key=lambda x: x[2], reverse=True)
    print("\n  TOP 5:")
    for sym, ac, wr, tot, pnl in ranked[:5]:
        print(f"    {sym:<10} ({ac:<6}) WR={wr:>5.1f}%  trades={tot}  PnL={pnl:+.2f}%")
    print("\n  BOTTOM 5:")
    for sym, ac, wr, tot, pnl in ranked[-5:]:
        print(f"    {sym:<10} ({ac:<6}) WR={wr:>5.1f}%  trades={tot}  PnL={pnl:+.2f}%")

    # ── 5. Zero-signal symbols ──
    print(f"\n{'='*75}")
    print("  5. ZERO-SIGNAL SYMBOLS")
    print(f"{'='*75}")
    zeros = [r for r in results if r.emitted == 0]
    if zeros:
        for r in zeros:
            print(f"    ⚠️  {r.symbol} ({r.asset_class}): {r.raw} raw hits, 0 emitted")
    else:
        print("    [OK] All symbols emitted at least 1 signal")

    # ── 6. Strategies below 30% ──
    print(f"\n{'='*75}")
    print("  6. STRATEGIES BELOW 30% WIN RATE (min 5 trades)")
    print(f"{'='*75}")
    flagged = False
    for sn in sorted(all_strats):
        wins = sum(r.strat_wins.get(sn, 0) for r in results)
        losses = sum(r.strat_losses.get(sn, 0) for r in results)
        tot = wins + losses
        if tot >= 5:
            wr = wins / tot * 100
            if wr < 30:
                print(f"    ⚠️  {sn}: {wr:.1f}% ({wins}W/{losses}L)")
                flagged = True
    if not flagged:
        print("    [OK] All strategies above 30%")

    # ── 7. Over-filter alerts ──
    print(f"\n{'='*75}")
    print("  7. OVER-FILTER ALERTS (>60% block rate)")
    print(f"{'='*75}")
    flagged = False
    for ac in SYMBOLS:
        rs = [r for r in results if r.asset_class == ac]
        raw = sum(r.raw for r in rs)
        reg = sum(r.regime_f for r in rs)
        chp = sum(r.choppy_f for r in rs)
        vol = sum(r.volume_f for r in rs)
        after_regime = raw - reg
        after_choppy = after_regime - chp
        if raw > 10 and reg / raw > 0.6:
            print(f"    ⚠️  {ac} regime: blocks {reg}/{raw} = {reg/raw*100:.0f}%")
            flagged = True
        if after_regime > 10 and chp / after_regime > 0.6:
            print(f"    ⚠️  {ac} choppy: blocks {chp}/{after_regime} = {chp/after_regime*100:.0f}%")
            flagged = True
        if after_choppy > 10 and vol / after_choppy > 0.6:
            print(f"    ⚠️  {ac} volume: blocks {vol}/{after_choppy} = {vol/after_choppy*100:.0f}%")
            flagged = True
    if not flagged:
        print("    [OK] No filter exceeds 60% block rate")

    print()


if __name__ == "__main__":
    main()

"""
NovaFX Signal Quality Backtest — commit fc1986c

Self-contained backtest using pure numpy (no ta-lib dependency).
Fetches 90d of 1H candles via Yahoo Finance Chart API.
Replicates the signal pipeline: 5 strategies, regime, chop, volume, confluence.

RULE 1 compliant: Yahoo Finance only, NOT TwelveData.
"""

import time
from dataclasses import dataclass, field

import httpx
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Pure numpy indicators (no ta-lib)
# ---------------------------------------------------------------------------

def ema(values: np.ndarray, period: int) -> np.ndarray:
    result = np.full_like(values, np.nan, dtype=float)
    if len(values) < period:
        return result
    k = 2.0 / (period + 1)
    result[period - 1] = np.mean(values[:period])
    for i in range(period, len(values)):
        result[i] = values[i] * k + result[i - 1] * (1 - k)
    return result


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    result = np.full_like(closes, np.nan, dtype=float)
    if len(closes) < period + 1:
        return result
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            result[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return result


def macd(closes: np.ndarray):
    ema12 = ema(closes, 12)
    ema26 = ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = np.full_like(macd_line, np.nan)
    valid = ~np.isnan(macd_line)
    if valid.sum() >= 9:
        first_valid = np.argmax(valid)
        signal_line[first_valid + 8] = np.mean(macd_line[first_valid:first_valid + 9])
        k = 2.0 / 10
        for i in range(first_valid + 9, len(macd_line)):
            if not np.isnan(macd_line[i]):
                signal_line[i] = macd_line[i] * k + signal_line[i - 1] * (1 - k)
    return macd_line, signal_line


def bollinger(closes: np.ndarray, period: int = 20, dev: float = 2.0):
    mid = np.full_like(closes, np.nan, dtype=float)
    upper = np.full_like(closes, np.nan, dtype=float)
    lower = np.full_like(closes, np.nan, dtype=float)
    for i in range(period - 1, len(closes)):
        window = closes[i - period + 1:i + 1]
        m = np.mean(window)
        s = np.std(window)
        mid[i] = m
        upper[i] = m + dev * s
        lower[i] = m - dev * s
    return upper, mid, lower


def compute_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    if len(tr) < period:
        return None
    val = float(np.mean(tr[-period:]))
    return val if not np.isnan(val) else None


def adx(highs, lows, closes, period=14):
    """Simplified ADX calculation."""
    if len(closes) < period * 2:
        return np.nan
    plus_dm = np.maximum(np.diff(highs), 0)
    minus_dm = np.maximum(-np.diff(lows), 0)
    mask = plus_dm > minus_dm
    plus_dm[~mask] = 0
    minus_dm[mask] = 0
    tr = np.maximum(highs[1:] - lows[1:],
                    np.maximum(np.abs(highs[1:] - closes[:-1]),
                               np.abs(lows[1:] - closes[:-1])))
    atr_vals = np.convolve(tr, np.ones(period)/period, mode='valid')
    plus_di = np.convolve(plus_dm, np.ones(period)/period, mode='valid')
    minus_di = np.convolve(minus_dm, np.ones(period)/period, mode='valid')
    if len(atr_vals) == 0 or len(plus_di) == 0:
        return np.nan
    n = min(len(atr_vals), len(plus_di), len(minus_di))
    plus_di = plus_di[:n] / atr_vals[:n] * 100
    minus_di = minus_di[:n] / atr_vals[:n] * 100
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10) * 100
    if len(dx) < period:
        return float(np.mean(dx)) if len(dx) > 0 else np.nan
    return float(np.mean(dx[-period:]))


# ---------------------------------------------------------------------------
# Signal strategies (replicating signal_engine.py logic)
# ---------------------------------------------------------------------------

def strat_ema_cross(closes):
    e9 = ema(closes, 9)
    e21 = ema(closes, 21)
    if np.isnan(e9[-1]) or np.isnan(e9[-2]) or np.isnan(e21[-1]) or np.isnan(e21[-2]):
        return None
    slope = e21[-1] - e21[-3] if len(e21) >= 3 and not np.isnan(e21[-3]) else 0
    if e9[-2] <= e21[-2] and e9[-1] > e21[-1]:
        return "BUY" if slope >= 0 else None
    if e9[-2] >= e21[-2] and e9[-1] < e21[-1]:
        return "SELL" if slope <= 0 else None
    return None


def strat_rsi_adaptive(closes):
    r = rsi(closes)
    if np.isnan(r[-1]):
        return None
    sma50 = np.mean(closes[-50:]) if len(closes) >= 50 else closes[-1]
    if closes[-1] > sma50:
        buy_t, sell_t = 40, 80
    else:
        buy_t, sell_t = 20, 60
    if r[-1] < buy_t:
        return "BUY"
    if r[-1] > sell_t:
        return "SELL"
    return None


def strat_macd_zero(closes):
    ml, sl = macd(closes)
    if any(np.isnan(v) for v in [ml[-1], ml[-2], sl[-1], sl[-2]]):
        return None
    if ml[-2] <= sl[-2] and ml[-1] > sl[-1]:
        return "BUY" if ml[-1] >= 0 else None
    if ml[-2] >= sl[-2] and ml[-1] < sl[-1]:
        return "SELL" if ml[-1] <= 0 else None
    return None


def strat_bb_reversion(closes):
    u, m, l = bollinger(closes)
    if any(np.isnan(v) for v in [u[-1], l[-1], m[-1]]):
        return None
    if len(closes) >= 25 and not np.isnan(m[-5]):
        bw = (u[-1] - l[-1]) / m[-1]
        bw_prev = (u[-5] - l[-5]) / m[-5] if not any(np.isnan(v) for v in [u[-5], l[-5], m[-5]]) else bw
        if bw > bw_prev * 1.3:
            return None
        slope = abs(m[-1] - m[-5]) / m[-1]
        if slope > 0.005:
            return None
    if closes[-1] <= l[-1]:
        return "BUY"
    if closes[-1] >= u[-1]:
        return "SELL"
    return None


def strat_rsi_divergence(closes):
    r = rsi(closes)
    if len(closes) < 50 or np.any(np.isnan(r[-30:])):
        return None
    w = closes[-30:]
    rw = r[-30:]
    lows, highs = [], []
    for i in range(2, len(w) - 2):
        if w[i] < w[i-1] and w[i] < w[i-2] and w[i] < w[i+1] and w[i] < w[i+2]:
            lows.append(i)
        if w[i] > w[i-1] and w[i] > w[i-2] and w[i] > w[i+1] and w[i] > w[i+2]:
            highs.append(i)
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if w[i2] < w[i1] and rw[i2] > rw[i1] and rw[i2] < 40:
            return "BUY"
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if w[i2] > w[i1] and rw[i2] < rw[i1] and rw[i2] > 60:
            return "SELL"
    return None


ALL_STRATS = [
    ("ema_cross", strat_ema_cross),
    ("rsi_adaptive", strat_rsi_adaptive),
    ("macd_zero", strat_macd_zero),
    ("bb_reversion", strat_bb_reversion),
    ("rsi_divergence", strat_rsi_divergence),
]

# Regime strategy map
REGIME_ALLOWED = {
    "trending": {"ema_cross", "macd_zero"},
    "mean_reverting": {"rsi_adaptive", "bb_reversion", "rsi_divergence"},
    "ranging": set(),
}


def detect_regime_simple(highs, lows, closes):
    adx_val = adx(highs, lows, closes)
    if np.isnan(adx_val):
        return "ranging"
    u, m, l = bollinger(closes)
    if not np.isnan(u[-1]) and not np.isnan(l[-1]) and not np.isnan(m[-1]) and m[-1] > 0:
        bw = (u - l) / m
        bw_valid = bw[~np.isnan(bw)]
        if len(bw_valid) > 0:
            pct = (bw_valid < bw_valid[-1]).sum() / len(bw_valid) * 100
            if adx_val > 25 and pct > 50:
                return "trending"
            if adx_val < 20 and pct < 30:
                return "mean_reverting"
    return "ranging"


def is_choppy_simple(highs, lows, closes):
    count = 0
    adx_val = adx(highs, lows, closes)
    if not np.isnan(adx_val) and adx_val < 20:
        count += 1
    atr_val = compute_atr(highs, lows, closes)
    if atr_val and len(closes) > 30:
        recent = atr_val
        older_atrs = []
        for i in range(len(closes) - 28, len(closes) - 14):
            a = compute_atr(highs[:i+14], lows[:i+14], closes[:i+14])
            if a:
                older_atrs.append(a)
        if older_atrs and (np.array(older_atrs) < recent).mean() < 0.3:
            count += 1
    return count >= 2


ATR_CONFIG = {
    "forex": {"sl": 1.5, "tp": 2.0},
    "crypto": {"sl": 2.0, "tp": 3.0},
    "stocks": {"sl": 2.0, "tp": 3.0},
    "commodities": {"sl": 2.0, "tp": 3.0},
}

FALLBACK = {
    "forex": {"sl": 0.003, "tp": 0.006},
    "crypto": {"sl": 0.015, "tp": 0.03},
    "stocks": {"sl": 0.01, "tp": 0.02},
    "commodities": {"sl": 0.008, "tp": 0.016},
}


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_yahoo(ticker):
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {"range": "90d", "interval": "1h", "includePrePost": "false"}
    try:
        resp = httpx.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"},
                         timeout=15, follow_redirects=True)
        if resp.status_code != 200:
            return None
        data = resp.json()
        result = data.get("chart", {}).get("result", [])
        if not result:
            return None
        ts = result[0]["timestamp"]
        q = result[0]["indicators"]["quote"][0]
        df = pd.DataFrame({
            "open": q.get("open"), "high": q.get("high"),
            "low": q.get("low"), "close": q.get("close"),
            "volume": q.get("volume"),
        }, index=pd.to_datetime(ts, unit="s", utc=True))
        df = df.dropna(subset=["close"])
        df["volume"] = df["volume"].fillna(0)
        return df if len(df) >= 100 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------

SYMBOLS = {
    "forex": {"EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
              "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X", "USDCHF": "CHF=X",
              "NZDUSD": "NZDUSD=X", "EURGBP": "EURGBP=X"},
    "stocks": {"AAPL": "AAPL", "MSFT": "MSFT", "NVDA": "NVDA",
               "TSLA": "TSLA", "SPY": "SPY", "QQQ": "QQQ"},
    "crypto": {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD",
               "SOLUSDT": "SOL-USD", "XRPUSDT": "XRP-USD"},
    "commodities": {"XAUUSD": "GC=F", "XAGUSD": "SI=F"},
}


@dataclass
class S:
    raw: int = 0; regime_f: int = 0; choppy_f: int = 0; vol_f: int = 0
    no_conf: int = 0; emitted: int = 0; nans: int = 0
    wins: int = 0; losses: int = 0; opens: int = 0
    pnl: float = 0.0; win_pnls: list = field(default_factory=list)
    loss_pnls: list = field(default_factory=list); atr_sls: list = field(default_factory=list)
    regimes: dict = field(default_factory=lambda: {"trending": 0, "mean_reverting": 0, "ranging": 0})
    strats: dict = field(default_factory=dict); atr_n: int = 0


def run(symbol, ticker, ac, s):
    df = fetch_yahoo(ticker)
    if df is None:
        print(f"    [no data]"); return
    print(f"    {len(df)} bars", end="", flush=True)
    W = 250
    if len(df) < W + 20:
        print(" [insufficient]"); return
    C = df["close"].values.astype(float)
    H = df["high"].values.astype(float)
    L = df["low"].values.astype(float)
    V = df["volume"].values.astype(float)
    tc = 0
    for i in range(0, len(df) - W - 20, 4):
        c = C[i:i+W]; h = H[i:i+W]; l = L[i:i+W]; v = V[i:i+W]
        hits = []
        for name, fn in ALL_STRATS:
            try:
                sig = fn(c)
                if sig:
                    hits.append((name, sig))
                    s.strats[name] = s.strats.get(name, 0) + 1
            except Exception:
                s.nans += 1
        if not hits:
            continue
        s.raw += len(hits)
        regime = detect_regime_simple(h, l, c)
        s.regimes[regime] = s.regimes.get(regime, 0) + 1
        allowed = REGIME_ALLOWED.get(regime, set())
        filtered = [(n, sig) for n, sig in hits if n in allowed]
        s.regime_f += len(hits) - len(filtered)
        if not filtered:
            continue
        if is_choppy_simple(h, l, c):
            s.choppy_f += len(filtered); continue
        avg_v = np.mean(v[:-1]) if len(v) > 1 else 0
        if avg_v > 0 and v[-1] < avg_v:
            s.vol_f += len(filtered); continue
        buys = [x for x in filtered if x[1] == "BUY"]
        sells = [x for x in filtered if x[1] == "SELL"]
        if len(buys) >= 2:
            d = "BUY"
        elif len(sells) >= 2:
            d = "SELL"
        else:
            s.no_conf += len(filtered); continue
        price = c[-1]
        atr_val = compute_atr(h, l, c)
        cfg = ATR_CONFIG.get(ac, ATR_CONFIG["forex"])
        fb = FALLBACK.get(ac, FALLBACK["forex"])
        if atr_val and atr_val > 0:
            sl_d = atr_val * cfg["sl"]; tp_d = atr_val * cfg["tp"]
            s.atr_sls.append(sl_d / price * 100); s.atr_n += 1
        else:
            sl_d = price * fb["sl"]; tp_d = price * fb["tp"]
        sl = price - sl_d if d == "BUY" else price + sl_d
        tp1 = price + tp_d if d == "BUY" else price - tp_d
        s.emitted += 1
        idx = i + W - 1
        if idx + 20 < len(df):
            outcome, pnl_pct = "OPEN", 0.0
            entry = C[idx]
            for j in range(idx + 1, min(idx + 21, len(df))):
                if d == "BUY":
                    if L[j] <= sl:
                        outcome = "SL"; pnl_pct = (sl - entry) / entry * 100; break
                    if H[j] >= tp1:
                        outcome = "TP1"; pnl_pct = (tp1 - entry) / entry * 100; break
                else:
                    if H[j] >= sl:
                        outcome = "SL"; pnl_pct = (entry - sl) / entry * 100; break
                    if L[j] <= tp1:
                        outcome = "TP1"; pnl_pct = (entry - tp1) / entry * 100; break
            else:
                last = C[min(idx + 20, len(df) - 1)]
                pnl_pct = ((last - entry) / entry if d == "BUY" else (entry - last) / entry) * 100
            s.pnl += pnl_pct
            if outcome == "TP1":
                s.wins += 1; s.win_pnls.append(pnl_pct)
            elif outcome == "SL":
                s.losses += 1; s.loss_pnls.append(pnl_pct)
            else:
                s.opens += 1
            tc += 1
    print(f" -> {tc} trades")


def main():
    print("=" * 70)
    print("  NOVAFX BACKTEST — commit fc1986c")
    print("  90d | 1H | Yahoo Finance | Pure numpy")
    print("=" * 70)
    all_s = {}
    for ac, syms in SYMBOLS.items():
        print(f"\n[{ac.upper()}]")
        st = S()
        for sym, tick in syms.items():
            print(f"  {sym} ({tick})", end=" ", flush=True)
            run(sym, tick, ac, st)
            time.sleep(0.3)
        all_s[ac] = st

    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    hdr = f"{'Asset':<14} {'Raw':>5} {'Emit':>5} {'Win':>4} {'Loss':>4} {'WR%':>6} {'RR':>6} {'PnL%':>8} {'ATR%':>5}"
    print(hdr); print("-" * 70)
    tr = te = tw = tl = 0; tp = 0.0
    for ac, st in all_s.items():
        tot = st.wins + st.losses + st.opens
        wr = st.wins / tot * 100 if tot else 0
        aw = np.mean(st.win_pnls) if st.win_pnls else 0
        al = abs(np.mean(st.loss_pnls)) if st.loss_pnls else 1
        rr = aw / al if al > 0 else 0
        ap = st.atr_n / st.emitted * 100 if st.emitted else 0
        print(f"{ac:<14} {st.raw:>5} {st.emitted:>5} {st.wins:>4} {st.losses:>4} {wr:>5.1f}% {f'1:{rr:.1f}':>6} {st.pnl:>+7.2f}% {ap:>4.0f}%")
        tr += st.raw; te += st.emitted; tw += st.wins; tl += st.losses; tp += st.pnl
    print("-" * 70)
    gw = tw / (tw + tl) * 100 if tw + tl else 0
    gf = (1 - te / tr) * 100 if tr else 0
    print(f"{'TOTAL':<14} {tr:>5} {te:>5} {tw:>4} {tl:>4} {gw:>5.1f}% {'—':>6} {tp:>+7.2f}% {'—':>5}")

    print(f"\n{'=' * 70}\n  FILTER BREAKDOWN\n{'=' * 70}")
    print(f"{'Asset':<14} {'Regime':>7} {'Choppy':>7} {'Volume':>7} {'NoCnfl':>7} {'NaN':>5}")
    print("-" * 70)
    for ac, st in all_s.items():
        print(f"{ac:<14} {st.regime_f:>7} {st.choppy_f:>7} {st.vol_f:>7} {st.no_conf:>7} {st.nans:>5}")

    print(f"\n{'=' * 70}\n  REGIME DISTRIBUTION\n{'=' * 70}")
    for ac, st in all_s.items():
        t = sum(st.regimes.values())
        if t:
            p = {k: f"{v/t*100:.0f}%" for k, v in st.regimes.items()}
            print(f"  {ac:<14} {p}")

    print(f"\n{'=' * 70}\n  ATR SL DISTANCE (% of price)\n{'=' * 70}")
    for ac, st in all_s.items():
        if st.atr_sls:
            print(f"  {ac:<14} min={min(st.atr_sls):.3f}%  avg={np.mean(st.atr_sls):.3f}%  max={max(st.atr_sls):.3f}%")
        else:
            print(f"  {ac:<14} [no ATR data]")

    print(f"\n{'=' * 70}\n  STRATEGY HITS\n{'=' * 70}")
    for ac, st in all_s.items():
        print(f"  {ac}: {st.strats}")

    print(f"\n{'=' * 70}\n  FLAGS\n{'=' * 70}")
    flags = []
    if tr and gf > 95:
        flags.append(f"[WARN] Filter rate {gf:.0f}% — may be too aggressive")
    if tw + tl and gw < 30:
        flags.append(f"[WARN] Win rate {gw:.0f}% — below 30%")
    for ac, st in all_s.items():
        if st.nans:
            flags.append(f"[BUG] {ac}: {st.nans} NaN exceptions")
        if st.raw > 20 and st.emitted == 0:
            flags.append(f"[WARN] {ac}: {st.raw} raw but 0 emitted")
        if st.atr_sls and np.mean(st.atr_sls) > 5:
            flags.append(f"[WARN] {ac}: ATR SL avg {np.mean(st.atr_sls):.1f}% too wide")
    if not flags:
        flags.append("[OK] No critical issues")
    for f in flags:
        print(f"  {f}")
    print()


if __name__ == "__main__":
    main()

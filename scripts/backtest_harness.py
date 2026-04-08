"""
NovaFX Backtest Harness — commit fc1986c
Built with: httpx, numpy, pandas only. No ta, no yfinance.
"""

import httpx
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════════
# SECTION 1 — Data Fetcher
# ═══════════════════════════════════════════════════════════════════════


def fetch_ohlcv(symbol: str, interval: str = "1h", range_: str = "90d") -> pd.DataFrame:
    """Fetch OHLCV from Yahoo Finance chart API.

    Returns DataFrame with columns: timestamp, open, high, low, close, volume
    sorted ascending by timestamp.
    """
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": interval, "range": range_, "includePrePost": "false"}
    headers = {"User-Agent": "Mozilla/5.0 NovaFX-Backtest/1.0"}

    resp = httpx.get(url, params=params, headers=headers, timeout=15, follow_redirects=True)
    if resp.status_code != 200:
        raise ValueError(f"Yahoo API returned {resp.status_code} for {symbol}")

    data = resp.json()
    chart = data.get("chart", {})
    error = chart.get("error")
    if error:
        raise ValueError(f"Yahoo API error for {symbol}: {error}")

    results = chart.get("result", [])
    if not results:
        raise ValueError(f"No data returned for {symbol}")

    result = results[0]
    timestamps = result.get("timestamp")
    if not timestamps:
        raise ValueError(f"No timestamps for {symbol}")

    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
        "open": quote.get("open"),
        "high": quote.get("high"),
        "low": quote.get("low"),
        "close": quote.get("close"),
        "volume": quote.get("volume"),
    })

    # Drop rows where OHLC are all NaN
    df = df.dropna(subset=["open", "high", "low", "close"], how="all")

    # Forward-fill close, zero-fill volume
    df["close"] = df["close"].ffill()
    df["open"] = df["open"].ffill()
    df["high"] = df["high"].ffill()
    df["low"] = df["low"].ffill()
    df["volume"] = df["volume"].fillna(0)

    # Drop any remaining NaN rows
    df = df.dropna(subset=["close"])

    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
# SECTION 2 — Pure numpy/pandas indicators
# ═══════════════════════════════════════════════════════════════════════


def calc_ema(close: pd.Series, period: int) -> pd.Series:
    """EMA using pandas ewm."""
    return close.ewm(span=period, adjust=False).mean()


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR with EMA smoothing (Wilder)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI with Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Returns (macd_line, signal_line, histogram)."""
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calc_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calc_bollinger(close: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Returns (upper, middle, lower, bandwidth)."""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + std_dev * std
    lower = middle - std_dev * std
    bandwidth = (upper - lower) / middle
    return upper, middle, lower, bandwidth


def calc_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ADX indicator — returns ADX series."""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)

    # Zero out when the other is larger
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    atr = calc_atr(high, low, close, period)

    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx


def calc_hurst(close: pd.Series, lags: int = 20) -> float:
    """Hurst exponent via variance of log returns at different lags."""
    if len(close) < lags * 2:
        return 0.5
    log_returns = np.log(close / close.shift(1)).dropna()
    tau = []
    for lag in range(2, lags):
        diffs = log_returns.diff(lag).dropna()
        if len(diffs) < 2:
            continue
        s = diffs.std()
        if s > 0:
            tau.append(s)
    if len(tau) < 2:
        return 0.5
    log_lags = np.log(np.arange(2, 2 + len(tau)))
    log_tau = np.log(tau)
    try:
        slope = np.polyfit(log_lags, log_tau, 1)[0]
        return float(slope)
    except Exception:
        return 0.5


# ═══════════════════════════════════════════════════════════════════════
# SECTION 3 — Strategies (replicating signal_engine.py from fc1986c)
# ═══════════════════════════════════════════════════════════════════════


def strat_ema_cross(df: pd.DataFrame) -> str | None:
    """EMA 9/21 cross with slope confirmation."""
    if len(df) < 30:
        return None
    e9 = calc_ema(df["close"], 9)
    e21 = calc_ema(df["close"], 21)
    pf, cf = e9.iloc[-2], e9.iloc[-1]
    ps, cs = e21.iloc[-2], e21.iloc[-1]
    if any(np.isnan(v) for v in [pf, cf, ps, cs]):
        return None
    slope = cs - e21.iloc[-3] if len(e21) >= 3 and not np.isnan(e21.iloc[-3]) else 0
    if pf <= ps and cf > cs:
        return "BUY" if slope >= 0 else None
    if pf >= ps and cf < cs:
        return "SELL" if slope <= 0 else None
    return None


def strat_rsi_adaptive(df: pd.DataFrame) -> str | None:
    """RSI with adaptive thresholds based on SMA50 trend context."""
    if len(df) < 50:
        return None
    r = calc_rsi(df["close"])
    val = r.iloc[-1]
    if np.isnan(val):
        return None
    sma50 = df["close"].rolling(50).mean().iloc[-1]
    price = df["close"].iloc[-1]
    if np.isnan(sma50):
        bt, st = 30, 70
    elif price > sma50:
        bt, st = 40, 80
    else:
        bt, st = 20, 60
    if val < bt:
        return "BUY"
    if val > st:
        return "SELL"
    return None


def strat_macd_zero(df: pd.DataFrame) -> str | None:
    """MACD cross with zero-line filter."""
    if len(df) < 50:
        return None
    ml, sl, _ = calc_macd(df["close"])
    pm, cm = ml.iloc[-2], ml.iloc[-1]
    ps, cs = sl.iloc[-2], sl.iloc[-1]
    if any(np.isnan(v) for v in [pm, cm, ps, cs]):
        return None
    if pm <= ps and cm > cs:
        return "BUY" if cm >= 0 else None
    if pm >= ps and cm < cs:
        return "SELL" if cm <= 0 else None
    return None


def strat_bb_reversion(df: pd.DataFrame) -> str | None:
    """Bollinger Band reversion with trend gate."""
    if len(df) < 30:
        return None
    u, m, l, bw = calc_bollinger(df["close"])
    if any(np.isnan(v) for v in [u.iloc[-1], l.iloc[-1], m.iloc[-1]]):
        return None
    # Reject expanding bands
    if len(df) >= 25 and not np.isnan(bw.iloc[-5]):
        if bw.iloc[-1] > bw.iloc[-5] * 1.3:
            return None
        slope = abs(m.iloc[-1] - m.iloc[-5]) / m.iloc[-1] if m.iloc[-1] > 0 else 0
        if slope > 0.005:
            return None
    if df["close"].iloc[-1] <= l.iloc[-1]:
        return "BUY"
    if df["close"].iloc[-1] >= u.iloc[-1]:
        return "SELL"
    return None


def strat_rsi_divergence(df: pd.DataFrame) -> str | None:
    """RSI divergence — price makes new extreme, RSI doesn't confirm.
    Includes SMA50 directional filter: only BUY below SMA50, only SELL above."""
    if len(df) < 50:
        return None
    r = calc_rsi(df["close"])
    c = df["close"].values
    rv = r.values
    if np.any(np.isnan(rv[-30:])):
        return None

    # SMA50 trend gate for divergence direction
    sma50 = df["close"].rolling(50).mean().iloc[-1]
    price = c[-1]
    if np.isnan(sma50):
        return None

    w = c[-30:]
    rw = rv[-30:]
    lows, highs = [], []
    for i in range(2, len(w) - 2):
        if w[i] < w[i-1] and w[i] < w[i-2] and w[i] < w[i+1] and w[i] < w[i+2]:
            lows.append(i)
        if w[i] > w[i-1] and w[i] > w[i-2] and w[i] > w[i+1] and w[i] > w[i+2]:
            highs.append(i)
    # Bullish divergence only when price is below SMA50 (oversold context)
    if len(lows) >= 2 and price < sma50:
        i1, i2 = lows[-2], lows[-1]
        if w[i2] < w[i1] and rw[i2] > rw[i1] and rw[i2] < 40:
            return "BUY"
    # Bearish divergence only when price is above SMA50 (overbought context)
    if len(highs) >= 2 and price > sma50:
        i1, i2 = highs[-2], highs[-1]
        if w[i2] > w[i1] and rw[i2] < rw[i1] and rw[i2] > 60:
            return "SELL"
    return None


def strat_momentum_breakout(df: pd.DataFrame) -> str | None:
    """Momentum breakout: price breaks 20-bar high/low with trend + RSI confirmation."""
    if len(df) < 50:
        return None
    close = df["close"]
    high = df["high"]
    low = df["low"]
    price = close.iloc[-1]

    # 20-bar channel (excluding current bar)
    high_20 = high.iloc[-21:-1].max()
    low_20 = low.iloc[-21:-1].min()
    if np.isnan(high_20) or np.isnan(low_20):
        return None

    # Trend filter: EMA20 vs EMA50
    ema20 = calc_ema(close, 20).iloc[-1]
    ema50 = calc_ema(close, 50).iloc[-1]
    if np.isnan(ema20) or np.isnan(ema50):
        return None

    # RSI filter: avoid overbought breakout buys / oversold breakout sells
    rsi = calc_rsi(close).iloc[-1]
    if np.isnan(rsi):
        return None

    # BUY: close above 20-bar high, uptrend, RSI not extreme overbought
    if price > high_20 and ema20 > ema50 and rsi < 80:
        return "BUY"
    # SELL: close below 20-bar low, downtrend, RSI not extreme oversold
    if price < low_20 and ema20 < ema50 and rsi > 20:
        return "SELL"
    return None


def strat_donchian_breakout(df: pd.DataFrame) -> str | None:
    """Donchian channel breakout: close above/below 20-bar high/low."""
    if len(df) < 25:
        return None
    close = df["close"]
    high = df["high"]
    low = df["low"]
    price = close.iloc[-1]

    # Entry channel: 20-bar (excluding current bar)
    entry_high = high.iloc[-21:-1].max()
    entry_low = low.iloc[-21:-1].min()
    if np.isnan(entry_high) or np.isnan(entry_low):
        return None

    # Require channel width > 0.2% to avoid noise breakouts
    channel_width = (entry_high - entry_low) / entry_low if entry_low > 0 else 0
    if channel_width < 0.002:
        return None

    # BUY on upper breakout, SELL on lower breakout
    if price > entry_high:
        return "BUY"
    if price < entry_low:
        return "SELL"
    return None


def strat_macd_trend(df: pd.DataFrame) -> str | None:
    """MACD crossover with SMA50 trend filter (no zero-line requirement)."""
    if len(df) < 50:
        return None
    ml, sl, _ = calc_macd(df["close"])
    pm, cm = ml.iloc[-2], ml.iloc[-1]
    ps, cs = sl.iloc[-2], sl.iloc[-1]
    if any(np.isnan(v) for v in [pm, cm, ps, cs]):
        return None

    sma50 = df["close"].rolling(50).mean().iloc[-1]
    price = df["close"].iloc[-1]
    if np.isnan(sma50):
        return None

    # BUY: MACD crosses above signal AND price > SMA50
    if pm <= ps and cm > cs and price > sma50:
        return "BUY"
    # SELL: MACD crosses below signal AND price < SMA50
    if pm >= ps and cm < cs and price < sma50:
        return "SELL"
    return None


ALL_STRATEGIES = [
    ("ema_cross", strat_ema_cross),
    ("rsi_adaptive", strat_rsi_adaptive),
    ("macd_zero", strat_macd_zero),
    ("bb_reversion", strat_bb_reversion),
    ("rsi_divergence", strat_rsi_divergence),
    ("momentum_breakout", strat_momentum_breakout),
    ("donchian_breakout", strat_donchian_breakout),
    ("macd_trend", strat_macd_trend),
]


def run_strategy_class(name: str, data: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    """Run a strategy using the src/strategies registry (class-based).

    Bridge between the inline backtest harness and the modular strategy classes.
    Returns the signal DataFrame from BaseStrategy.generate_signals().
    """
    from src.strategies import get_strategy
    strategy = get_strategy(name, params=params)
    return strategy.generate_signals(data)


# ═══════════════════════════════════════════════════════════════════════
# SECTION 4 — Filters (regime, chop, volume, confluence)
# ═══════════════════════════════════════════════════════════════════════

REGIME_ALLOWED = {
    "trending": {"ema_cross", "macd_zero", "rsi_adaptive", "momentum_breakout", "donchian_breakout", "macd_trend"},
    "mean_reverting": {"rsi_adaptive", "bb_reversion", "rsi_divergence", "ema_cross", "macd_zero"},
    "ranging": {"rsi_adaptive", "bb_reversion", "rsi_divergence"},
}


def detect_regime(df: pd.DataFrame) -> str:
    """ADX-based regime detection with loosened thresholds."""
    if len(df) < 50:
        return "mean_reverting"
    adx_val = calc_adx(df["high"], df["low"], df["close"]).iloc[-1]
    if np.isnan(adx_val):
        return "mean_reverting"
    if adx_val > 15:
        return "trending"
    if adx_val < 12:
        return "mean_reverting"
    # ADX 12-15: use Hurst as tiebreaker
    h = calc_hurst(df["close"])
    if h > 0.55:
        return "trending"
    return "mean_reverting"


def is_choppy(df: pd.DataFrame) -> bool:
    """Composite chop detector: ADX + ATR percentile + BB width."""
    if len(df) < 50:
        return False
    count = 0
    # ADX < 20
    adx_val = calc_adx(df["high"], df["low"], df["close"]).iloc[-1]
    if not np.isnan(adx_val) and adx_val < 20:
        count += 1
    # ATR percentile
    atr_series = calc_atr(df["high"], df["low"], df["close"])
    atr_clean = atr_series.dropna()
    if len(atr_clean) > 14:
        pct = (atr_clean < atr_clean.iloc[-1]).sum() / len(atr_clean) * 100
        if pct < 30:
            count += 1
    # BB width percentile
    _, _, _, bw = calc_bollinger(df["close"])
    bw_clean = bw.dropna()
    if len(bw_clean) > 5:
        bw_pct = (bw_clean < bw_clean.iloc[-1]).sum() / len(bw_clean)
        if bw_pct < 0.25:
            count += 1
    return count >= 2


def check_volume(df: pd.DataFrame, lookback: int = 20, asset_class: str = "forex") -> bool:
    """Volume filter with per-asset thresholds."""
    if "volume" not in df.columns or len(df) < lookback + 1:
        return True
    vols = df["volume"].tail(lookback + 1)
    avg = vols.iloc[:-1].mean()
    if avg == 0:
        return True
    thresholds = {"forex": 1.2, "stock": 1.0, "crypto": 1.1, "commodities": 1.2}
    mult = thresholds.get(asset_class, 1.0)
    return vols.iloc[-1] >= avg * mult


# ═══════════════════════════════════════════════════════════════════════
# SECTION 5 — Backtest Runner
# ═══════════════════════════════════════════════════════════════════════

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
        "BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD",
        "SOLUSDT": "SOL-USD", "XRPUSDT": "XRP-USD",
    },
    "commodities": {
        "XAUUSD": "GC=F", "XAGUSD": "SI=F",
    },
}

# ATR multipliers — adjusted for slippage compensation (~16% R:R compression)
# Crypto: wider TP to offset execution costs on lower-liquidity alts
# Forex: tighter spreads need less compensation
ATR_MULT = {
    "forex": {"sl": 1.5, "tp": 3.0},
    "crypto": {"sl": 1.5, "tp": 3.5},
    "stocks": {"sl": 2.0, "tp": 3.2},
    "commodities": {"sl": 2.0, "tp": 3.2},
}

PCT_FALLBACK = {
    "forex": {"sl": 0.003, "tp": 0.006},
    "crypto": {"sl": 0.015, "tp": 0.03},
    "stocks": {"sl": 0.01, "tp": 0.02},
    "commodities": {"sl": 0.008, "tp": 0.016},
}


@dataclass
class AssetStats:
    raw: int = 0
    regime_f: int = 0
    choppy_f: int = 0
    volume_f: int = 0
    no_conf: int = 0
    emitted: int = 0
    nans: int = 0
    wins: int = 0
    losses: int = 0
    opens: int = 0
    pnl: float = 0.0
    win_pnls: list = field(default_factory=list)
    loss_pnls: list = field(default_factory=list)
    atr_sls: list = field(default_factory=list)
    regimes: dict = field(default_factory=lambda: {"trending": 0, "mean_reverting": 0, "ranging": 0})
    strats: dict = field(default_factory=dict)
    atr_n: int = 0


def simulate_trade(highs, lows, closes, idx, direction, sl, tp1, bars=20):
    """Simulate trade from idx forward. Returns (outcome, pnl_pct)."""
    entry = closes[idx]
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


def backtest_symbol(nova_sym: str, yf_ticker: str, asset_class: str, stats: AssetStats):
    """Run full signal pipeline on one symbol."""
    df = fetch_ohlcv(yf_ticker, "1h", "90d")
    if df is None or len(df) < 0:
        print("    [no data]")
        return

    print(f"    {len(df)} bars", end="", flush=True)
    W = 250
    if len(df) < W + 20:
        print(" [insufficient]")
        return

    H = df["high"].values.astype(float)
    L = df["low"].values.astype(float)
    C = df["close"].values.astype(float)
    tc = 0
    last_signal_bar = -999  # Cooldown: min 16 bars (~16h) between signals

    for start in range(0, len(df) - W - 20, 4):
        # Signal cooldown check
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
                    stats.strats[name] = stats.strats.get(name, 0) + 1
            except Exception:
                stats.nans += 1

        if not hits:
            continue
        stats.raw += len(hits)

        # Regime filter
        regime = detect_regime(window)
        stats.regimes[regime] = stats.regimes.get(regime, 0) + 1
        allowed = REGIME_ALLOWED.get(regime, set())
        filtered = [(n, s) for n, s in hits if n in allowed]
        stats.regime_f += len(hits) - len(filtered)
        if not filtered:
            continue

        # Chop filter
        if is_choppy(window):
            stats.choppy_f += len(filtered)
            continue

        # Volume filter
        if not check_volume(window, asset_class=asset_class):
            stats.volume_f += len(filtered)
            continue

        # Confluence
        buys = [x for x in filtered if x[1] == "BUY"]
        sells = [x for x in filtered if x[1] == "SELL"]
        if len(buys) >= 1:
            direction = "BUY"
        elif len(sells) >= 1:
            direction = "SELL"
        else:
            stats.no_conf += len(filtered)
            continue

        # Trend alignment gate: reject BUY if EMA20 < EMA50, reject SELL if EMA20 > EMA50
        ema20 = calc_ema(window["close"], 20).iloc[-1]
        ema50 = calc_ema(window["close"], 50).iloc[-1] if len(window) >= 50 else ema20
        if not (np.isnan(ema20) or np.isnan(ema50)):
            if direction == "BUY" and ema20 < ema50:
                stats.no_conf += len(filtered)
                continue
            if direction == "SELL" and ema20 > ema50:
                stats.no_conf += len(filtered)
                continue

        # Compute SL/TP
        price = C[start + W - 1]
        atr_val = calc_atr(window["high"], window["low"], window["close"]).iloc[-1]
        atr_ok = not np.isnan(atr_val) and atr_val > 0

        mult = ATR_MULT.get(asset_class, ATR_MULT["forex"])
        fb = PCT_FALLBACK.get(asset_class, PCT_FALLBACK["forex"])

        if atr_ok:
            sl_d = atr_val * mult["sl"]
            tp_d = atr_val * mult["tp"]
            stats.atr_sls.append(sl_d / price * 100)
            stats.atr_n += 1
        else:
            sl_d = price * fb["sl"]
            tp_d = price * fb["tp"]

        sl = price - sl_d if direction == "BUY" else price + sl_d
        tp1 = price + tp_d if direction == "BUY" else price - tp_d

        stats.emitted += 1
        last_signal_bar = current_bar

        # Simulate trade
        idx = start + W - 1
        if idx + 20 < len(df):
            outcome, pnl = simulate_trade(H, L, C, idx, direction, sl, tp1)
            stats.pnl += pnl
            if outcome == "TP1":
                stats.wins += 1
                stats.win_pnls.append(pnl)
            elif outcome == "SL":
                stats.losses += 1
                stats.loss_pnls.append(pnl)
            else:
                stats.opens += 1
            tc += 1

    print(f" -> {tc} trades")


def print_report(all_stats: dict[str, AssetStats]):
    """Print the full backtest report."""
    print(f"\n{'='*70}")
    print("  NOVAFX BACKTEST — commit fc1986c")
    print("  90d | 1H | Yahoo Finance API | Pure numpy/pandas")
    print(f"{'='*70}")

    hdr = f"{'Asset':<14} {'Raw':>5} {'Emit':>5} {'Win':>4} {'Loss':>4} {'WR%':>6} {'RR':>6} {'PnL%':>8} {'ATR%':>5}"
    print(f"\n{hdr}")
    print("-" * 70)

    tr = te = tw = tl = 0
    tp = 0.0

    for ac, s in all_stats.items():
        tot = s.wins + s.losses + s.opens
        wr = s.wins / tot * 100 if tot else 0
        aw = np.mean(s.win_pnls) if s.win_pnls else 0
        al = abs(np.mean(s.loss_pnls)) if s.loss_pnls else 1
        rr = aw / al if al > 0 else 0
        ap = s.atr_n / s.emitted * 100 if s.emitted else 0
        print(f"{ac:<14} {s.raw:>5} {s.emitted:>5} {s.wins:>4} {s.losses:>4} "
              f"{wr:>5.1f}% {f'1:{rr:.1f}':>6} {s.pnl:>+7.2f}% {ap:>4.0f}%")
        tr += s.raw; te += s.emitted; tw += s.wins; tl += s.losses; tp += s.pnl

    print("-" * 70)
    gw = tw / (tw + tl) * 100 if tw + tl else 0
    gf = (1 - te / tr) * 100 if tr else 0
    print(f"{'TOTAL':<14} {tr:>5} {te:>5} {tw:>4} {tl:>4} "
          f"{gw:>5.1f}% {'—':>6} {tp:>+7.2f}% {'—':>5}")

    # Filter breakdown
    print(f"\n{'='*70}\n  FILTER BREAKDOWN\n{'='*70}")
    print(f"{'Asset':<14} {'Regime':>7} {'Choppy':>7} {'Volume':>7} {'NoCnfl':>7} {'NaN':>5}")
    print("-" * 70)
    for ac, s in all_stats.items():
        print(f"{ac:<14} {s.regime_f:>7} {s.choppy_f:>7} {s.volume_f:>7} {s.no_conf:>7} {s.nans:>5}")

    # Regime distribution
    print(f"\n{'='*70}\n  REGIME DISTRIBUTION\n{'='*70}")
    for ac, s in all_stats.items():
        t = sum(s.regimes.values())
        if t:
            p = {k: f"{v/t*100:.0f}%" for k, v in s.regimes.items()}
            print(f"  {ac:<14} {p}")

    # ATR analysis
    print(f"\n{'='*70}\n  ATR SL DISTANCE (% of price)\n{'='*70}")
    for ac, s in all_stats.items():
        if s.atr_sls:
            print(f"  {ac:<14} min={min(s.atr_sls):.3f}%  avg={np.mean(s.atr_sls):.3f}%  max={max(s.atr_sls):.3f}%")
        else:
            print(f"  {ac:<14} [no ATR data]")

    # Strategy hits
    print(f"\n{'='*70}\n  STRATEGY HITS\n{'='*70}")
    for ac, s in all_stats.items():
        print(f"  {ac}: {s.strats}")

    # Flags
    print(f"\n{'='*70}\n  FLAGS\n{'='*70}")
    flags = []
    if tr and gf > 95:
        flags.append(f"[WARN] Filter rate {gf:.0f}% — may be too aggressive")
    if tw + tl and gw < 30:
        flags.append(f"[WARN] Win rate {gw:.0f}% — below 30%")
    for ac, s in all_stats.items():
        if s.nans:
            flags.append(f"[BUG] {ac}: {s.nans} NaN exceptions")
        if s.raw > 20 and s.emitted == 0:
            flags.append(f"[WARN] {ac}: {s.raw} raw hits but 0 emitted — filters too tight")
        if s.atr_sls and np.mean(s.atr_sls) > 5:
            flags.append(f"[WARN] {ac}: ATR SL avg {np.mean(s.atr_sls):.1f}% — stops too wide")
    if not flags:
        flags.append("[OK] No critical issues")
    for f in flags:
        print(f"  {f}")
    print()


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import sys
    import time

    # Quick mode: test 3 symbols only
    if "--quick" in sys.argv:
        SYMBOLS = {
            "forex": {"EURUSD": "EURUSD=X"},
            "stocks": {"AAPL": "AAPL"},
            "crypto": {"BTCUSDT": "BTC-USD"},
        }

    print("=" * 70)
    print("  NOVAFX BACKTEST HARNESS — commit fc1986c")
    print("  90d | 1H | Yahoo Finance API | Pure numpy/pandas")
    print("=" * 70)

    all_stats: dict[str, AssetStats] = {}

    for ac, syms in SYMBOLS.items():
        print(f"\n[{ac.upper()}]")
        stats = AssetStats()
        for sym, tick in syms.items():
            print(f"  {sym} ({tick})", end=" ", flush=True)
            try:
                backtest_symbol(sym, tick, ac, stats)
            except Exception as e:
                print(f"    [ERROR: {e}]")
            time.sleep(0.3)
        all_stats[ac] = stats

    print_report(all_stats)

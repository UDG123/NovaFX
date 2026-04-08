"""
NovaFX Backtest Harness — commit fc1986c
Built with: httpx, numpy, pandas only. No ta, no yfinance.
"""

import httpx
import numpy as np
import pandas as pd


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
# TEST SECTION 1 + 2
# ═══════════════════════════════════════════════════════════════════════


if __name__ == "__main__":
    import time

    test_symbols = [("EURUSD=X", "forex"), ("AAPL", "stock"), ("BTC-USD", "crypto")]

    for sym, cls in test_symbols:
        print(f"\n{'='*60}")
        print(f"  {sym} ({cls})")
        print(f"{'='*60}")
        try:
            df = fetch_ohlcv(sym, "1h", "90d")
            print(f"  Rows: {len(df)}")
            print(f"  Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

            # Test indicators
            c = df["close"]
            h = df["high"]
            l = df["low"]

            ema20 = calc_ema(c, 20)
            atr14 = calc_atr(h, l, c, 14)
            rsi14 = calc_rsi(c, 14)
            macd_l, sig_l, hist = calc_macd(c)
            bb_u, bb_m, bb_l, bb_bw = calc_bollinger(c)
            adx14 = calc_adx(h, l, c, 14)
            hurst = calc_hurst(c)

            print(f"\n  EMA(20) last 5:  {ema20.tail().values}")
            print(f"  ATR(14) last 5:  {atr14.tail().values}")
            print(f"  RSI(14) last 5:  {rsi14.tail().values}")
            print(f"  MACD last 5:     {macd_l.tail().values}")
            print(f"  BB upper last 5: {bb_u.tail().values}")
            print(f"  ADX(14) last 5:  {adx14.tail().values}")
            print(f"  Hurst:           {hurst:.4f}")

            # NaN check
            for name, series in [("EMA", ema20), ("ATR", atr14), ("RSI", rsi14),
                                  ("MACD", macd_l), ("BB_upper", bb_u), ("ADX", adx14)]:
                tail_nans = series.tail(100).isna().sum()
                if tail_nans > 0:
                    print(f"  [WARN] {name} has {tail_nans} NaN in last 100 bars")
                else:
                    print(f"  [OK]   {name} — no NaN in last 100 bars")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
        time.sleep(0.5)

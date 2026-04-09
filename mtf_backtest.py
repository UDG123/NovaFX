#!/usr/bin/env python3
"""
MTF Signal Logic Backtest
-------------------------
Tests the multi-timeframe RSI + MACD confluence strategy across trading desks.

Logic:
- BUY: 1h RSI < 38 + 4h RSI > 42 + MACD hist > 0
- SELL: 1h RSI > 62 + 4h RSI < 58 + MACD hist < 0
- SL: 1.5x ATR, TP: 3.0x ATR
"""

import asyncio
import httpx
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Optional
import time
import sys

# TwelveData API key
TWELVEDATA_API_KEY = "225f9dbd75eb4d77ac8448fffa7970f5"
BASE_URL = "https://api.twelvedata.com"

# Trading desks
DESKS = {
    "FOREX": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "USD/CHF", "NZD/USD", "EUR/GBP"],
    "CRYPTO": ["BTC/USD", "ETH/USD", "SOL/USD", "XRP/USD", "ADA/USD"],
    "STOCKS": ["AAPL", "TSLA", "NVDA", "MSFT", "META"],
}

# Rate limit handling
REQUEST_DELAY = 8.5  # TwelveData has 8 requests/minute on free tier


def calc_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI for entire series."""
    deltas = np.diff(closes, prepend=closes[0])
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    rsi = np.full(len(closes), 50.0)

    for i in range(period, len(closes)):
        avg_gain = np.mean(gains[i-period+1:i+1])
        avg_loss = np.mean(losses[i-period+1:i+1])
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


def calc_ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    k = 2 / (period + 1)
    ema = np.zeros(len(data))
    ema[0] = data[0]
    for i in range(1, len(data)):
        ema[i] = data[i] * k + ema[i-1] * (1 - k)
    return ema


def calc_macd_hist(closes: np.ndarray) -> np.ndarray:
    """Calculate MACD histogram."""
    if len(closes) < 26:
        return np.zeros(len(closes))

    ema12 = calc_ema(closes, 12)
    ema26 = calc_ema(closes, 26)
    macd_line = ema12 - ema26
    signal_line = calc_ema(macd_line, 9)
    return macd_line - signal_line


def calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR."""
    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - np.roll(closes, 1)),
            np.abs(lows - np.roll(closes, 1))
        )
    )
    tr[0] = highs[0] - lows[0]

    atr = np.zeros(len(closes))
    atr[:period] = np.mean(tr[:period])

    for i in range(period, len(closes)):
        atr[i] = np.mean(tr[i-period+1:i+1])

    return atr


async def fetch_ohlcv(client: httpx.AsyncClient, symbol: str, interval: str, outputsize: int = 2160) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data from TwelveData."""
    # TwelveData uses the symbol as-is (with slashes for forex/crypto)
    api_symbol = symbol

    url = f"{BASE_URL}/time_series"
    params = {
        "symbol": api_symbol,
        "interval": interval,
        "outputsize": outputsize,
        "apikey": TWELVEDATA_API_KEY,
    }

    try:
        resp = await client.get(url, params=params, timeout=30)
        data = resp.json()

        if "values" not in data:
            print(f"  [WARN] No data for {symbol} ({interval}): {data.get('message', 'Unknown error')}")
            return None

        df = pd.DataFrame(data["values"])
        df["datetime"] = pd.to_datetime(df["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].astype(float)

        return df

    except Exception as e:
        print(f"  [ERROR] Fetching {symbol}: {e}")
        return None


def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h data to 4h."""
    df = df_1h.copy()
    df = df.set_index("datetime")

    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
    }
    if "volume" in df.columns:
        agg_dict["volume"] = "sum"

    resampled = df.resample("4h").agg(agg_dict).dropna()

    return resampled.reset_index()


def generate_signals(df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
    """
    Generate MTF signals.
    BUY: 1h RSI < 38 + 4h RSI > 42 + MACD hist > 0
    SELL: 1h RSI > 62 + 4h RSI < 58 + MACD hist < 0
    """
    # Calculate indicators on 1h
    closes_1h = df_1h["close"].values
    highs_1h = df_1h["high"].values
    lows_1h = df_1h["low"].values

    rsi_1h = calc_rsi(closes_1h)
    macd_hist = calc_macd_hist(closes_1h)
    atr = calc_atr(highs_1h, lows_1h, closes_1h)

    # Calculate 4h RSI
    closes_4h = df_4h["close"].values
    rsi_4h_full = calc_rsi(closes_4h)

    # Map 4h RSI back to 1h timeframe
    df_1h = df_1h.copy()
    df_1h["rsi_1h"] = rsi_1h
    df_1h["macd_hist"] = macd_hist
    df_1h["atr"] = atr

    # Create 4h datetime mapping
    df_4h_map = df_4h.copy()
    df_4h_map["rsi_4h"] = rsi_4h_full
    df_4h_map = df_4h_map[["datetime", "rsi_4h"]]

    # Map 4h RSI to 1h bars
    df_1h["datetime_4h"] = df_1h["datetime"].dt.floor("4h")
    df_4h_map["datetime_4h"] = df_4h_map["datetime"].dt.floor("4h")
    df_4h_map = df_4h_map.drop_duplicates("datetime_4h", keep="last")

    df_1h = df_1h.merge(df_4h_map[["datetime_4h", "rsi_4h"]], on="datetime_4h", how="left")
    df_1h["rsi_4h"] = df_1h["rsi_4h"].ffill()

    # Generate signals
    df_1h["signal"] = 0

    # BUY conditions
    buy_mask = (df_1h["rsi_1h"] < 38) & (df_1h["rsi_4h"] > 42) & (df_1h["macd_hist"] > 0)
    df_1h.loc[buy_mask, "signal"] = 1

    # SELL conditions
    sell_mask = (df_1h["rsi_1h"] > 62) & (df_1h["rsi_4h"] < 58) & (df_1h["macd_hist"] < 0)
    df_1h.loc[sell_mask, "signal"] = -1

    return df_1h


def simulate_trades(df: pd.DataFrame, sl_mult: float = 1.5, tp_mult: float = 3.0) -> list:
    """
    Simulate trades with ATR-based SL/TP.
    Returns list of trade results.
    """
    trades = []
    in_trade = False
    trade_entry = None
    trade_dir = None
    trade_sl = None
    trade_tp = None
    entry_idx = None

    for i in range(50, len(df)):  # Skip warmup period
        row = df.iloc[i]

        if in_trade:
            # Check exit conditions
            if trade_dir == 1:  # Long
                if row["low"] <= trade_sl:
                    # Hit stop loss
                    pnl = trade_sl - trade_entry
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "BUY",
                        "entry": trade_entry,
                        "exit": trade_sl,
                        "pnl_pct": (pnl / trade_entry) * 100,
                        "result": "LOSS",
                    })
                    in_trade = False
                elif row["high"] >= trade_tp:
                    # Hit take profit
                    pnl = trade_tp - trade_entry
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "BUY",
                        "entry": trade_entry,
                        "exit": trade_tp,
                        "pnl_pct": (pnl / trade_entry) * 100,
                        "result": "WIN",
                    })
                    in_trade = False
            else:  # Short
                if row["high"] >= trade_sl:
                    # Hit stop loss
                    pnl = trade_entry - trade_sl
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "SELL",
                        "entry": trade_entry,
                        "exit": trade_sl,
                        "pnl_pct": (pnl / trade_entry) * 100,
                        "result": "LOSS",
                    })
                    in_trade = False
                elif row["low"] <= trade_tp:
                    # Hit take profit
                    pnl = trade_entry - trade_tp
                    trades.append({
                        "entry_idx": entry_idx,
                        "exit_idx": i,
                        "direction": "SELL",
                        "entry": trade_entry,
                        "exit": trade_tp,
                        "pnl_pct": (pnl / trade_entry) * 100,
                        "result": "WIN",
                    })
                    in_trade = False

        # Check for new signal if not in trade
        if not in_trade and row["signal"] != 0:
            in_trade = True
            trade_entry = row["close"]
            trade_dir = row["signal"]
            entry_idx = i
            atr = row["atr"]

            if trade_dir == 1:  # Long
                trade_sl = trade_entry - (atr * sl_mult)
                trade_tp = trade_entry + (atr * tp_mult)
            else:  # Short
                trade_sl = trade_entry + (atr * sl_mult)
                trade_tp = trade_entry - (atr * tp_mult)

    return trades


async def backtest_symbol(client: httpx.AsyncClient, symbol: str) -> Optional[dict]:
    """Run backtest for a single symbol."""
    print(f"  Fetching {symbol}...")

    # Fetch 1h data (90 days = 2160 hourly bars)
    df_1h = await fetch_ohlcv(client, symbol, "1h", 2160)

    if df_1h is None or len(df_1h) < 100:
        return None

    # Resample to 4h
    df_4h = resample_to_4h(df_1h)

    if len(df_4h) < 30:
        return None

    # Generate signals
    df_signals = generate_signals(df_1h, df_4h)

    # Simulate trades
    trades = simulate_trades(df_signals)

    if not trades:
        return {
            "symbol": symbol,
            "total_trades": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "total_pnl": 0,
        }

    # Calculate metrics
    wins = [t for t in trades if t["result"] == "WIN"]
    win_rate = len(wins) / len(trades) * 100
    avg_pnl = np.mean([t["pnl_pct"] for t in trades])
    total_pnl = sum([t["pnl_pct"] for t in trades])

    return {
        "symbol": symbol,
        "total_trades": len(trades),
        "win_rate": round(win_rate, 1),
        "avg_pnl": round(avg_pnl, 3),
        "total_pnl": round(total_pnl, 2),
    }


async def backtest_desk(desk_name: str, symbols: list) -> dict:
    """Backtest all symbols in a desk."""
    print(f"\n{'='*50}")
    print(f"BACKTESTING {desk_name} DESK")
    print(f"{'='*50}")

    results = []

    async with httpx.AsyncClient() as client:
        for symbol in symbols:
            result = await backtest_symbol(client, symbol)
            if result:
                results.append(result)
            # Rate limit delay
            await asyncio.sleep(REQUEST_DELAY)

    if not results:
        return {
            "desk": desk_name,
            "total_trades": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "best_pair": "N/A",
            "worst_pair": "N/A",
            "symbols": [],
        }

    # Aggregate metrics
    total_trades = sum(r["total_trades"] for r in results)

    # Weighted win rate
    if total_trades > 0:
        weighted_wins = sum(r["total_trades"] * r["win_rate"] / 100 for r in results)
        win_rate = weighted_wins / total_trades * 100
    else:
        win_rate = 0

    # Average P&L across symbols
    valid_results = [r for r in results if r["total_trades"] > 0]
    if valid_results:
        avg_pnl = np.mean([r["avg_pnl"] for r in valid_results])
        best = max(valid_results, key=lambda x: x["total_pnl"])
        worst = min(valid_results, key=lambda x: x["total_pnl"])
    else:
        avg_pnl = 0
        best = {"symbol": "N/A", "total_pnl": 0}
        worst = {"symbol": "N/A", "total_pnl": 0}

    return {
        "desk": desk_name,
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "avg_pnl": round(avg_pnl, 3),
        "best_pair": f"{best['symbol']} ({best['total_pnl']:+.2f}%)",
        "worst_pair": f"{worst['symbol']} ({worst['total_pnl']:+.2f}%)",
        "symbols": results,
    }


async def main():
    """Run full backtest across all desks."""
    print("="*60)
    print("MTF SIGNAL BACKTEST - 90 DAYS")
    print("Strategy: 1h RSI + 4h RSI + MACD Confluence")
    print("SL: 1.5x ATR | TP: 3.0x ATR")
    print("="*60)
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")

    all_results = []

    for desk_name, symbols in DESKS.items():
        result = await backtest_desk(desk_name, symbols)
        all_results.append(result)

    # Generate report
    report = []
    report.append("="*60)
    report.append("MTF SIGNAL BACKTEST RESULTS - 90 DAYS")
    report.append("Strategy: 1h RSI < 38 + 4h RSI > 42 + MACD hist > 0 = BUY")
    report.append("          1h RSI > 62 + 4h RSI < 58 + MACD hist < 0 = SELL")
    report.append("SL: 1.5x ATR | TP: 3.0x ATR (Risk:Reward = 1:2)")
    report.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
    report.append("="*60)

    for result in all_results:
        report.append("")
        report.append(f"--- {result['desk']} DESK ---")
        report.append(f"Total Trades:    {result['total_trades']}")
        report.append(f"Win Rate:        {result['win_rate']:.1f}%")
        report.append(f"Avg P&L/Trade:   {result['avg_pnl']:+.3f}%")
        report.append(f"Best Pair:       {result['best_pair']}")
        report.append(f"Worst Pair:      {result['worst_pair']}")
        report.append("")
        report.append("  Symbol Performance:")
        for sym in result["symbols"]:
            if sym["total_trades"] > 0:
                report.append(f"    {sym['symbol']:12} | Trades: {sym['total_trades']:3} | Win: {sym['win_rate']:5.1f}% | Avg: {sym['avg_pnl']:+.3f}%")

    report.append("")
    report.append("="*60)

    # Calculate overall stats
    total_trades = sum(r["total_trades"] for r in all_results)
    if total_trades > 0:
        weighted_wins = sum(r["total_trades"] * r["win_rate"] / 100 for r in all_results)
        overall_win = weighted_wins / total_trades * 100
        overall_avg = np.mean([r["avg_pnl"] for r in all_results if r["total_trades"] > 0])
    else:
        overall_win = 0
        overall_avg = 0

    report.append("OVERALL SUMMARY")
    report.append(f"Total Trades:    {total_trades}")
    report.append(f"Overall Win %:   {overall_win:.1f}%")
    report.append(f"Avg P&L/Trade:   {overall_avg:+.3f}%")
    report.append("="*60)

    report_text = "\n".join(report)

    # Print to console
    print("\n" + report_text)

    # Save to file
    output_path = "/home/user/NovaFX/backtest_results.txt"
    with open(output_path, "w") as f:
        f.write(report_text)

    print(f"\nResults saved to: {output_path}")

    return all_results


if __name__ == "__main__":
    asyncio.run(main())

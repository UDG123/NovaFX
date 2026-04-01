import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator

from backtester.app.models.backtest import PhaseResult, Trade

logger = logging.getLogger(__name__)

# Commission per trade (entry + exit combined)
COMMISSION_RATES = {
    "forex": 0.0002,
    "crypto": 0.001,
    "indices": 0.0005,
    "commodities": 0.0002,
}

MARKET_SYMBOLS = {
    "forex": [
        "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
        "NZDUSD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
    ],
    "crypto": ["BTCUSD", "ETHUSD", "BTCUSDT", "ETHUSDT", "SOLUSD", "SOLUSDT"],
    "indices": ["SPX500", "NAS100", "US30", "DE40", "UK100", "JP225"],
    "commodities": ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"],
}


def detect_market(symbol: str) -> str:
    symbol_upper = symbol.upper().replace("/", "").replace("-", "")
    for market, symbols in MARKET_SYMBOLS.items():
        if symbol_upper in symbols:
            return market
    if any(symbol_upper.endswith(c) for c in ("USD", "JPY", "GBP", "EUR", "CHF", "AUD", "CAD", "NZD")):
        if len(symbol_upper) == 6:
            return "forex"
    if any(t in symbol_upper for t in ("BTC", "ETH", "SOL", "BNB", "USDT", "USDC")):
        return "crypto"
    return "forex"


def get_commission_rate(symbol: str) -> float:
    market = detect_market(symbol)
    return COMMISSION_RATES.get(market, COMMISSION_RATES["forex"])


# ── Multi-Timeframe confirmation ──────────────────────────────────────────────


def get_htf_bias(df_htf: pd.DataFrame) -> Optional[str]:
    """Determine higher-timeframe trend bias using EMA 9/21 on 4h candles.

    Returns "BUY" if EMA9 > EMA21 (bullish), "SELL" if EMA9 < EMA21 (bearish),
    or None if insufficient data or EMAs are equal.
    """
    if df_htf is None or len(df_htf) < 21:
        return None

    ema9 = EMAIndicator(close=df_htf["close"], window=9).ema_indicator()
    ema21 = EMAIndicator(close=df_htf["close"], window=21).ema_indicator()

    fast = ema9.iloc[-1]
    slow = ema21.iloc[-1]

    if np.isnan(fast) or np.isnan(slow):
        return None

    if fast > slow:
        return "BUY"
    elif fast < slow:
        return "SELL"
    return None


def filter_signals_by_mtf(
    signals: list[dict], htf_bias: Optional[str],
) -> tuple[list[dict], int]:
    """Filter signals to only those agreeing with the higher-timeframe bias.

    Returns (filtered_signals, number_of_signals_removed).
    If htf_bias is None (no data / inconclusive), all signals pass through.
    """
    if htf_bias is None:
        return signals, 0

    kept = [s for s in signals if s["action"] == htf_bias]
    removed = len(signals) - len(kept)
    return kept, removed


# ── Backtest engine ───────────────────────────────────────────────────────────


def run_backtest(
    df: pd.DataFrame,
    signals: list[dict],
    symbol: str,
    strategy: str,
    phase: str = "backtest",
    df_htf: Optional[pd.DataFrame] = None,
) -> PhaseResult:
    """Run a backtest over OHLCV data with a list of entry/exit signals.

    Each signal dict has:
        action: "BUY" or "SELL"
        entry_idx: int  — row index for entry
        exit_idx: int   — row index for exit

    If df_htf (4h candles) is provided, signals are filtered by
    higher-timeframe EMA 9/21 trend alignment before execution.
    """
    # Multi-timeframe filter
    mtf_filtered = 0
    if df_htf is not None:
        htf_bias = get_htf_bias(df_htf)
        signals, mtf_filtered = filter_signals_by_mtf(signals, htf_bias)
        if mtf_filtered:
            logger.info(
                "MTF filter: %s bias on %s, removed %d counter-trend signals",
                htf_bias, symbol, mtf_filtered,
            )

    commission_rate = get_commission_rate(symbol)
    trades: list[Trade] = []
    equity_curve: list[float] = [0.0]

    for sig in signals:
        action = sig["action"]
        entry_idx = sig["entry_idx"]
        exit_idx = sig["exit_idx"]

        if entry_idx >= len(df) or exit_idx >= len(df):
            continue

        entry_price = float(df.iloc[entry_idx]["close"])
        exit_price = float(df.iloc[exit_idx]["close"])

        if action == "BUY":
            raw_pnl_pct = (exit_price - entry_price) / entry_price
        else:
            raw_pnl_pct = (entry_price - exit_price) / entry_price

        commission_pct = commission_rate
        net_pnl_pct = raw_pnl_pct - commission_pct

        entry_time = _row_time(df, entry_idx)
        exit_time = _row_time(df, exit_idx)

        trades.append(Trade(
            symbol=symbol,
            action=action,
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=entry_time,
            exit_time=exit_time,
            pnl_pct=round(raw_pnl_pct * 100, 4),
            commission_pct=round(commission_pct * 100, 4),
            net_pnl_pct=round(net_pnl_pct * 100, 4),
            result="WIN" if net_pnl_pct > 0 else "LOSS",
        ))

        equity_curve.append(equity_curve[-1] + net_pnl_pct)

    wins = sum(1 for t in trades if t.result == "WIN")
    losses = len(trades) - wins
    gross_pnl = sum(t.pnl_pct for t in trades)
    total_commission = sum(t.commission_pct for t in trades)
    net_pnl = sum(t.net_pnl_pct for t in trades)
    max_dd = _max_drawdown(equity_curve)

    return PhaseResult(
        phase=phase,
        symbol=symbol,
        strategy=strategy,
        total_trades=len(trades),
        wins=wins,
        losses=losses,
        win_rate=round(wins / len(trades) * 100, 2) if trades else 0.0,
        gross_pnl_pct=round(gross_pnl, 4),
        total_commission_pct=round(total_commission, 4),
        net_pnl_pct=round(net_pnl, 4),
        max_drawdown_pct=round(max_dd * 100, 4),
        mtf_filtered=mtf_filtered,
        trades=trades,
        started_at=trades[0].entry_time if trades else None,
        ended_at=trades[-1].exit_time if trades else None,
    )


def _row_time(df: pd.DataFrame, idx: int) -> datetime:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index[idx].to_pydatetime().replace(tzinfo=timezone.utc)
    if "timestamp" in df.columns:
        ts = df.iloc[idx]["timestamp"]
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime().replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = peak - val
        if dd > max_dd:
            max_dd = dd
    return max_dd

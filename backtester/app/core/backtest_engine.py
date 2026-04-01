import logging
from datetime import datetime, timezone

import pandas as pd

from backtester.app.models.backtest import PhaseResult, Trade

logger = logging.getLogger(__name__)

# Commission per trade (entry + exit combined)
# Forex: fixed 0.0002 (≈2 pips spread cost as a ratio of price)
# Crypto: 0.1% = 0.001
# Indices: 0.05% = 0.0005
# Commodities: falls back to forex rate
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


def run_backtest(
    df: pd.DataFrame,
    signals: list[dict],
    symbol: str,
    strategy: str,
    phase: str = "backtest",
) -> PhaseResult:
    """Run a backtest over OHLCV data with a list of entry/exit signals.

    Each signal dict has:
        action: "BUY" or "SELL"
        entry_idx: int  — row index for entry
        exit_idx: int   — row index for exit
    """
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

        # Commission expressed as % of entry price, deducted from pnl
        # For forex this is a fixed ratio (0.0002); for others it's a percentage
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

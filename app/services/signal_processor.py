import numpy as np
import pandas as pd

from app.config import settings
from app.models.signals import IncomingSignal, ProcessedSignal

# ATR multipliers per asset class
# ATR multipliers — adjusted for slippage compensation (~16% R:R compression)
ATR_CONFIG = {
    "forex": {"sl_mult": 1.5, "tp1_mult": 3.0, "tp2_mult": 4.5, "tp3_mult": 6.0},
    "crypto": {"sl_mult": 1.5, "tp1_mult": 3.5, "tp2_mult": 5.5, "tp3_mult": 8.0},
    "indices": {"sl_mult": 2.0, "tp1_mult": 3.2, "tp2_mult": 5.0, "tp3_mult": 7.0},
    "commodities": {"sl_mult": 2.0, "tp1_mult": 3.2, "tp2_mult": 5.0, "tp3_mult": 7.0},
    "stocks": {"sl_mult": 2.0, "tp1_mult": 3.2, "tp2_mult": 5.0, "tp3_mult": 7.0},
}

# Fallback percentages (only used if ATR unavailable)
FALLBACK_CONFIG = {
    "forex": {"sl_pct": 0.3, "tp_pct": 0.6},
    "crypto": {"sl_pct": 1.5, "tp_pct": 3.0},
    "indices": {"sl_pct": 0.5, "tp_pct": 1.0},
    "commodities": {"sl_pct": 0.8, "tp_pct": 1.6},
    "stocks": {"sl_pct": 1.0, "tp_pct": 2.0},
}

MARKET_CONFIG = {
    "forex": {
        "symbols": [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "NZDUSD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
        ],
    },
    "crypto": {
        "symbols": [
            "BTCUSD", "ETHUSD", "BTCUSDT", "ETHUSDT",
            "SOLUSD", "SOLUSDT", "BNBUSD", "BNBUSDT",
            "XRPUSD", "XRPUSDT",
        ],
    },
    "indices": {
        "symbols": ["SPX500", "NAS100", "US30", "DE40", "UK100", "JP225"],
    },
    "commodities": {
        "symbols": ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"],
    },
}


def detect_market(symbol: str) -> str:
    symbol_upper = symbol.upper().replace("/", "").replace("-", "")
    for market, cfg in MARKET_CONFIG.items():
        if symbol_upper in cfg["symbols"]:
            return market
    if any(symbol_upper.endswith(c) for c in ("USD", "JPY", "GBP", "EUR", "CHF", "AUD", "CAD", "NZD")):
        if len(symbol_upper) == 6:
            return "forex"
    if any(t in symbol_upper for t in ("BTC", "ETH", "SOL", "BNB", "USDT", "USDC", "XRP")):
        return "crypto"
    if symbol_upper in ("AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ", "GOOGL", "AMZN", "META", "AMD"):
        return "stocks"
    return "forex"


def compute_atr(df: pd.DataFrame, period: int = 14) -> float | None:
    """Compute ATR from OHLCV DataFrame."""
    if df is None or len(df) < period + 1:
        return None
    high = df["high"].values
    low = df["low"].values
    close = df["close"].values
    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
    )
    if len(tr) < period:
        return None
    atr = float(np.mean(tr[-period:]))
    return atr if not np.isnan(atr) else None


def process_signal(signal: IncomingSignal, df: pd.DataFrame = None) -> ProcessedSignal:
    """Process signal with ATR-based stops when data available, fallback to percentage."""
    market = detect_market(signal.symbol)
    atr = compute_atr(df) if df is not None else None

    if atr and atr > 0:
        cfg = ATR_CONFIG.get(market, ATR_CONFIG["forex"])
        sl_dist = atr * cfg["sl_mult"]
        tp1_dist = atr * cfg["tp1_mult"]
        tp2_dist = atr * cfg["tp2_mult"]
        tp3_dist = atr * cfg["tp3_mult"]

        if signal.action == "BUY":
            sl = signal.sl if signal.sl else round(signal.price - sl_dist, 6)
            tp1 = signal.tp if signal.tp else round(signal.price + tp1_dist, 6)
            tp2 = round(signal.price + tp2_dist, 6)
            tp3 = round(signal.price + tp3_dist, 6)
        else:
            sl = signal.sl if signal.sl else round(signal.price + sl_dist, 6)
            tp1 = signal.tp if signal.tp else round(signal.price - tp1_dist, 6)
            tp2 = round(signal.price - tp2_dist, 6)
            tp3 = round(signal.price - tp3_dist, 6)
    else:
        fb = FALLBACK_CONFIG.get(market, FALLBACK_CONFIG["forex"])
        sl_pct = fb["sl_pct"] / 100
        tp_pct = fb["tp_pct"] / 100
        if signal.action == "BUY":
            sl = signal.sl if signal.sl else round(signal.price * (1 - sl_pct), 6)
            tp1 = signal.tp if signal.tp else round(signal.price * (1 + tp_pct), 6)
            tp2 = round(signal.price * (1 + tp_pct * 2), 6)
            tp3 = round(signal.price * (1 + tp_pct * 3), 6)
        else:
            sl = signal.sl if signal.sl else round(signal.price * (1 + sl_pct), 6)
            tp1 = signal.tp if signal.tp else round(signal.price * (1 - tp_pct), 6)
            tp2 = round(signal.price * (1 - tp_pct * 2), 6)
            tp3 = round(signal.price * (1 - tp_pct * 3), 6)

    risk_per_unit = abs(signal.price - sl)
    reward_per_unit = abs(tp1 - signal.price)
    risk_reward = round(reward_per_unit / risk_per_unit, 2) if risk_per_unit > 0 else 0.0

    risk_amount = round(settings.ACCOUNT_BALANCE * (settings.DEFAULT_RISK_PCT / 100), 2)
    position_size = round(risk_amount / risk_per_unit, 4) if risk_per_unit > 0 else 0.0

    return ProcessedSignal(
        symbol=signal.symbol.upper(),
        action=signal.action,
        entry_price=signal.price,
        stop_loss=sl,
        take_profit_1=tp1,
        take_profit_2=tp2,
        take_profit_3=tp3,
        risk_reward=risk_reward,
        position_size=position_size,
        risk_amount=risk_amount,
        timeframe=signal.timeframe,
        source=signal.source,
        indicator=signal.indicator,
    )

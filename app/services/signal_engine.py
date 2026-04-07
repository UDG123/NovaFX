import logging
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, EMAIndicator, MACD
from ta.volatility import BollingerBands

from app.data.fetcher import fetch_ohlcv
from app.models.signals import IncomingSignal
from app.services.htf_bias import get_htf_bias
from app.services.regime import detect_regime, filter_signals_by_regime, is_choppy

logger = logging.getLogger(__name__)

WATCHLIST = [
    # Forex majors
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    # Forex crosses
    "EURGBP", "EURJPY", "GBPJPY",
    # Crypto
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    # Stocks
    "AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ",
    # Commodities
    "XAUUSD", "XAGUSD",
    # Indices
    "SPX500",
]

FOREX_SYMBOLS_SET = {
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY",
}

CORRELATION_GROUPS = {
    "USD_LONGS": {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"},
    "USD_SHORTS": {"USDJPY", "USDCAD", "USDCHF"},
    "RISK_ON": {"BTCUSDT", "ETHUSDT", "SOLUSDT", "SPY", "QQQ", "NVDA"},
    "SAFE_HAVEN": {"XAUUSD", "USDJPY", "USDCHF"},
}


# ---------------------------------------------------------------------------
# Session filters
# ---------------------------------------------------------------------------


def _is_forex_active_session() -> bool:
    """Check if current time is during London or NY session."""
    hour = datetime.now(timezone.utc).hour
    return 7 <= hour <= 21


def _get_correlation_group(symbol: str) -> str | None:
    s = symbol.upper().replace("/", "")
    for group_name, members in CORRELATION_GROUPS.items():
        if s in members:
            return group_name
    return None


# ---------------------------------------------------------------------------
# Volume filter
# ---------------------------------------------------------------------------


def check_volume_confirmation(df: pd.DataFrame, lookback: int = 20) -> bool:
    """Check if current volume is at least average (relative volume >= 1.0)."""
    if "volume" not in df.columns or len(df) < lookback + 1:
        return True
    volumes = df["volume"].tail(lookback + 1)
    if volumes.iloc[:-1].mean() == 0:
        return True
    avg_vol = volumes.iloc[:-1].mean()
    current_vol = volumes.iloc[-1]
    rvol = current_vol / avg_vol if avg_vol > 0 else 1.0
    return rvol >= 1.0


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def strategy_ema_cross(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 30:
        return None
    ema9 = EMAIndicator(close=df["close"], window=9).ema_indicator()
    ema21 = EMAIndicator(close=df["close"], window=21).ema_indicator()

    prev_fast, curr_fast = ema9.iloc[-2], ema9.iloc[-1]
    prev_slow, curr_slow = ema21.iloc[-2], ema21.iloc[-1]

    if any(np.isnan(v) for v in [prev_fast, curr_fast, prev_slow, curr_slow]):
        return None

    # Slope confirmation: slow EMA must move in signal direction
    ema21_slope = curr_slow - ema21.iloc[-3] if len(ema21) >= 3 else 0

    if prev_fast <= prev_slow and curr_fast > curr_slow:
        if ema21_slope < 0:
            return None
        action = "BUY"
    elif prev_fast >= prev_slow and curr_fast < curr_slow:
        if ema21_slope > 0:
            return None
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol, action=action, price=float(df["close"].iloc[-1]),
        timeframe="15m", source="signal_engine", indicator="EMA 9/21 Cross",
    )


def strategy_rsi_extreme(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 50:
        return None
    rsi = RSIIndicator(close=df["close"], window=14).rsi()
    current_rsi = rsi.iloc[-1]

    if np.isnan(current_rsi):
        return None

    # Adaptive thresholds based on trend context
    sma50 = df["close"].rolling(50).mean().iloc[-1]
    price = float(df["close"].iloc[-1])

    if np.isnan(sma50):
        buy_threshold, sell_threshold = 30, 70
    elif price > sma50:
        buy_threshold, sell_threshold = 40, 80
    else:
        buy_threshold, sell_threshold = 20, 60

    if current_rsi < buy_threshold:
        action = "BUY"
    elif current_rsi > sell_threshold:
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol, action=action, price=price,
        timeframe="15m", source="signal_engine",
        indicator=f"RSI 14 Reversal ({current_rsi:.1f}, thresh={buy_threshold}/{sell_threshold})",
    )


def strategy_macd_cross(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 50:
        return None
    macd_ind = MACD(close=df["close"])
    macd_line = macd_ind.macd()
    signal_line = macd_ind.macd_signal()

    prev_macd, curr_macd = macd_line.iloc[-2], macd_line.iloc[-1]
    prev_sig, curr_sig = signal_line.iloc[-2], signal_line.iloc[-1]

    if any(np.isnan(v) for v in [prev_macd, curr_macd, prev_sig, curr_sig]):
        return None

    if prev_macd <= prev_sig and curr_macd > curr_sig:
        if curr_macd < 0:
            return None  # Below zero = weak buy
        action = "BUY"
    elif prev_macd >= prev_sig and curr_macd < curr_sig:
        if curr_macd > 0:
            return None  # Above zero = weak sell
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol, action=action, price=float(df["close"].iloc[-1]),
        timeframe="15m", source="signal_engine", indicator="MACD Cross",
    )


def strategy_bollinger_reversion(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 30:
        return None
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    middle = bb.bollinger_mavg().iloc[-1]
    close = float(df["close"].iloc[-1])

    if any(np.isnan(v) for v in [upper, lower, middle]):
        return None

    # Reject if bands are expanding (trending, not reverting)
    if len(df) >= 25:
        upper_prev = bb.bollinger_hband().iloc[-5]
        lower_prev = bb.bollinger_lband().iloc[-5]
        middle_prev = bb.bollinger_mavg().iloc[-5]
        if not any(np.isnan(v) for v in [upper_prev, lower_prev, middle_prev]) and middle_prev > 0:
            bb_width = (upper - lower) / middle
            bb_width_prev = (upper_prev - lower_prev) / middle_prev
            if bb_width > bb_width_prev * 1.3:
                return None

    # Flat bands only
    if middle > 0:
        middle_slope = abs(middle - bb.bollinger_mavg().iloc[-5]) / middle if len(df) >= 25 else 0
        if middle_slope > 0.005:
            return None

    if close <= lower:
        action = "BUY"
    elif close >= upper:
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol, action=action, price=close,
        timeframe="15m", source="signal_engine",
        indicator=f"BB Reversion (L={lower:.5g} U={upper:.5g})",
    )


def strategy_rsi_divergence(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    """Detect bullish/bearish RSI divergence."""
    if len(df) < 50:
        return None

    rsi = RSIIndicator(close=df["close"], window=14).rsi()
    closes = df["close"].values
    rsi_vals = rsi.values

    if np.any(np.isnan(rsi_vals[-30:])):
        return None

    # Find swing lows/highs in last 30 bars using simple comparison
    window = closes[-30:]
    rsi_window = rsi_vals[-30:]
    offset = len(closes) - 30

    lows = []
    highs = []
    for i in range(2, len(window) - 2):
        if window[i] < window[i - 1] and window[i] < window[i - 2] and window[i] < window[i + 1] and window[i] < window[i + 2]:
            lows.append(i)
        if window[i] > window[i - 1] and window[i] > window[i - 2] and window[i] > window[i + 1] and window[i] > window[i + 2]:
            highs.append(i)

    # Bullish divergence: price lower low, RSI higher low
    if len(lows) >= 2:
        i1, i2 = lows[-2], lows[-1]
        if window[i2] < window[i1] and rsi_window[i2] > rsi_window[i1]:
            if rsi_window[i2] < 40:
                return IncomingSignal(
                    symbol=symbol, action="BUY", price=float(closes[-1]),
                    timeframe="15m", source="signal_engine",
                    indicator=f"RSI Divergence (RSI={rsi_vals[-1]:.1f})",
                )

    # Bearish divergence: price higher high, RSI lower high
    if len(highs) >= 2:
        i1, i2 = highs[-2], highs[-1]
        if window[i2] > window[i1] and rsi_window[i2] < rsi_window[i1]:
            if rsi_window[i2] > 60:
                return IncomingSignal(
                    symbol=symbol, action="SELL", price=float(closes[-1]),
                    timeframe="15m", source="signal_engine",
                    indicator=f"RSI Divergence (RSI={rsi_vals[-1]:.1f})",
                )

    return None


STRATEGIES = [
    strategy_ema_cross,
    strategy_rsi_extreme,
    strategy_macd_cross,
    strategy_bollinger_reversion,
    strategy_rsi_divergence,
]

MIN_CONFLUENCE = 2
COOLDOWN_SECONDS = 3600

_signal_cooldown: dict[str, tuple[float, float]] = {}  # key -> (timestamp, price)


async def run_signal_engine() -> list[tuple[IncomingSignal, dict, pd.DataFrame]]:
    logger.info("Signal engine running - scanning %d symbols", len(WATCHLIST))
    signals: list[tuple[IncomingSignal, dict, pd.DataFrame]] = []
    emitted_groups: set[str] = set()

    for symbol in WATCHLIST:
        s = symbol.upper().replace("/", "")

        # Session filter for forex
        if s in FOREX_SYMBOLS_SET and not _is_forex_active_session():
            continue

        df = await fetch_ohlcv(symbol, limit=250)
        if df is None or df.empty:
            continue

        # Chop detector
        if is_choppy(df):
            logger.info("Choppy market detected for %s — suppressing", symbol)
            continue

        raw: list[IncomingSignal] = []
        for strategy in STRATEGIES:
            try:
                signal = strategy(df, symbol)
                if signal:
                    raw.append(signal)
            except Exception:
                logger.exception("Strategy %s failed on %s", strategy.__name__, symbol)

        if not raw:
            continue

        regime = detect_regime(df)
        raw = filter_signals_by_regime(raw, regime)

        if not raw:
            logger.info("Regime filter removed all signals for %s (regime=%s)", symbol, regime)
            continue

        # Volume filter
        if not check_volume_confirmation(df):
            logger.info("Volume too low for %s — suppressing signals", symbol)
            continue

        buy_signals = [s for s in raw if s.action == "BUY"]
        sell_signals = [s for s in raw if s.action == "SELL"]

        if len(buy_signals) >= MIN_CONFLUENCE:
            direction, agreeing = "BUY", buy_signals
        elif len(sell_signals) >= MIN_CONFLUENCE:
            direction, agreeing = "SELL", sell_signals
        else:
            logger.info(
                "No confluence for %s (BUY=%d, SELL=%d) - skipping",
                symbol, len(buy_signals), len(sell_signals),
            )
            continue

        # Correlation filter
        group = _get_correlation_group(symbol)
        if group and group in emitted_groups:
            logger.info(
                "Correlation filter: %s %s suppressed (group %s already active)",
                direction, symbol, group,
            )
            continue

        # Price-movement-aware cooldown
        cooldown_key = f"{symbol}:{direction}"
        now = time.monotonic()
        last = _signal_cooldown.get(cooldown_key)
        if last is not None:
            last_time, last_price = last
            time_elapsed = now - last_time
            price_moved = abs(float(df["close"].iloc[-1]) - last_price) / last_price
            if time_elapsed < COOLDOWN_SECONDS and price_moved < 0.01:
                logger.info("Signal suppressed for %s %s — cooldown active", symbol, direction)
                continue

        indicators = ", ".join(sig.indicator for sig in agreeing if sig.indicator)
        signal = IncomingSignal(
            symbol=symbol,
            action=direction,
            price=float(df["close"].iloc[-1]),
            timeframe="15m",
            source="signal_engine",
            indicator=f"Confluence: {indicators}",
            confluence_count=len(agreeing),
        )
        bias = get_htf_bias(symbol, direction)
        signals.append((signal, bias, df))
        _signal_cooldown[cooldown_key] = (now, float(df["close"].iloc[-1]))
        if group:
            emitted_groups.add(group)
        logger.info(
            "Confluence %s on %s (%d/%d strategies agree)",
            direction, symbol, len(agreeing), len(STRATEGIES),
        )

    # Purge expired cooldown entries
    expired = [k for k, v in _signal_cooldown.items() if (time.monotonic() - v[0]) > COOLDOWN_SECONDS * 2]
    for k in expired:
        del _signal_cooldown[k]

    logger.info("Signal engine complete - %d signals emitted", len(signals))
    return signals

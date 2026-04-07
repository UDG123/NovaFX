import logging
import time
from typing import Optional

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import EMAIndicator, MACD
from ta.volatility import BollingerBands

from app.data.fetcher import fetch_ohlcv
from app.models.signals import IncomingSignal
from app.services.htf_bias import get_htf_bias
from app.services.regime import detect_regime, filter_signals_by_regime

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


def strategy_ema_cross(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 21:
        return None
    ema9 = EMAIndicator(close=df["close"], window=9).ema_indicator()
    ema21 = EMAIndicator(close=df["close"], window=21).ema_indicator()

    prev_fast, curr_fast = ema9.iloc[-2], ema9.iloc[-1]
    prev_slow, curr_slow = ema21.iloc[-2], ema21.iloc[-1]

    if prev_fast <= prev_slow and curr_fast > curr_slow:
        action = "BUY"
    elif prev_fast >= prev_slow and curr_fast < curr_slow:
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol,
        action=action,
        price=float(df["close"].iloc[-1]),
        timeframe="15m",
        source="signal_engine",
        indicator="EMA 9/21 Cross",
    )


def strategy_rsi_extreme(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 15:
        return None
    rsi = RSIIndicator(close=df["close"], window=14).rsi()
    current_rsi = rsi.iloc[-1]

    if np.isnan(current_rsi):
        return None

    if current_rsi < 30:
        action = "BUY"
    elif current_rsi > 70:
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol,
        action=action,
        price=float(df["close"].iloc[-1]),
        timeframe="15m",
        source="signal_engine",
        indicator=f"RSI 14 Reversal ({current_rsi:.1f})",
    )


def strategy_macd_cross(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 35:
        return None
    macd_ind = MACD(close=df["close"])
    macd_line = macd_ind.macd()
    signal_line = macd_ind.macd_signal()

    prev_macd, curr_macd = macd_line.iloc[-2], macd_line.iloc[-1]
    prev_sig, curr_sig = signal_line.iloc[-2], signal_line.iloc[-1]

    if prev_macd <= prev_sig and curr_macd > curr_sig:
        action = "BUY"
    elif prev_macd >= prev_sig and curr_macd < curr_sig:
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol,
        action=action,
        price=float(df["close"].iloc[-1]),
        timeframe="15m",
        source="signal_engine",
        indicator="MACD Cross",
    )


def strategy_bollinger_reversion(df: pd.DataFrame, symbol: str) -> Optional[IncomingSignal]:
    if len(df) < 20:
        return None
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    upper = bb.bollinger_hband().iloc[-1]
    lower = bb.bollinger_lband().iloc[-1]
    close = float(df["close"].iloc[-1])

    if np.isnan(upper) or np.isnan(lower):
        return None

    if close <= lower:
        action = "BUY"
    elif close >= upper:
        action = "SELL"
    else:
        return None

    return IncomingSignal(
        symbol=symbol,
        action=action,
        price=close,
        timeframe="15m",
        source="signal_engine",
        indicator=f"BB Reversion (L={lower:.5g} U={upper:.5g})",
    )


STRATEGIES = [
    strategy_ema_cross,
    strategy_rsi_extreme,
    strategy_macd_cross,
    strategy_bollinger_reversion,
]

MIN_CONFLUENCE = 2
COOLDOWN_SECONDS = 3600  # 1 hour — don't re-signal same symbol+direction

_signal_cooldown: dict[str, float] = {}  # "SYMBOL:DIRECTION" -> timestamp


async def run_signal_engine() -> list[tuple[IncomingSignal, dict]]:
    logger.info("Signal engine running - scanning %d symbols", len(WATCHLIST))
    signals: list[tuple[IncomingSignal, dict]] = []

    for symbol in WATCHLIST:
        df = await fetch_ohlcv(symbol)
        if df is None or df.empty:
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

        # Cooldown check — suppress duplicate signals within window
        cooldown_key = f"{symbol}:{direction}"
        now = time.monotonic()
        last_emitted = _signal_cooldown.get(cooldown_key)
        if last_emitted is not None and (now - last_emitted) < COOLDOWN_SECONDS:
            logger.info(
                "Signal suppressed for %s %s — cooldown active (%ds)",
                symbol, direction, COOLDOWN_SECONDS,
            )
            continue

        indicators = ", ".join(s.indicator for s in agreeing if s.indicator)
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
        signals.append((signal, bias))
        _signal_cooldown[cooldown_key] = now
        logger.info(
            "Confluence %s on %s (%d/%d strategies agree)",
            direction, symbol, len(agreeing), len(STRATEGIES),
        )

    # Purge expired cooldown entries
    expired = [k for k, v in _signal_cooldown.items() if (time.monotonic() - v) > COOLDOWN_SECONDS]
    for k in expired:
        del _signal_cooldown[k]

    logger.info("Signal engine complete - %d signals emitted", len(signals))
    return signals

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def make_ohlcv():
    """Factory fixture: returns a function that builds synthetic OHLCV DataFrames.

    Usage:
        df = make_ohlcv(prices)          # close = prices, open/high/low derived
        df = make_ohlcv(prices, volume=500)
    """

    def _build(close_prices: list[float], volume: int = 100) -> pd.DataFrame:
        closes = np.array(close_prices, dtype=float)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) * 1.002
        lows = np.minimum(opens, closes) * 0.998
        return pd.DataFrame({
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": [volume] * len(closes),
        })

    return _build


@pytest.fixture
def bullish_ema_df(make_ohlcv):
    """50-bar DataFrame where EMA9 crosses above EMA21 between [-2] and [-1].

    Slow decline keeps EMA9 below EMA21, then a 5-bar uptick at the end
    produces the crossover right at the last bar.
    """
    decline = [100.0 - i * 0.3 for i in range(45)]
    uptick = [decline[-1] + 0.5, decline[-1] + 1.5, decline[-1] + 3.0,
              decline[-1] + 5.0, decline[-1] + 8.0]
    return make_ohlcv(decline + uptick)


@pytest.fixture
def bearish_ema_df(make_ohlcv):
    """50-bar DataFrame where EMA9 crosses below EMA21 between [-2] and [-1]."""
    incline = [100.0 + i * 0.3 for i in range(45)]
    downtick = [incline[-1] - 0.5, incline[-1] - 1.5, incline[-1] - 3.0,
                incline[-1] - 5.0, incline[-1] - 8.0]
    return make_ohlcv(incline + downtick)


@pytest.fixture
def oversold_rsi_df(make_ohlcv):
    """50-bar DataFrame that drives RSI14 below 30 (oversold -> BUY)."""
    prices = [100.0 - i * 0.8 for i in range(50)]
    return make_ohlcv(prices)


@pytest.fixture
def overbought_rsi_df(make_ohlcv):
    """50-bar DataFrame that drives RSI14 above 70 (overbought -> SELL)."""
    prices = [100.0 + i * 0.8 for i in range(50)]
    return make_ohlcv(prices)


@pytest.fixture
def bullish_macd_df(make_ohlcv):
    """70-bar DataFrame that produces a bullish MACD crossover at [-1].

    Steady decline drives MACD below signal, then a single uptick bar at the
    end pushes MACD line above signal line.
    """
    decline = [100.0 - i * 0.15 for i in range(69)]
    decline.append(decline[-1] + 0.8)
    return make_ohlcv(decline)


@pytest.fixture
def bearish_macd_df(make_ohlcv):
    """70-bar DataFrame that produces a bearish MACD crossover at [-1]."""
    incline = [100.0 + i * 0.15 for i in range(69)]
    incline.append(incline[-1] - 0.8)
    return make_ohlcv(incline)


@pytest.fixture
def flat_df(make_ohlcv):
    """50-bar constant-price DataFrame — no strategy should trigger."""
    return make_ohlcv([100.0] * 50)


@pytest.fixture
def backtest_df(make_ohlcv):
    """10-bar DataFrame for backtest engine tests with known prices."""
    prices = [100, 102, 104, 103, 101, 105, 108, 107, 110, 112]
    return make_ohlcv(prices)

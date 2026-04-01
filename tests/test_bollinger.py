"""Tests for Bollinger Band Reversion strategy, backtester registry, and strategy_state."""

import numpy as np
import pandas as pd
import pytest

from app.services.signal_engine import strategy_bollinger_reversion
from app.services.strategy_state import VALID_STRATEGIES, is_valid_strategy
from backtester.app.strategies.bollinger_reversion import generate_signals
from backtester.app.strategies.registry import (
    STRATEGY_REGISTRY,
    get_strategy,
    list_strategies,
)


@pytest.fixture
def make_ohlcv():
    def _build(close_prices: list[float], volume: int = 100) -> pd.DataFrame:
        closes = np.array(close_prices, dtype=float)
        opens = np.roll(closes, 1)
        opens[0] = closes[0]
        highs = np.maximum(opens, closes) * 1.002
        lows = np.minimum(opens, closes) * 0.998
        return pd.DataFrame({
            "open": opens, "high": highs, "low": lows,
            "close": closes, "volume": [volume] * len(closes),
        })
    return _build


# ── NovaFX signal engine: strategy_bollinger_reversion ────────────────────────


class TestBollingerReversionSignal:
    def test_buy_at_lower_band(self, make_ohlcv):
        """Price dropping below lower band triggers BUY."""
        prices = [100.0] * 30 + [100.0 - i * 0.5 for i in range(1, 11)]
        df = make_ohlcv(prices)
        signal = strategy_bollinger_reversion(df, "EURUSD")
        assert signal is not None
        assert signal.action == "BUY"
        assert "BB Reversion" in signal.indicator
        assert signal.symbol == "EURUSD"

    def test_sell_at_upper_band(self, make_ohlcv):
        """Price rising above upper band triggers SELL."""
        prices = [100.0] * 30 + [100.0 + i * 0.5 for i in range(1, 11)]
        df = make_ohlcv(prices)
        signal = strategy_bollinger_reversion(df, "BTCUSD")
        assert signal is not None
        assert signal.action == "SELL"
        assert "BB Reversion" in signal.indicator

    def test_price_inside_bands_returns_none(self, make_ohlcv):
        """Price oscillating gently within bands returns None."""
        prices = [100.0 + 0.05 * (i % 3 - 1) for i in range(40)]
        df = make_ohlcv(prices)
        signal = strategy_bollinger_reversion(df, "EURUSD")
        assert signal is None

    def test_insufficient_data_returns_none(self, make_ohlcv):
        df = make_ohlcv([100.0] * 10)
        signal = strategy_bollinger_reversion(df, "EURUSD")
        assert signal is None

    def test_price_matches_last_close(self, make_ohlcv):
        prices = [100.0] * 30 + [100.0 - i * 0.5 for i in range(1, 11)]
        df = make_ohlcv(prices)
        signal = strategy_bollinger_reversion(df, "EURUSD")
        assert signal is not None
        assert signal.price == float(df["close"].iloc[-1])

    def test_indicator_contains_band_values(self, make_ohlcv):
        prices = [100.0] * 30 + [100.0 - i * 0.5 for i in range(1, 11)]
        df = make_ohlcv(prices)
        signal = strategy_bollinger_reversion(df, "EURUSD")
        assert "L=" in signal.indicator
        assert "U=" in signal.indicator


# ── Backtester: generate_signals ──────────────────────────────────────────────


class TestBollingerBacktesterSignals:
    def test_generates_buy_signals_on_lower_touch(self, make_ohlcv):
        """Drop to lower band then revert to SMA should produce a BUY signal."""
        prices = [100.0] * 25 + [100.0 - i * 0.6 for i in range(1, 8)]
        # Then revert back up past the middle band
        prices += [prices[-1] + i * 1.0 for i in range(1, 10)]
        df = make_ohlcv(prices)
        signals = generate_signals(df)
        buy_signals = [s for s in signals if s["action"] == "BUY"]
        assert len(buy_signals) >= 1
        for s in buy_signals:
            assert s["entry_idx"] < s["exit_idx"]

    def test_generates_sell_signals_on_upper_touch(self, make_ohlcv):
        """Rise to upper band then revert to SMA should produce a SELL signal."""
        prices = [100.0] * 25 + [100.0 + i * 0.6 for i in range(1, 8)]
        prices += [prices[-1] - i * 1.0 for i in range(1, 10)]
        df = make_ohlcv(prices)
        signals = generate_signals(df)
        sell_signals = [s for s in signals if s["action"] == "SELL"]
        assert len(sell_signals) >= 1

    def test_empty_on_insufficient_data(self, make_ohlcv):
        df = make_ohlcv([100.0] * 10)
        assert generate_signals(df) == []

    def test_no_signals_when_price_stays_inside(self, make_ohlcv):
        prices = [100.0 + 0.05 * (i % 3 - 1) for i in range(50)]
        df = make_ohlcv(prices)
        assert generate_signals(df) == []


# ── Backtester registry ───────────────────────────────────────────────────────


class TestStrategyRegistry:
    def test_bollinger_registered(self):
        assert "BollingerBandReversion" in STRATEGY_REGISTRY

    def test_get_strategy_returns_module(self):
        mod = get_strategy("BollingerBandReversion")
        assert hasattr(mod, "generate_signals")

    def test_get_strategy_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown strategy"):
            get_strategy("NonExistent")

    def test_list_strategies_includes_bollinger(self):
        names = list_strategies()
        assert "BollingerBandReversion" in names


# ── NovaFX strategy_state ────────────────────────────────────────────────────


class TestStrategyState:
    def test_all_four_strategies_listed(self):
        assert len(VALID_STRATEGIES) == 4
        assert "EMA 9/21 Cross" in VALID_STRATEGIES
        assert "RSI 14 Reversal" in VALID_STRATEGIES
        assert "MACD Cross" in VALID_STRATEGIES
        assert "BollingerBandReversion" in VALID_STRATEGIES

    def test_is_valid_strategy_true(self):
        assert is_valid_strategy("BollingerBandReversion") is True

    def test_is_valid_strategy_false(self):
        assert is_valid_strategy("FakeStrategy") is False

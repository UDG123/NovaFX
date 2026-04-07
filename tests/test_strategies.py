"""Tests for signal engine strategies (EMA cross, RSI extreme, MACD cross)
and the confluence filter in run_signal_engine."""

from unittest.mock import AsyncMock, patch

import pytest

from app.models.signals import IncomingSignal
from app.services.signal_engine import (
    MIN_CONFLUENCE,
    run_signal_engine,
    strategy_ema_cross,
    strategy_macd_cross,
    strategy_rsi_extreme,
)


class TestEMACross:
    def test_bullish_crossover_returns_buy(self, bullish_ema_df):
        signal = strategy_ema_cross(bullish_ema_df, "EURUSD")
        assert signal is not None
        assert signal.action == "BUY"
        assert signal.indicator == "EMA 9/21 Cross"
        assert signal.symbol == "EURUSD"

    def test_bearish_crossover_returns_sell(self, bearish_ema_df):
        signal = strategy_ema_cross(bearish_ema_df, "EURUSD")
        assert signal is not None
        assert signal.action == "SELL"

    def test_flat_market_returns_none(self, flat_df):
        signal = strategy_ema_cross(flat_df, "EURUSD")
        assert signal is None

    def test_insufficient_data_returns_none(self, make_ohlcv):
        short_df = make_ohlcv([100.0] * 10)
        signal = strategy_ema_cross(short_df, "EURUSD")
        assert signal is None

    def test_price_matches_last_close(self, bullish_ema_df):
        signal = strategy_ema_cross(bullish_ema_df, "EURUSD")
        assert signal is not None
        assert signal.price == float(bullish_ema_df["close"].iloc[-1])


class TestRSIExtreme:
    def test_oversold_returns_buy(self, oversold_rsi_df):
        signal = strategy_rsi_extreme(oversold_rsi_df, "BTCUSD")
        assert signal is not None
        assert signal.action == "BUY"
        assert "RSI 14 Reversal" in signal.indicator

    def test_overbought_returns_sell(self, overbought_rsi_df):
        signal = strategy_rsi_extreme(overbought_rsi_df, "BTCUSD")
        assert signal is not None
        assert signal.action == "SELL"

    def test_neutral_rsi_returns_none(self, make_ohlcv):
        """RSI around 50 (alternating up/down) should not trigger."""
        prices = [100.0 + (1 if i % 2 == 0 else -1) for i in range(50)]
        df = make_ohlcv(prices)
        signal = strategy_rsi_extreme(df, "BTCUSD")
        assert signal is None

    def test_insufficient_data_returns_none(self, make_ohlcv):
        short_df = make_ohlcv([100.0] * 5)
        signal = strategy_rsi_extreme(short_df, "BTCUSD")
        assert signal is None


class TestMACDCross:
    def test_bullish_crossover_returns_buy(self, bullish_macd_df):
        signal = strategy_macd_cross(bullish_macd_df, "XAUUSD")
        assert signal is not None
        assert signal.action == "BUY"
        assert signal.indicator == "MACD Cross"

    def test_bearish_crossover_returns_sell(self, bearish_macd_df):
        signal = strategy_macd_cross(bearish_macd_df, "XAUUSD")
        assert signal is not None
        assert signal.action == "SELL"

    def test_flat_market_returns_none(self, flat_df):
        signal = strategy_macd_cross(flat_df, "XAUUSD")
        assert signal is None

    def test_insufficient_data_returns_none(self, make_ohlcv):
        short_df = make_ohlcv([100.0] * 20)
        signal = strategy_macd_cross(short_df, "XAUUSD")
        assert signal is None


_DUMMY_BIAS = {
    "strength": "MODERATE",
    "h1_trend": "neutral",
    "h4_trend": "neutral",
    "signal_action": "BUY",
    "agreements": 1,
    "emoji": "\u26a0\ufe0f",
    "label": "test",
}


def _confluence_patches():
    """Return common patches for regime filter and HTF bias."""
    return (
        patch("app.services.signal_engine.detect_regime", return_value="trending"),
        patch("app.services.signal_engine.filter_signals_by_regime", side_effect=lambda signals, regime: signals),
        patch("app.services.signal_engine.get_htf_bias", return_value=_DUMMY_BIAS),
    )


class TestConfluenceFilter:
    """Tests that run_signal_engine only emits signals when >=2 strategies agree."""

    @pytest.mark.asyncio
    async def test_two_buy_signals_emits(self):
        """Two BUY + one None -> emits a BUY with confluence_count=2."""
        buy = IncomingSignal(symbol="EURUSD", action="BUY", price=1.1, source="signal_engine", indicator="A")
        buy2 = IncomingSignal(symbol="EURUSD", action="BUY", price=1.1, source="signal_engine", indicator="B")

        p1, p2, p3 = _confluence_patches()
        with patch("app.services.signal_engine.fetch_ohlcv", new_callable=AsyncMock) as mock_fetch, \
             patch("app.services.signal_engine.WATCHLIST", ["EURUSD"]), \
             p1, p2, p3, \
             patch("app.services.signal_engine.STRATEGIES", [lambda df, s: buy, lambda df, s: buy2, lambda df, s: None]):
            mock_fetch.return_value = _dummy_df()
            signals = await run_signal_engine()

        assert len(signals) == 1
        signal, bias = signals[0]
        assert signal.action == "BUY"
        assert signal.confluence_count == 2
        assert "A" in signal.indicator
        assert "B" in signal.indicator

    @pytest.mark.asyncio
    async def test_three_sell_signals_emits(self):
        """All three SELL -> emits with confluence_count=3."""
        sell = lambda df, s: IncomingSignal(symbol="EURUSD", action="SELL", price=1.1, source="signal_engine", indicator="X")

        p1, p2, p3 = _confluence_patches()
        with patch("app.services.signal_engine.fetch_ohlcv", new_callable=AsyncMock) as mock_fetch, \
             patch("app.services.signal_engine.WATCHLIST", ["EURUSD"]), \
             p1, p2, p3, \
             patch("app.services.signal_engine.STRATEGIES", [sell, sell, sell]):
            mock_fetch.return_value = _dummy_df()
            signals = await run_signal_engine()

        assert len(signals) == 1
        signal, bias = signals[0]
        assert signal.confluence_count == 3

    @pytest.mark.asyncio
    async def test_one_signal_only_no_confluence(self):
        """Only one strategy triggers -> no signal emitted."""
        buy = IncomingSignal(symbol="EURUSD", action="BUY", price=1.1, source="signal_engine", indicator="A")

        p1, p2, p3 = _confluence_patches()
        with patch("app.services.signal_engine.fetch_ohlcv", new_callable=AsyncMock) as mock_fetch, \
             patch("app.services.signal_engine.WATCHLIST", ["EURUSD"]), \
             p1, p2, p3, \
             patch("app.services.signal_engine.STRATEGIES", [lambda df, s: buy, lambda df, s: None, lambda df, s: None]):
            mock_fetch.return_value = _dummy_df()
            signals = await run_signal_engine()

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_mixed_directions_no_confluence(self):
        """One BUY + one SELL + one None -> no majority, no signal."""
        buy = IncomingSignal(symbol="EURUSD", action="BUY", price=1.1, source="signal_engine", indicator="A")
        sell = IncomingSignal(symbol="EURUSD", action="SELL", price=1.1, source="signal_engine", indicator="B")

        p1, p2, p3 = _confluence_patches()
        with patch("app.services.signal_engine.fetch_ohlcv", new_callable=AsyncMock) as mock_fetch, \
             patch("app.services.signal_engine.WATCHLIST", ["EURUSD"]), \
             p1, p2, p3, \
             patch("app.services.signal_engine.STRATEGIES", [lambda df, s: buy, lambda df, s: sell, lambda df, s: None]):
            mock_fetch.return_value = _dummy_df()
            signals = await run_signal_engine()

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_no_data_no_signals(self):
        """fetch_ohlcv returns None -> no signals."""
        with patch("app.services.signal_engine.fetch_ohlcv", new_callable=AsyncMock) as mock_fetch, \
             patch("app.services.signal_engine.WATCHLIST", ["EURUSD"]):
            mock_fetch.return_value = None
            signals = await run_signal_engine()

        assert len(signals) == 0

    @pytest.mark.asyncio
    async def test_multiple_symbols_independent(self):
        """Confluence is checked per-symbol, not across symbols."""
        buy_a = IncomingSignal(symbol="EURUSD", action="BUY", price=1.1, source="signal_engine", indicator="A")
        buy_b = IncomingSignal(symbol="GBPUSD", action="BUY", price=1.3, source="signal_engine", indicator="B")

        p1, p2, p3 = _confluence_patches()
        with patch("app.services.signal_engine.fetch_ohlcv", new_callable=AsyncMock) as mock_fetch, \
             patch("app.services.signal_engine.WATCHLIST", ["EURUSD", "GBPUSD"]), \
             p1, p2, p3, \
             patch("app.services.signal_engine.STRATEGIES", [
                 lambda df, s: buy_a if s == "EURUSD" else buy_b,
                 lambda df, s: None,
                 lambda df, s: None,
             ]):
            mock_fetch.return_value = _dummy_df()
            signals = await run_signal_engine()

        assert len(signals) == 0

    def test_min_confluence_is_two(self):
        assert MIN_CONFLUENCE == 2


def _dummy_df():
    import pandas as pd
    return pd.DataFrame({
        "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0], "volume": [100],
    })

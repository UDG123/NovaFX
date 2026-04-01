"""Tests for signal engine strategies (EMA cross, RSI extreme, MACD cross)."""

from app.services.signal_engine import (
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

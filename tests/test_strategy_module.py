"""Unit tests for the modular strategy classes in src/strategies/."""

import numpy as np
import pandas as pd
import pytest

from src.strategies import get_strategy, list_strategies, STRATEGY_REGISTRY, BaseStrategy


@pytest.fixture
def sample_data():
    """Generate 500-bar trending OHLCV data."""
    np.random.seed(42)
    n = 500
    dates = pd.date_range(start="2024-01-01", periods=n, freq="1h")
    returns = np.random.randn(n) * 0.01 + 0.0001
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.005)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    volume = np.random.randint(1000, 10000, n).astype(float)

    return pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)


class TestRegistry:
    def test_all_strategies_registered(self):
        expected = ["ema_cross", "momentum_breakout", "donchian_breakout",
                     "macd_trend", "macd_zero", "bb_reversion",
                     "rsi_adaptive", "rsi_divergence"]
        for name in expected:
            assert name in STRATEGY_REGISTRY, f"{name} missing from registry"

    def test_get_strategy_returns_instance(self):
        for name in STRATEGY_REGISTRY:
            s = get_strategy(name)
            assert isinstance(s, BaseStrategy)
            assert s.name == name

    def test_get_strategy_with_params(self):
        s = get_strategy("momentum_breakout", params={"lookback": 30})
        assert s.params["lookback"] == 30

    def test_get_strategy_invalid(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("does_not_exist")

    def test_list_strategies(self):
        result = list_strategies()
        assert len(result) == len(STRATEGY_REGISTRY)
        for meta in result.values():
            assert "default_params" in meta
            assert "param_grid" in meta


class TestSignalGeneration:
    @pytest.mark.parametrize("name", list(STRATEGY_REGISTRY.keys()))
    def test_returns_correct_columns(self, name, sample_data):
        s = get_strategy(name)
        result = s.generate_signals(sample_data)
        assert isinstance(result, pd.DataFrame)
        for col in ["signal", "entry_price", "stop_loss", "take_profit"]:
            assert col in result.columns, f"{col} missing from {name}"
        assert len(result) == len(sample_data)

    @pytest.mark.parametrize("name", list(STRATEGY_REGISTRY.keys()))
    def test_signals_valid_values(self, name, sample_data):
        s = get_strategy(name)
        result = s.generate_signals(sample_data)
        assert result["signal"].isin([-1, 0, 1]).all(), f"{name} has invalid signal values"

    @pytest.mark.parametrize("name", list(STRATEGY_REGISTRY.keys()))
    def test_stops_set_for_signals(self, name, sample_data):
        s = get_strategy(name)
        result = s.generate_signals(sample_data)
        sigs = result[result["signal"] != 0]
        if len(sigs) > 0:
            assert sigs["stop_loss"].notna().all(), f"{name} has NaN stop_loss"
            assert sigs["take_profit"].notna().all(), f"{name} has NaN take_profit"


class TestRSIDivergenceFix:
    def test_ranging_regime_only(self):
        s = get_strategy("rsi_divergence")
        assert "ranging" in s.config.allowed_regimes
        assert "trending" not in s.config.allowed_regimes

    def test_validate_regime(self):
        s = get_strategy("rsi_divergence")
        assert s.validate_regime("ranging")
        assert s.validate_regime("mean_reverting")
        assert not s.validate_regime("trending")
        assert not s.validate_regime("bull")


class TestBacktestBridge:
    def test_run_strategy_class(self, sample_data):
        from scripts.backtest_harness import run_strategy_class
        result = run_strategy_class("macd_trend", sample_data)
        assert isinstance(result, pd.DataFrame)
        assert "signal" in result.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

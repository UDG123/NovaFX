"""Tests for VectorBT adapters."""

import numpy as np
import pandas as pd
import pytest

vbt = pytest.importorskip("vectorbt")

from src.vectorbt_adapters import (
    VBTStrategyAdapter, VBTParamOptimizer, vbt_backtest, prepare_vbt_data,
)
from src.strategies import STRATEGY_REGISTRY


@pytest.fixture
def sample_data():
    np.random.seed(42)
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="1h")
    returns = np.random.randn(n) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "open": close * (1 + np.random.randn(n) * 0.001),
        "high": close * (1 + np.abs(np.random.randn(n)) * 0.005),
        "low": close * (1 - np.abs(np.random.randn(n)) * 0.005),
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }, index=dates)


class TestVBTStrategyAdapter:
    def test_init(self, sample_data):
        adapter = VBTStrategyAdapter("macd_trend", sample_data, "BTC-USD")
        assert adapter.strategy_name == "macd_trend"

    def test_generate_signals(self, sample_data):
        adapter = VBTStrategyAdapter("macd_trend", sample_data, "BTC-USD")
        signals = adapter.generate_signals()
        assert "long_entries" in signals
        assert "short_entries" in signals
        assert len(signals["long_entries"]) == len(sample_data)

    def test_run_backtest(self, sample_data):
        adapter = VBTStrategyAdapter("ema_cross", sample_data, "BTC-USD")
        result = adapter.run_backtest()
        assert hasattr(result, "total_return")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "portfolio")

    def test_get_param_grid(self, sample_data):
        adapter = VBTStrategyAdapter("momentum_breakout", sample_data, "BTC-USD")
        grid = adapter.get_param_grid()
        assert "lookback" in grid
        assert isinstance(grid["lookback"], list)


class TestVBTParamOptimizer:
    def test_optimize_small_grid(self, sample_data):
        opt = VBTParamOptimizer("macd_zero", sample_data, "BTC-USD")
        small_grid = {"fast": [10, 12], "slow": [24, 26], "signal": [9]}
        result = opt.optimize(param_grid=small_grid, min_trades=1, verbose=False)
        assert result.best_params is not None
        assert result.n_valid > 0


class TestConvenience:
    def test_vbt_backtest(self, sample_data):
        result = vbt_backtest("ema_cross", sample_data, "BTC-USD")
        assert result.total_return is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

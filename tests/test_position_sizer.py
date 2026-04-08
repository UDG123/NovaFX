"""Tests for position sizing module."""
import pytest
from src.execution.position_sizer import (
    PositionSizer, SizingConfig, SizingMethod,
    fixed_fractional_size, kelly_size, volatility_adjusted_size,
)


class TestFixedFractional:
    def test_basic(self):
        # 2% risk, 5% stop = 40% position
        s = fixed_fractional_size(100000, 100, 95, 0.02)
        assert abs(s - 0.40) < 0.01

    def test_tight_stop(self):
        s = fixed_fractional_size(100000, 100, 99, 0.01)
        assert abs(s - 1.0) < 0.01


class TestKelly:
    def test_positive(self):
        # Kelly = 0.55 - 0.45/1.5 = 0.25, half = 0.125
        s = kelly_size(0.55, 0.03, 0.02, fraction=0.5)
        assert s > 0

    def test_negative(self):
        s = kelly_size(0.40, 0.02, 0.03, fraction=1.0)
        assert s == 0


class TestVolAdj:
    def test_high_vol(self):
        adj = volatility_adjusted_size(0.10, 0.60, 0.15)
        assert adj == 0.10 * 0.30  # Floored at 0.3x

    def test_low_vol(self):
        adj = volatility_adjusted_size(0.10, 0.10, 0.15)
        assert abs(adj - 0.10 * 1.5) < 1e-9


class TestPositionSizer:
    @pytest.fixture
    def sizer(self):
        return PositionSizer(SizingConfig(method=SizingMethod.ADAPTIVE,
                                           risk_per_trade=0.02, max_position_pct=0.20))

    def test_basic(self, sizer):
        r = sizer.calculate_size(100000, 50000, 48000)
        assert 0 < r.position_size <= 0.20
        assert r.dollar_amount > 0
        assert r.position_units > 0

    def test_drawdown_scaling(self, sizer):
        # Use wider stop so base size doesn't hit the 20% cap
        normal = sizer.calculate_size(100000, 100, 90, current_drawdown=0.0)
        reduced = sizer.calculate_size(100000, 100, 90, current_drawdown=0.15)
        assert reduced.position_size < normal.position_size
        assert reduced.dd_adjustment < 1.0

    def test_dd_halt(self, sizer):
        r = sizer.calculate_size(100000, 100, 95, current_drawdown=0.30)
        assert r.position_size == 0
        assert r.dd_adjustment == 0

    def test_regime(self, sizer):
        trend = sizer.calculate_size(100000, 100, 95, regime="trending")
        vol = sizer.calculate_size(100000, 100, 95, regime="volatile")
        assert vol.position_size < trend.position_size

    def test_cap(self, sizer):
        r = sizer.calculate_size(100000, 100, 99.5)
        assert r.position_size == 0.20
        assert r.capped

    def test_notes(self, sizer):
        r = sizer.calculate_size(100000, 100, 95, regime="ranging", current_drawdown=0.12)
        assert any("Base" in n for n in r.notes)

    def test_kelly_integration(self, sizer):
        stats = {"win_rate": 0.55, "avg_win": 0.03, "avg_loss": 0.02, "n_trades": 50}
        r = sizer.calculate_size(100000, 100, 95, strategy_stats=stats)
        assert r.kelly_adjustment != 1.0  # Kelly should have adjusted

    def test_correlation_block(self, sizer):
        existing = {"ETH-USD": 0.25}
        corrs = {"ETH-USD": 0.85}
        r = sizer.calculate_size(100000, 50000, 48000,
                                  existing_positions=existing, correlations=corrs)
        assert r.correlation_adjustment < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

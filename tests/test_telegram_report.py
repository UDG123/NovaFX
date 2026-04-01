"""Tests for backtester cycle report formatting."""

from backtester.app.models.backtest import PhaseResult
from backtester.app.services.telegram import (
    TELEGRAM_MAX_LENGTH,
    _progress_bar,
    _win_rate_indicator,
    format_cycle_report,
)


def _make_result(strategy: str, win_rate: float = 50.0, net_pnl_pct: float = 0.0, **kw) -> PhaseResult:
    defaults = dict(
        phase="backtest", symbol="EURUSD", strategy=strategy,
        total_trades=10, wins=int(win_rate / 10), losses=10 - int(win_rate / 10),
        win_rate=win_rate, gross_pnl_pct=net_pnl_pct + 0.5,
        total_commission_pct=0.5, net_pnl_pct=net_pnl_pct, max_drawdown_pct=1.0,
    )
    defaults.update(kw)
    return PhaseResult(**defaults)


class TestProgressBar:
    def test_zero_value(self):
        bar = _progress_bar(0.0)
        assert len(bar) == 10
        assert "\u2588" not in bar

    def test_full_value(self):
        bar = _progress_bar(100.0)
        assert bar == "\u2588" * 10

    def test_half_value(self):
        bar = _progress_bar(50.0)
        assert "\u2588" in bar
        assert len(bar) == 10

    def test_over_max_clamps(self):
        bar = _progress_bar(200.0, max_value=100.0)
        assert bar == "\u2588" * 10

    def test_negative_treated_as_zero(self):
        bar = _progress_bar(-10.0)
        assert "\u2588" not in bar


class TestWinRateIndicator:
    def test_green_above_55(self):
        assert _win_rate_indicator(60.0) == "\U0001f7e2"

    def test_yellow_between_45_55(self):
        assert _win_rate_indicator(50.0) == "\U0001f7e1"
        assert _win_rate_indicator(45.0) == "\U0001f7e1"

    def test_red_below_45(self):
        assert _win_rate_indicator(44.9) == "\U0001f534"
        assert _win_rate_indicator(0.0) == "\U0001f534"


class TestFormatCycleReport:
    def test_empty_results(self):
        msg = format_cycle_report([])
        assert "No results" in msg
        assert len(msg) <= TELEGRAM_MAX_LENGTH

    def test_single_strategy(self):
        results = [_make_result("EMA Cross", win_rate=65.0, net_pnl_pct=5.0)]
        msg = format_cycle_report(results)
        assert "EMA Cross" in msg
        assert "65.0%" in msg
        assert "\U0001f3c6" in msg  # trophy for best
        assert len(msg) <= TELEGRAM_MAX_LENGTH

    def test_best_strategy_highlighted(self):
        results = [
            _make_result("EMA", win_rate=40.0, net_pnl_pct=1.0),
            _make_result("RSI", win_rate=70.0, net_pnl_pct=8.0),
            _make_result("MACD", win_rate=55.0, net_pnl_pct=3.0),
        ]
        msg = format_cycle_report(results)
        # RSI should be the best (highest composite score)
        assert "Best: RSI" in msg

    def test_win_rate_colors_present(self):
        results = [
            _make_result("good", win_rate=60.0),
            _make_result("mid", win_rate=50.0),
            _make_result("bad", win_rate=30.0),
        ]
        msg = format_cycle_report(results)
        assert "\U0001f7e2" in msg  # green
        assert "\U0001f7e1" in msg  # yellow
        assert "\U0001f534" in msg  # red

    def test_progress_bars_present(self):
        results = [_make_result("EMA", win_rate=75.0, net_pnl_pct=10.0)]
        msg = format_cycle_report(results)
        assert "\u2588" in msg  # filled blocks
        assert "\u2591" in msg  # empty blocks

    def test_summary_section(self):
        results = [
            _make_result("A", total_trades=5, net_pnl_pct=2.0, total_commission_pct=0.3),
            _make_result("B", total_trades=8, net_pnl_pct=3.0, total_commission_pct=0.4),
        ]
        msg = format_cycle_report(results)
        assert "Trades: 13" in msg
        assert "Fees:" in msg

    def test_respects_telegram_limit(self):
        # Generate many strategies to force truncation
        results = [_make_result(f"Strategy_{i}", win_rate=50.0 + i * 0.5) for i in range(100)]
        msg = format_cycle_report(results)
        assert len(msg) <= TELEGRAM_MAX_LENGTH

    def test_truncation_message_shown(self):
        results = [_make_result(f"Strategy_{i:03d}", win_rate=50.0 + i * 0.1) for i in range(100)]
        msg = format_cycle_report(results)
        assert "truncated" in msg

    def test_custom_cycle_label(self):
        results = [_make_result("EMA")]
        msg = format_cycle_report(results, cycle_label="Phase 2")
        assert "Phase 2 Report" in msg

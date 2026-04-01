"""Tests for orchestrator: composite scoring and best strategy selection."""

import pytest

from backtester.app.core.orchestrator import composite_score, pick_best_strategy
from backtester.app.models.backtest import PhaseResult


def _make_result(
    strategy: str,
    win_rate: float = 50.0,
    net_pnl_pct: float = 0.0,
    max_drawdown_pct: float = 0.0,
    **kwargs,
) -> PhaseResult:
    defaults = dict(
        phase="backtest",
        symbol="EURUSD",
        strategy=strategy,
        total_trades=10,
        wins=5,
        losses=5,
        win_rate=win_rate,
        gross_pnl_pct=net_pnl_pct + 0.5,
        total_commission_pct=0.5,
        net_pnl_pct=net_pnl_pct,
        max_drawdown_pct=max_drawdown_pct,
    )
    defaults.update(kwargs)
    return PhaseResult(**defaults)


class TestCompositeScore:
    def test_score_formula(self):
        # score = (win_rate * 0.4) + (net_pnl_pct * 0.4) - (max_drawdown_pct * 0.2)
        result = _make_result("ema", win_rate=60.0, net_pnl_pct=5.0, max_drawdown_pct=2.0)
        expected = 60.0 * 0.4 + 5.0 * 0.4 - 2.0 * 0.2
        assert composite_score(result) == pytest.approx(expected)

    def test_high_drawdown_penalized(self):
        low_dd = _make_result("a", win_rate=60.0, net_pnl_pct=5.0, max_drawdown_pct=1.0)
        high_dd = _make_result("b", win_rate=60.0, net_pnl_pct=5.0, max_drawdown_pct=20.0)
        assert composite_score(low_dd) > composite_score(high_dd)

    def test_higher_win_rate_scores_better(self):
        low_wr = _make_result("a", win_rate=40.0, net_pnl_pct=5.0)
        high_wr = _make_result("b", win_rate=80.0, net_pnl_pct=5.0)
        assert composite_score(high_wr) > composite_score(low_wr)

    def test_higher_pnl_scores_better(self):
        low_pnl = _make_result("a", win_rate=50.0, net_pnl_pct=1.0)
        high_pnl = _make_result("b", win_rate=50.0, net_pnl_pct=10.0)
        assert composite_score(high_pnl) > composite_score(low_pnl)

    def test_zero_everything_scores_zero(self):
        result = _make_result("flat", win_rate=0.0, net_pnl_pct=0.0, max_drawdown_pct=0.0)
        assert composite_score(result) == 0.0


class TestPickBestStrategy:
    def test_picks_highest_score(self):
        results = [
            _make_result("ema", win_rate=50.0, net_pnl_pct=2.0, max_drawdown_pct=1.0),
            _make_result("rsi", win_rate=70.0, net_pnl_pct=8.0, max_drawdown_pct=1.0),
            _make_result("macd", win_rate=55.0, net_pnl_pct=3.0, max_drawdown_pct=5.0),
        ]
        best = pick_best_strategy(results)
        assert best is not None
        assert best.strategy == "rsi"

    def test_accounts_for_drawdown(self):
        # High PnL but massive drawdown should lose to moderate but stable
        results = [
            _make_result("risky", win_rate=60.0, net_pnl_pct=10.0, max_drawdown_pct=50.0),
            _make_result("stable", win_rate=60.0, net_pnl_pct=8.0, max_drawdown_pct=2.0),
        ]
        best = pick_best_strategy(results)
        assert best.strategy == "stable"

    def test_single_result_returns_it(self):
        result = _make_result("only_one", win_rate=55.0, net_pnl_pct=3.0)
        best = pick_best_strategy([result])
        assert best.strategy == "only_one"

    def test_empty_list_returns_none(self):
        assert pick_best_strategy([]) is None

    def test_tie_returns_one(self):
        a = _make_result("a", win_rate=50.0, net_pnl_pct=5.0, max_drawdown_pct=1.0)
        b = _make_result("b", win_rate=50.0, net_pnl_pct=5.0, max_drawdown_pct=1.0)
        best = pick_best_strategy([a, b])
        assert best is not None
        assert best.strategy in ("a", "b")

"""Tests for backtest results SQLite store and /history endpoint."""

import sqlite3
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backtester.app.data import results_store
from backtester.app.data.results_store import (
    clear_results,
    get_recent_results,
    init_db,
    save_result,
)
from backtester.app.models.backtest import BacktestSummary


@pytest.fixture(autouse=True)
def _use_temp_db(tmp_path):
    """Redirect the store to a temporary database for each test."""
    original_path = results_store.DB_PATH
    results_store.DB_PATH = tmp_path / "test_results.db"
    init_db()
    yield
    results_store.DB_PATH = original_path


def _make_summary(
    strategy: str = "EMA Cross",
    score: float = 25.0,
    win_rate: float = 60.0,
    trades: int = 20,
    symbols: str = "EURUSD,BTCUSD",
) -> BacktestSummary:
    return BacktestSummary(
        strategy_name=strategy,
        composite_score=score,
        backtest_win_rate=win_rate,
        forward_win_rate=None,
        total_trades=trades,
        symbols_tested=symbols,
        ran_at=datetime.now(timezone.utc),
    )


class TestSaveResult:
    def test_returns_row_id(self):
        row_id = save_result(_make_summary())
        assert isinstance(row_id, int)
        assert row_id >= 1

    def test_increments_id(self):
        id1 = save_result(_make_summary("A"))
        id2 = save_result(_make_summary("B"))
        assert id2 == id1 + 1

    def test_persists_all_fields(self):
        summary = BacktestSummary(
            strategy_name="RSI Reversal",
            composite_score=30.5,
            backtest_win_rate=65.0,
            forward_win_rate=58.0,
            total_trades=15,
            symbols_tested="GBPUSD,XAUUSD",
            ran_at=datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        save_result(summary)
        results = get_recent_results(1)
        assert len(results) == 1
        r = results[0]
        assert r.strategy_name == "RSI Reversal"
        assert r.composite_score == 30.5
        assert r.backtest_win_rate == 65.0
        assert r.forward_win_rate == 58.0
        assert r.total_trades == 15
        assert r.symbols_tested == "GBPUSD,XAUUSD"


class TestGetRecentResults:
    def test_empty_db(self):
        assert get_recent_results() == []

    def test_returns_newest_first(self):
        save_result(_make_summary("First", score=10.0))
        save_result(_make_summary("Second", score=20.0))
        save_result(_make_summary("Third", score=30.0))
        results = get_recent_results()
        assert results[0].strategy_name == "Third"
        assert results[-1].strategy_name == "First"

    def test_respects_limit(self):
        for i in range(15):
            save_result(_make_summary(f"Strat_{i}"))
        results = get_recent_results(limit=5)
        assert len(results) == 5

    def test_default_limit_10(self):
        for i in range(20):
            save_result(_make_summary(f"Strat_{i}"))
        results = get_recent_results()
        assert len(results) == 10

    def test_returns_backtest_summary_instances(self):
        save_result(_make_summary())
        results = get_recent_results()
        assert isinstance(results[0], BacktestSummary)
        assert results[0].id is not None


class TestClearResults:
    def test_clears_all(self):
        save_result(_make_summary("A"))
        save_result(_make_summary("B"))
        clear_results()
        assert get_recent_results() == []


class TestHistoryEndpoint:
    @pytest.fixture
    def client(self):
        from backtester import app
        return TestClient(app, raise_server_exceptions=False)

    def test_returns_empty_list(self, client):
        resp = client.get("/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["results"] == []

    def test_returns_saved_results(self, client):
        save_result(_make_summary("EMA Cross", score=25.0))
        save_result(_make_summary("RSI", score=30.0))
        resp = client.get("/history")
        data = resp.json()
        assert data["count"] == 2
        assert data["results"][0]["strategy_name"] == "RSI"
        assert data["results"][1]["strategy_name"] == "EMA Cross"

    def test_limit_param(self, client):
        for i in range(5):
            save_result(_make_summary(f"S{i}"))
        resp = client.get("/history?limit=2")
        assert resp.json()["count"] == 2

    def test_limit_capped_at_100(self, client):
        # Should not error even with a large limit
        resp = client.get("/history?limit=999")
        assert resp.status_code == 200

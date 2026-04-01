"""Tests for /health endpoints on both NovaFX and Backtester."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app.services.bot_state import BotState
from app.models.signals import IncomingSignal
from backtester.app.core.state import BacktesterState


# ── NovaFX /health ────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _reset_states():
    BotState.reset()
    BacktesterState.reset()
    yield
    BotState.reset()
    BacktesterState.reset()


@pytest.fixture
def novafx_client():
    from app.main import app
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def backtester_client():
    from backtester import app
    return TestClient(app, raise_server_exceptions=False)


class TestNovaFXHealth:
    def test_503_when_scheduler_not_running(self, novafx_client):
        # Default state: scheduler is None -> not running
        resp = novafx_client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert "scheduler not running" in data["reason"]
        assert data["service"] == "NovaFX Signal Bot"

    def test_200_when_scheduler_running(self, novafx_client):
        state = BotState.get()
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        mock_job = MagicMock()
        mock_job.next_run_time = datetime(2026, 4, 1, 15, 0, 0, tzinfo=timezone.utc)
        mock_scheduler.get_job.return_value = mock_job
        state.scheduler = mock_scheduler

        resp = novafx_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "NovaFX Signal Bot"
        assert isinstance(data["uptime_seconds"], (int, float))
        assert data["uptime_seconds"] >= 0
        assert data["scheduler_next_run"] is not None

    def test_includes_active_strategy(self, novafx_client):
        state = BotState.get()
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        mock_scheduler.get_job.return_value = None
        state.scheduler = mock_scheduler

        sig = IncomingSignal(symbol="EURUSD", action="BUY", price=1.1, indicator="EMA 9/21")
        state.record_signal(sig)

        resp = novafx_client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["active_strategy"] == "EMA 9/21"

    def test_includes_fetch_times(self, novafx_client):
        state = BotState.get()
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        mock_scheduler.get_job.return_value = None
        state.scheduler = mock_scheduler

        now = datetime.now(timezone.utc)
        state.last_fetch_times["EURUSD"] = now
        state.last_fetch_times["BTCUSD"] = now

        resp = novafx_client.get("/health")
        data = resp.json()
        assert "EURUSD" in data["last_fetch_per_symbol"]
        assert "BTCUSD" in data["last_fetch_per_symbol"]

    def test_no_active_strategy_initially(self, novafx_client):
        state = BotState.get()
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        mock_scheduler.get_job.return_value = None
        state.scheduler = mock_scheduler

        resp = novafx_client.get("/health")
        assert resp.json()["active_strategy"] is None


# ── Backtester /health ────────────────────────────────────────────────────────


class TestBacktesterHealth:
    def test_503_when_scheduler_not_running(self, backtester_client):
        resp = backtester_client.get("/health")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["service"] == "NovaFX Backtester"

    def test_200_when_scheduler_running(self, backtester_client):
        state = BacktesterState.get()
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        mock_job = MagicMock()
        mock_job.next_run_time = datetime(2026, 4, 1, 16, 0, 0, tzinfo=timezone.utc)
        mock_scheduler.get_job.return_value = mock_job
        state.scheduler = mock_scheduler

        resp = backtester_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "NovaFX Backtester"
        assert data["uptime_seconds"] >= 0
        assert data["scheduler_next_run"] is not None

    def test_includes_last_cycle_time(self, backtester_client):
        state = BacktesterState.get()
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        mock_scheduler.get_job.return_value = None
        state.scheduler = mock_scheduler
        state.record_cycle()

        resp = backtester_client.get("/health")
        data = resp.json()
        assert data["last_backtest_cycle"] is not None

    def test_no_cycle_initially(self, backtester_client):
        state = BacktesterState.get()
        mock_scheduler = MagicMock()
        mock_scheduler.running = True
        mock_scheduler.get_job.return_value = None
        state.scheduler = mock_scheduler

        resp = backtester_client.get("/health")
        assert resp.json()["last_backtest_cycle"] is None


# ── BotState extras ───────────────────────────────────────────────────────────


class TestBotStateExtras:
    def test_record_fetch(self):
        state = BotState.get()
        state.record_fetch("EURUSD")
        assert "EURUSD" in state.last_fetch_times
        assert isinstance(state.last_fetch_times["EURUSD"], datetime)

    def test_uptime_positive(self):
        assert BotState.uptime_seconds() >= 0

    def test_scheduler_running_false_when_none(self):
        state = BotState.get()
        assert state.scheduler_running() is False

    def test_scheduler_running_true(self):
        state = BotState.get()
        mock = MagicMock()
        mock.running = True
        state.scheduler = mock
        assert state.scheduler_running() is True


class TestBacktesterStateExtras:
    def test_record_cycle(self):
        state = BacktesterState.get()
        state.record_cycle()
        assert state.last_cycle_time is not None

    def test_uptime_positive(self):
        assert BacktesterState.uptime_seconds() >= 0

    def test_scheduler_running_false_when_none(self):
        state = BacktesterState.get()
        assert state.scheduler_running() is False

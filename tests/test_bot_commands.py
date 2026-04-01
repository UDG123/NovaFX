"""Tests for /status command handler and bot state."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.signals import IncomingSignal
from app.services.bot_commands import _format_status_message, poll_telegram_commands
from app.services.bot_state import BotState


@pytest.fixture(autouse=True)
def reset_state():
    """Reset singleton state before each test."""
    BotState.reset()
    yield
    BotState.reset()


class TestBotState:
    def test_singleton(self):
        a = BotState.get()
        b = BotState.get()
        assert a is b

    def test_reset(self):
        a = BotState.get()
        BotState.reset()
        b = BotState.get()
        assert a is not b

    def test_initial_state(self):
        state = BotState.get()
        assert state.last_signal is None
        assert state.last_signal_time is None
        assert state.active_strategy is None
        assert state.scheduler is None

    def test_record_signal(self):
        state = BotState.get()
        sig = IncomingSignal(
            symbol="EURUSD", action="BUY", price=1.1,
            source="signal_engine", indicator="EMA 9/21 Cross",
        )
        state.record_signal(sig)

        assert state.last_signal is sig
        assert state.last_signal_time is not None
        assert state.active_strategy == "EMA 9/21 Cross"

    def test_record_signal_updates_time(self):
        state = BotState.get()
        sig1 = IncomingSignal(symbol="EURUSD", action="BUY", price=1.1, indicator="A")
        state.record_signal(sig1)
        t1 = state.last_signal_time

        sig2 = IncomingSignal(symbol="GBPUSD", action="SELL", price=1.3, indicator="B")
        state.record_signal(sig2)

        assert state.last_signal is sig2
        assert state.active_strategy == "B"
        assert state.last_signal_time >= t1

    def test_get_next_run_time_no_scheduler(self):
        state = BotState.get()
        assert state.get_next_run_time() is None

    def test_get_next_run_time_with_scheduler(self):
        state = BotState.get()
        mock_scheduler = MagicMock()
        mock_job = MagicMock()
        expected_time = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
        mock_job.next_run_time = expected_time
        mock_scheduler.get_job.return_value = mock_job
        state.scheduler = mock_scheduler

        assert state.get_next_run_time() == expected_time
        mock_scheduler.get_job.assert_called_with("signal_engine")

    def test_get_next_run_time_no_job(self):
        state = BotState.get()
        mock_scheduler = MagicMock()
        mock_scheduler.get_job.return_value = None
        state.scheduler = mock_scheduler

        assert state.get_next_run_time() is None


class TestFormatStatusMessage:
    def test_no_signals_yet(self):
        msg = _format_status_message()
        assert "NovaFX Status" in msg
        assert "No signals generated yet" in msg
        assert "Enabled" in msg

    def test_with_signal(self):
        state = BotState.get()
        sig = IncomingSignal(
            symbol="BTCUSD", action="BUY", price=65000.0,
            indicator="Confluence: EMA, RSI",
        )
        state.record_signal(sig)

        msg = _format_status_message()
        assert "BTCUSD" in msg
        assert "65000.0" in msg
        assert "Confluence: EMA, RSI" in msg

    def test_engine_disabled(self):
        with patch("app.services.bot_commands.settings") as mock_settings:
            mock_settings.SIGNAL_ENGINE_ENABLED = False
            mock_settings.SIGNAL_ENGINE_INTERVAL_MINUTES = 15
            msg = _format_status_message()
            assert "\u274c Disabled" in msg

    def test_next_run_shown(self):
        state = BotState.get()
        mock_scheduler = MagicMock()
        mock_job = MagicMock()
        mock_job.next_run_time = datetime(2026, 4, 1, 14, 30, 0, tzinfo=timezone.utc)
        mock_scheduler.get_job.return_value = mock_job
        state.scheduler = mock_scheduler

        msg = _format_status_message()
        assert "2026-04-01 14:30:00 UTC" in msg

    def test_sell_signal_formatting(self):
        state = BotState.get()
        sig = IncomingSignal(symbol="EURUSD", action="SELL", price=1.085, indicator="MACD Cross")
        state.record_signal(sig)

        msg = _format_status_message()
        assert "\U0001f534 SELL" in msg
        assert "EURUSD" in msg


class TestPollTelegramCommands:
    @pytest.mark.asyncio
    async def test_skips_when_no_token(self):
        with patch("app.services.bot_commands.settings") as mock_settings:
            mock_settings.TELEGRAM_BOT_TOKEN = ""
            await poll_telegram_commands()
            # Should return immediately without error

    @pytest.mark.asyncio
    async def test_handles_status_command(self):
        import app.services.bot_commands as mod
        mod._last_update_id = 0

        fake_response = {
            "ok": True,
            "result": [{
                "update_id": 1,
                "message": {
                    "text": "/status",
                    "chat": {"id": 12345},
                },
            }],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp
        mock_client.post.return_value = mock_resp

        with patch("app.services.bot_commands.settings") as mock_settings, \
             patch("httpx.AsyncClient") as MockClient:
            mock_settings.TELEGRAM_BOT_TOKEN = "test-token"
            mock_settings.SIGNAL_ENGINE_ENABLED = True
            mock_settings.SIGNAL_ENGINE_INTERVAL_MINUTES = 15
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await poll_telegram_commands()

        mock_client.post.assert_called_once()
        call_kwargs = mock_client.post.call_args
        assert call_kwargs[1]["json"]["chat_id"] == 12345
        assert "NovaFX Status" in call_kwargs[1]["json"]["text"]

        assert mod._last_update_id == 1

    @pytest.mark.asyncio
    async def test_ignores_non_status_messages(self):
        import app.services.bot_commands as mod
        mod._last_update_id = 0

        fake_response = {
            "ok": True,
            "result": [{
                "update_id": 2,
                "message": {
                    "text": "hello there",
                    "chat": {"id": 12345},
                },
            }],
        }

        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp

        with patch("app.services.bot_commands.settings") as mock_settings, \
             patch("httpx.AsyncClient") as MockClient:
            mock_settings.TELEGRAM_BOT_TOKEN = "test-token"
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await poll_telegram_commands()

        mock_client.post.assert_not_called()
        assert mod._last_update_id == 2

    @pytest.mark.asyncio
    async def test_handles_empty_updates(self):
        fake_response = {"ok": True, "result": []}

        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_response
        mock_resp.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_resp

        with patch("app.services.bot_commands.settings") as mock_settings, \
             patch("httpx.AsyncClient") as MockClient:
            mock_settings.TELEGRAM_BOT_TOKEN = "test-token"
            MockClient.return_value.__aenter__ = AsyncMock(return_value=mock_client)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await poll_telegram_commands()  # should not raise

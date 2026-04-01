"""Singleton state tracker for the bot — stores last signal and scheduler ref."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.models.signals import IncomingSignal


class BotState:
    _instance: Optional[BotState] = None

    def __init__(self) -> None:
        self.last_signal: Optional[IncomingSignal] = None
        self.last_signal_time: Optional[datetime] = None
        self.active_strategy: Optional[str] = None
        self.scheduler: Optional[AsyncIOScheduler] = None

    @classmethod
    def get(cls) -> BotState:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def record_signal(self, signal: IncomingSignal) -> None:
        self.last_signal = signal
        self.last_signal_time = datetime.now(timezone.utc)
        if signal.indicator:
            self.active_strategy = signal.indicator

    def get_next_run_time(self) -> Optional[datetime]:
        if self.scheduler is None:
            return None
        job = self.scheduler.get_job("signal_engine")
        if job is None:
            return None
        return job.next_run_time

"""Singleton state tracker for the bot — stores last signal and scheduler ref."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.models.signals import IncomingSignal

_start_time = time.monotonic()


class BotState:
    _instance: Optional[BotState] = None

    def __init__(self) -> None:
        self.last_signal: Optional[IncomingSignal] = None
        self.last_signal_time: Optional[datetime] = None
        self.active_strategy: Optional[str] = None
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.last_fetch_times: dict[str, datetime] = {}
        self.signals_sent: int = 0

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
        self.signals_sent += 1
        if signal.indicator:
            self.active_strategy = signal.indicator

    def record_fetch(self, symbol: str) -> None:
        self.last_fetch_times[symbol] = datetime.now(timezone.utc)

    def get_next_run_time(self) -> Optional[datetime]:
        if self.scheduler is None:
            return None
        job = self.scheduler.get_job("signal_engine")
        if job is None:
            return None
        return job.next_run_time

    @staticmethod
    def uptime_seconds() -> float:
        return round(time.monotonic() - _start_time, 1)

    def scheduler_running(self) -> bool:
        return self.scheduler is not None and self.scheduler.running

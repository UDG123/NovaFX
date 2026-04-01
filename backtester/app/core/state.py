"""Singleton state tracker for the backtester service."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler

_start_time = time.monotonic()


class BacktesterState:
    _instance: Optional[BacktesterState] = None

    def __init__(self) -> None:
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.last_cycle_time: Optional[datetime] = None

    @classmethod
    def get(cls) -> BacktesterState:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    def record_cycle(self) -> None:
        self.last_cycle_time = datetime.now(timezone.utc)

    def get_next_run_time(self) -> Optional[datetime]:
        if self.scheduler is None:
            return None
        job = self.scheduler.get_job("backtest_cycle")
        if job is None:
            return None
        return job.next_run_time

    @staticmethod
    def uptime_seconds() -> float:
        return round(time.monotonic() - _start_time, 1)

    def scheduler_running(self) -> bool:
        return self.scheduler is not None and self.scheduler.running

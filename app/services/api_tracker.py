"""
TwelveData API credit tracker.
Tracks daily usage and warns when approaching the Grow55 limit (800 credits/day).
"""
import logging
from datetime import datetime, timezone
from threading import Lock

logger = logging.getLogger(__name__)

DAILY_LIMIT = 800
WARN_THRESHOLD = 0.80  # warn at 80%

class APITracker:
    _instance = None
    _lock = Lock()

    def __init__(self):
        self._calls: int = 0
        self._day: str = self._today()

    @classmethod
    def get(cls) -> "APITracker":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _reset_if_new_day(self):
        today = self._today()
        if today != self._day:
            logger.info(
                "API tracker: new day %s — resetting counter (was %d calls yesterday)",
                today, self._calls,
            )
            self._calls = 0
            self._day = today

    def record_call(self, count: int = 1):
        with self._lock:
            self._reset_if_new_day()
            self._calls += count
            used_pct = self._calls / DAILY_LIMIT * 100

            if self._calls >= DAILY_LIMIT:
                logger.error(
                    "TWELVEDATA LIMIT REACHED: %d/%d calls today (%.0f%%) — "
                    "further calls may fail",
                    self._calls, DAILY_LIMIT, used_pct,
                )
            elif used_pct >= WARN_THRESHOLD * 100:
                logger.warning(
                    "TwelveData usage: %d/%d calls today (%.0f%%) — "
                    "approaching daily limit",
                    self._calls, DAILY_LIMIT, used_pct,
                )
            else:
                logger.debug(
                    "TwelveData usage: %d/%d calls today (%.0f%%)",
                    self._calls, DAILY_LIMIT, used_pct,
                )

    @property
    def calls_today(self) -> int:
        with self._lock:
            self._reset_if_new_day()
            return self._calls

    @property
    def remaining(self) -> int:
        return max(0, DAILY_LIMIT - self.calls_today)

    @property
    def percent_used(self) -> float:
        return round(self.calls_today / DAILY_LIMIT * 100, 1)

    def status(self) -> dict:
        return {
            "calls_today": self.calls_today,
            "remaining": self.remaining,
            "limit": DAILY_LIMIT,
            "percent_used": self.percent_used,
            "date": self._day,
        }

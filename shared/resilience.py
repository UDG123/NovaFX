"""
NovaFX Shared Resilience Module.

Provides DataSource ABC, DataSourceManager with automatic failover,
circuit breakers, retry with jitter, two-layer caching (in-memory + Redis),
health scoring, and confidence tagging.

All signal generator services import this module.
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx
import redis.asyncio as redis
from cachetools import TTLCache
from circuitbreaker import CircuitBreaker, CircuitBreakerError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from shared.config import shared_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AllSourcesFailedError(Exception):
    """Raised when all data sources and cache are exhausted."""

    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        super().__init__(
            f"All data sources failed for {symbol} ({timeframe}) and no cache available"
        )


# ---------------------------------------------------------------------------
# Retry decorator (applied to fetch_candles implementations)
# ---------------------------------------------------------------------------


def with_retry(func):
    """Apply tenacity retry with exponential backoff + jitter."""
    return retry(
        wait=wait_exponential_jitter(
            initial=shared_settings.RETRY_INITIAL_WAIT,
            max=shared_settings.RETRY_MAX_WAIT,
            jitter=shared_settings.RETRY_JITTER,
        ),
        stop=stop_after_attempt(shared_settings.RETRY_MAX_ATTEMPTS),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.HTTPStatusError)
        ),
        reraise=True,
    )(func)


# ---------------------------------------------------------------------------
# DataSource ABC
# ---------------------------------------------------------------------------


class DataSource(ABC):
    """Abstract base class for all OHLCV data sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable source name (e.g. 'TwelveData', 'Binance')."""
        ...

    @property
    @abstractmethod
    def asset_class(self) -> str:
        """Asset class this source covers (e.g. 'crypto', 'forex')."""
        ...

    @abstractmethod
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[dict]:
        """
        Fetch OHLCV candles.

        Returns a list of dicts, each with keys:
            open, high, low, close, volume, timestamp
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the source is reachable and functional."""
        ...


# ---------------------------------------------------------------------------
# Health tracker per source
# ---------------------------------------------------------------------------


@dataclass
class SourceHealth:
    """Sliding-window health tracker for a single DataSource."""

    window_size: int = shared_settings.HEALTH_WINDOW_SIZE
    results: deque = field(default_factory=lambda: deque(maxlen=shared_settings.HEALTH_WINDOW_SIZE))
    latencies: deque = field(default_factory=lambda: deque(maxlen=shared_settings.HEALTH_WINDOW_SIZE))
    circuit_open: bool = False

    def record_success(self, latency_ms: float) -> None:
        self.results.append(True)
        self.latencies.append(latency_ms)

    def record_failure(self) -> None:
        self.results.append(False)

    @property
    def error_rate(self) -> float:
        if not self.results:
            return 0.0
        return 1.0 - (sum(self.results) / len(self.results))

    @property
    def avg_latency_ms(self) -> float:
        if not self.latencies:
            return 1000.0  # default high latency for unknown sources
        return sum(self.latencies) / len(self.latencies)

    @property
    def score(self) -> float:
        """Higher is better. (1 - error_rate) * (1 / avg_latency_ms)."""
        if self.circuit_open:
            return 0.0
        reliability = 1.0 - self.error_rate
        speed = 1.0 / max(self.avg_latency_ms, 1.0)
        return reliability * speed


# ---------------------------------------------------------------------------
# Two-layer cache
# ---------------------------------------------------------------------------


class CandleCache:
    """L1 in-memory + L2 Redis candle cache."""

    def __init__(self) -> None:
        self._l1: TTLCache = TTLCache(
            maxsize=shared_settings.L1_CACHE_MAXSIZE,
            ttl=shared_settings.L1_CACHE_TTL,
        )
        self._redis: Optional[redis.Redis] = None
        self._redis_available = False

    async def _get_redis(self) -> Optional[redis.Redis]:
        if self._redis is not None:
            return self._redis
        if not shared_settings.REDIS_URL:
            return None
        try:
            self._redis = redis.from_url(
                shared_settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=3,
            )
            await self._redis.ping()
            self._redis_available = True
            logger.info("Redis connected for candle cache")
            return self._redis
        except Exception:
            logger.warning("Redis unavailable — L2 cache disabled")
            self._redis_available = False
            return None

    @staticmethod
    def _key(asset_class: str, symbol: str, timeframe: str) -> str:
        return f"cache:{asset_class}:{symbol}:{timeframe}"

    async def get(
        self, asset_class: str, symbol: str, timeframe: str
    ) -> Optional[dict]:
        """
        Try L1 then L2. Returns dict with 'candles' and 'cached_at' or None.
        """
        key = self._key(asset_class, symbol, timeframe)

        # L1
        entry = self._l1.get(key)
        if entry is not None:
            return entry

        # L2
        r = await self._get_redis()
        if r:
            try:
                raw = await r.get(key)
                if raw:
                    entry = json.loads(raw)
                    # Promote to L1
                    self._l1[key] = entry
                    return entry
            except Exception:
                logger.debug("Redis read error for %s", key)

        return None

    async def set(
        self,
        asset_class: str,
        symbol: str,
        timeframe: str,
        candles: list[dict],
    ) -> None:
        """Write to both L1 and L2."""
        key = self._key(asset_class, symbol, timeframe)
        entry = {
            "candles": candles,
            "cached_at": time.time(),
        }

        # L1
        self._l1[key] = entry

        # L2
        r = await self._get_redis()
        if r:
            try:
                await r.set(
                    key,
                    json.dumps(entry),
                    ex=shared_settings.L2_CACHE_TTL,
                )
            except Exception:
                logger.debug("Redis write error for %s", key)

    def is_stale(self, entry: dict) -> bool:
        """Check if cached data is older than the stale threshold."""
        cached_at = entry.get("cached_at", 0)
        age = time.time() - cached_at
        return age > shared_settings.STALE_THRESHOLD_SECONDS


# ---------------------------------------------------------------------------
# DataSourceManager
# ---------------------------------------------------------------------------


class DataSourceManager:
    """
    Manages multiple DataSource instances with failover, caching,
    circuit breakers, and health scoring.
    """

    def __init__(self, sources: list[DataSource]) -> None:
        if not sources:
            raise ValueError("At least one DataSource is required")

        self._sources = list(sources)
        self._health: dict[str, SourceHealth] = {
            s.name: SourceHealth() for s in sources
        }
        self._breakers: dict[str, CircuitBreaker] = {
            s.name: CircuitBreaker(
                failure_threshold=shared_settings.CB_FAILURE_THRESHOLD,
                recovery_timeout=shared_settings.CB_RECOVERY_TIMEOUT,
                expected_exception=Exception,
                name=f"cb_{s.name}",
            )
            for s in sources
        }
        self._cache = CandleCache()
        self._ranked_sources: list[DataSource] = list(sources)
        self._last_rerank = 0.0
        self._health_task: Optional[asyncio.Task] = None

    # -- Public API ---------------------------------------------------------

    async def get_candles(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> dict:
        """
        Fetch candles with automatic failover.

        Returns:
            {
                "candles": list[dict],
                "source": str,
                "confidence": "HIGH" | "MEDIUM" | "LOW",
                "stale": bool,
            }
        """
        self._maybe_rerank()
        asset_class = self._ranked_sources[0].asset_class if self._ranked_sources else "unknown"

        for idx, source in enumerate(self._ranked_sources):
            health = self._health[source.name]
            breaker = self._breakers[source.name]

            # Skip if circuit is open
            if breaker.state == "open":
                health.circuit_open = True
                continue
            health.circuit_open = False

            try:
                t0 = time.monotonic()
                candles = await self._call_with_breaker(
                    breaker, source, symbol, timeframe, limit
                )
                latency_ms = (time.monotonic() - t0) * 1000
                health.record_success(latency_ms)

                if candles:
                    await self._cache.set(asset_class, symbol, timeframe, candles)

                    confidence = self._confidence_for_rank(idx)
                    logger.info(
                        "Fetched %d candles for %s from %s [%s]",
                        len(candles), symbol, source.name, confidence,
                    )
                    return {
                        "candles": candles,
                        "source": source.name,
                        "confidence": confidence,
                        "stale": False,
                    }

            except CircuitBreakerError:
                health.circuit_open = True
                logger.warning(
                    "Circuit open for %s — skipping", source.name
                )
                continue

            except Exception as exc:
                health.record_failure()
                logger.warning(
                    "Source %s failed for %s: %s", source.name, symbol, exc
                )
                continue

        # All sources failed — try cache
        cached = await self._cache.get(asset_class, symbol, timeframe)
        if cached:
            stale = self._cache.is_stale(cached)
            logger.warning(
                "All sources failed for %s — returning %s cache",
                symbol, "stale" if stale else "fresh",
            )
            return {
                "candles": cached["candles"],
                "source": "cache",
                "confidence": "LOW",
                "stale": stale,
            }

        raise AllSourcesFailedError(symbol, timeframe)

    async def start_health_checks(self) -> None:
        """Start background health check task."""
        if self._health_task is None or self._health_task.done():
            self._health_task = asyncio.create_task(self._health_check_loop())
            logger.info("Health check loop started")

    async def stop_health_checks(self) -> None:
        """Cancel the background health check task."""
        if self._health_task and not self._health_task.done():
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            logger.info("Health check loop stopped")

    def get_source_stats(self) -> list[dict]:
        """Return health stats for all sources."""
        return [
            {
                "name": s.name,
                "score": round(self._health[s.name].score, 6),
                "error_rate": round(self._health[s.name].error_rate, 3),
                "avg_latency_ms": round(self._health[s.name].avg_latency_ms, 1),
                "circuit_state": self._breakers[s.name].current_state,
            }
            for s in self._sources
        ]

    # -- Internal -----------------------------------------------------------

    async def _call_with_breaker(
        self,
        breaker: CircuitBreaker,
        source: DataSource,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> list[dict]:
        """Call fetch_candles through the circuit breaker."""

        @breaker
        async def _guarded():
            return await source.fetch_candles(symbol, timeframe, limit)

        return await _guarded()

    def _confidence_for_rank(self, rank: int) -> str:
        if rank == 0:
            return "HIGH"
        elif rank == 1:
            return "MEDIUM"
        return "LOW"

    def _maybe_rerank(self) -> None:
        """Re-rank sources by health score every RERANK_INTERVAL seconds."""
        now = time.monotonic()
        if now - self._last_rerank < shared_settings.RERANK_INTERVAL:
            return
        self._last_rerank = now

        self._ranked_sources = sorted(
            self._sources,
            key=lambda s: self._health[s.name].score,
            reverse=True,
        )

        if logger.isEnabledFor(logging.DEBUG):
            rankings = ", ".join(
                f"{s.name}={self._health[s.name].score:.4f}"
                for s in self._ranked_sources
            )
            logger.debug("Source ranking: %s", rankings)

    async def _health_check_loop(self) -> None:
        """Background loop that pings each source every HEALTH_CHECK_INTERVAL."""
        while True:
            try:
                await asyncio.sleep(shared_settings.HEALTH_CHECK_INTERVAL)
                for source in self._sources:
                    try:
                        t0 = time.monotonic()
                        healthy = await source.health_check()
                        latency_ms = (time.monotonic() - t0) * 1000
                        if healthy:
                            self._health[source.name].record_success(latency_ms)
                        else:
                            self._health[source.name].record_failure()
                    except Exception:
                        self._health[source.name].record_failure()
                        logger.debug(
                            "Health check failed for %s", source.name
                        )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Health check loop error")

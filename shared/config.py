"""Shared configuration for all NovaFX services."""

import os
from typing import Optional

from pydantic_settings import BaseSettings


class SharedSettings(BaseSettings):
    REDIS_URL: str = "redis://localhost:6379/0"
    DISPATCHER_URL: str = "http://localhost:8000"
    LOG_LEVEL: str = "INFO"

    # Circuit breaker
    CB_FAILURE_THRESHOLD: int = 5
    CB_RECOVERY_TIMEOUT: int = 60

    # Cache TTLs
    L1_CACHE_TTL: int = 3600       # 1 hour
    L1_CACHE_MAXSIZE: int = 512
    L2_CACHE_TTL: int = 7200       # 2 hours
    STALE_THRESHOLD_SECONDS: int = 900  # 15 minutes

    # Health check
    HEALTH_CHECK_INTERVAL: int = 30    # seconds
    HEALTH_WINDOW_SIZE: int = 20       # sliding window for scoring
    RERANK_INTERVAL: int = 60          # seconds

    # Retry
    RETRY_INITIAL_WAIT: int = 1
    RETRY_MAX_WAIT: int = 30
    RETRY_JITTER: int = 2
    RETRY_MAX_ATTEMPTS: int = 3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


shared_settings = SharedSettings()

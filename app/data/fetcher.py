import asyncio
import logging
import sqlite3
import time
from pathlib import Path

import httpx
import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

CACHE_DB_PATH = Path(__file__).parent / "ohlcv_cache.db"
CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2  # 2s, 4s, 8s

TWELVEDATA_BASE_URL = "https://api.twelvedata.com/time_series"

# --- Symbol mapping -----------------------------------------------------------
# NovaFX uses concatenated symbols (EURUSD); TwelveData expects slash-separated
# forex (EUR/USD), crypto (BTC/USD), or plain index names.

FOREX_BASES = {"EUR", "GBP", "USD", "JPY", "AUD", "CAD", "NZD", "CHF"}
CRYPTO_BASES = {"BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "DOT"}

INDEX_MAP = {
    "SPX500": "SPX",
    "NAS100": "IXIC",
    "US30": "DJI",
    "DE40": "GDAXI",
    "UK100": "FTSE",
    "JP225": "N225",
}

COMMODITY_MAP = {
    "XAUUSD": "XAU/USD",
    "XAGUSD": "XAG/USD",
    "USOIL": "WTI/USD",
    "UKOIL": "BRENT/USD",
}


def map_symbol(symbol: str) -> str:
    """Convert a NovaFX symbol to TwelveData format."""
    upper = symbol.upper().replace("/", "").replace("-", "")

    if upper in INDEX_MAP:
        return INDEX_MAP[upper]
    if upper in COMMODITY_MAP:
        return COMMODITY_MAP[upper]

    # Crypto: first 3-4 chars are base, rest is quote (BTC+USD, ETH+USDT)
    for base in sorted(CRYPTO_BASES, key=len, reverse=True):
        if upper.startswith(base) and len(upper) > len(base):
            quote = upper[len(base):]
            return f"{base}/{quote}"

    # Forex: 6-char pairs like EURUSD -> EUR/USD
    if len(upper) == 6:
        base, quote = upper[:3], upper[3:]
        if base in FOREX_BASES and quote in FOREX_BASES:
            return f"{base}/{quote}"

    return symbol


# --- Token-bucket rate limiter (60 requests / minute) -------------------------

class TokenBucket:
    """Async token-bucket rate limiter."""

    def __init__(self, capacity: int = 60, refill_seconds: float = 60.0):
        self._capacity = capacity
        self._tokens = float(capacity)
        self._refill_rate = capacity / refill_seconds  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            self._refill()
            while self._tokens < 1.0:
                wait = (1.0 - self._tokens) / self._refill_rate
                await asyncio.sleep(wait)
                self._refill()
            self._tokens -= 1.0

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now


_rate_limiter = TokenBucket(capacity=60, refill_seconds=60.0)

# --- TwelveData interval mapping -----------------------------------------------

INTERVAL_MAP = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day",
}

# --- SQLite cache --------------------------------------------------------------


def _init_cache_db() -> None:
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv_cache (
            cache_key TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            fetched_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


_init_cache_db()


def _cache_key(symbol: str, timeframe: str, limit: int) -> str:
    return f"{symbol}|{timeframe}|{limit}"


def _get_cached(key: str) -> pd.DataFrame | None:
    conn = sqlite3.connect(CACHE_DB_PATH)
    row = conn.execute(
        "SELECT data, fetched_at FROM ohlcv_cache WHERE cache_key = ?", (key,)
    ).fetchone()
    conn.close()

    if row is None:
        return None

    data_json, fetched_at = row
    if time.time() - fetched_at > CACHE_TTL_SECONDS:
        return None

    df = pd.read_json(data_json, orient="split")
    logger.info("Cache hit for %s", key)
    return df


def _set_cached(key: str, df: pd.DataFrame) -> None:
    data_json = df.to_json(orient="split")
    conn = sqlite3.connect(CACHE_DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO ohlcv_cache (cache_key, data, fetched_at) VALUES (?, ?, ?)",
        (key, data_json, time.time()),
    )
    conn.commit()
    conn.close()


# --- TwelveData fetch ----------------------------------------------------------


async def _fetch_twelvedata(
    symbol: str, timeframe: str, limit: int, client: httpx.AsyncClient
) -> pd.DataFrame | None:
    td_symbol = map_symbol(symbol)
    td_interval = INTERVAL_MAP.get(timeframe, "15min")

    params = {
        "symbol": td_symbol,
        "interval": td_interval,
        "outputsize": str(limit),
        "apikey": settings.TWELVEDATA_API_KEY,
        "format": "JSON",
    }

    await _rate_limiter.acquire()

    resp = await client.get(TWELVEDATA_BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    if "values" not in data:
        error_msg = data.get("message", data.get("status", "unknown error"))
        logger.warning("TwelveData error for %s (%s): %s", symbol, td_symbol, error_msg)
        return None

    rows = data["values"]
    df = pd.DataFrame(rows)
    df = df.rename(columns={"datetime": "timestamp"})
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # TwelveData returns newest first; reverse to chronological order
    df = df.iloc[::-1].reset_index(drop=True)
    df = df[["open", "high", "low", "close", "volume"]]
    return df


# --- Public API ----------------------------------------------------------------


async def fetch_ohlcv(
    symbol: str, timeframe: str = "15m", limit: int = 100
) -> pd.DataFrame | None:
    """Fetch OHLCV from TwelveData with SQLite cache (1h TTL) and retry."""
    if not settings.TWELVEDATA_API_KEY:
        logger.warning("TWELVEDATA_API_KEY not set - cannot fetch data for %s", symbol)
        return None

    key = _cache_key(symbol, timeframe, limit)

    cached = _get_cached(key)
    if cached is not None:
        return cached

    last_exc = None
    async with httpx.AsyncClient() as client:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = await _fetch_twelvedata(symbol, timeframe, limit, client)
                if df is not None and not df.empty:
                    _set_cached(key, df)
                    logger.info("Fetched %d bars for %s (attempt %d)", len(df), symbol, attempt)
                    return df
                logger.warning("Empty response for %s (attempt %d/%d)", symbol, attempt, MAX_RETRIES)
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code == 429:
                    logger.warning("Rate limited on %s, backing off (attempt %d/%d)", symbol, attempt, MAX_RETRIES)
                else:
                    logger.warning("HTTP %d for %s (attempt %d/%d)", exc.response.status_code, symbol, attempt, MAX_RETRIES)
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning("fetch_ohlcv %s failed (attempt %d/%d): %s", symbol, attempt, MAX_RETRIES, exc)

            if attempt < MAX_RETRIES:
                delay = BACKOFF_BASE_SECONDS ** attempt
                logger.info("Retrying %s in %ds...", symbol, delay)
                await asyncio.sleep(delay)

    if last_exc:
        logger.error("All %d fetch attempts failed for %s: %s", MAX_RETRIES, symbol, last_exc)
    return None

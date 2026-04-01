import asyncio
import json
import logging
import sqlite3
import time
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_DB_PATH = Path(__file__).parent / "ohlcv_cache.db"
CACHE_TTL_SECONDS = 3600  # 1 hour
MAX_RETRIES = 3
BACKOFF_BASE_SECONDS = 2  # 2s, 4s, 8s


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


def _fetch_yfinance(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch OHLCV data from yfinance.

    Maps common forex/crypto symbols to yfinance tickers (e.g. EURUSD -> EURUSD=X).
    """
    ticker_symbol = symbol.upper()

    # yfinance forex pairs need =X suffix
    if len(ticker_symbol) == 6 and not any(
        c.isdigit() for c in ticker_symbol
    ):
        ticker_symbol = f"{ticker_symbol}=X"

    # Map timeframe strings to yfinance intervals and periods
    interval_map = {
        "1m": ("1m", "1d"),
        "5m": ("5m", "5d"),
        "15m": ("15m", "5d"),
        "30m": ("30m", "5d"),
        "1h": ("1h", "30d"),
        "4h": ("1h", "60d"),  # yfinance doesn't have 4h; fetch 1h
        "1d": ("1d", "180d"),
    }
    interval, period = interval_map.get(timeframe, ("15m", "5d"))

    ticker = yf.Ticker(ticker_symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        return df

    df.columns = [c.lower() for c in df.columns]
    df = df[["open", "high", "low", "close", "volume"]].tail(limit)
    df = df.reset_index(drop=True)
    return df


async def fetch_ohlcv(
    symbol: str, timeframe: str = "15m", limit: int = 100
) -> pd.DataFrame | None:
    """Fetch OHLCV with SQLite caching (1h TTL) and exponential backoff retries."""
    key = _cache_key(symbol, timeframe, limit)

    cached = _get_cached(key)
    if cached is not None:
        return cached

    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            df = await asyncio.to_thread(_fetch_yfinance, symbol, timeframe, limit)
            if df is not None and not df.empty:
                _set_cached(key, df)
                logger.info("Fetched %d bars for %s (attempt %d)", len(df), symbol, attempt)
                return df
            logger.warning("Empty response for %s (attempt %d/%d)", symbol, attempt, MAX_RETRIES)
            last_exc = None
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "fetch_ohlcv %s failed (attempt %d/%d): %s",
                symbol, attempt, MAX_RETRIES, exc,
            )

        if attempt < MAX_RETRIES:
            delay = BACKOFF_BASE_SECONDS ** attempt  # 2s, 4s, 8s
            logger.info("Retrying %s in %ds...", symbol, delay)
            await asyncio.sleep(delay)

    if last_exc:
        logger.error("All %d fetch attempts failed for %s: %s", MAX_RETRIES, symbol, last_exc)
    return None

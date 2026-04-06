"""
NovaFX Stock DataSource implementations.

Four-tier failover:
  1. Alpaca (primary, paper trading — free, real-time IEX)
  2. TwelveData (Grow 55 plan, shared with forex)
  3. Finnhub (free, 60 calls/min)
  4. Alpha Vantage (free, 25 calls/day — last resort)
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import httpx

from shared.resilience import DataSource, with_retry

logger = logging.getLogger("novafx.stocks.sources")

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")


# ---------------------------------------------------------------------------
# 1. Alpaca — primary (free paper trading, real-time IEX data)
# ---------------------------------------------------------------------------


class AlpacaStockSource(DataSource):
    """Alpaca Markets — primary stock data source."""

    DATA_URL = "https://data.alpaca.markets/v2"
    PAPER_URL = "https://paper-api.alpaca.markets/v2"

    @property
    def name(self) -> str:
        return "alpaca-stocks"

    @property
    def asset_class(self) -> str:
        return "stocks"

    def _headers(self) -> dict:
        return {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }

    @with_retry
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[dict]:
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError("Alpaca API credentials not set")

        now = datetime.now(timezone.utc)
        start = (now - timedelta(days=7)).isoformat()
        end = now.isoformat()

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{self.DATA_URL}/stocks/{symbol}/bars",
                headers=self._headers(),
                params={
                    "timeframe": "1Hour",
                    "start": start,
                    "end": end,
                    "limit": limit,
                    "feed": "iex",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        bars = data.get("bars", [])
        if not bars:
            raise ValueError(f"Alpaca: no bars for {symbol}")

        candles = []
        for bar in bars:
            candles.append({
                "timestamp": bar["t"],
                "open": float(bar["o"]),
                "high": float(bar["h"]),
                "low": float(bar["l"]),
                "close": float(bar["c"]),
                "volume": int(bar["v"]),
            })
        return candles

    async def health_check(self) -> bool:
        if not ALPACA_API_KEY:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    f"{self.PAPER_URL}/clock",
                    headers=self._headers(),
                )
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 2. TwelveData — first fallback (shared Grow 55 plan)
# ---------------------------------------------------------------------------


class TwelveDataStockSource(DataSource):
    """TwelveData API — shared with forex service."""

    @property
    def name(self) -> str:
        return "twelvedata-stocks"

    @property
    def asset_class(self) -> str:
        return "stocks"

    @with_retry
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[dict]:
        if not TWELVEDATA_API_KEY:
            raise ValueError("TWELVEDATA_API_KEY not set")

        interval_map = {
            "1h": "1h", "4h": "4h", "1d": "1day",
            "15m": "15min", "5m": "5min", "30m": "30min",
        }
        td_interval = interval_map.get(timeframe, "1h")

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol": symbol,
                    "interval": td_interval,
                    "outputsize": str(limit),
                    "apikey": TWELVEDATA_API_KEY,
                    "format": "JSON",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("status") == "error":
            raise ValueError(f"TwelveData error: {data.get('message', 'unknown')}")
        if "values" not in data:
            raise ValueError(f"TwelveData: no values for {symbol}")

        candles = []
        for row in reversed(data["values"]):
            candles.append({
                "timestamp": row.get("datetime", ""),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })
        return candles

    async def health_check(self) -> bool:
        if not TWELVEDATA_API_KEY:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    "https://api.twelvedata.com/api_usage",
                    params={"apikey": TWELVEDATA_API_KEY},
                )
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 3. Finnhub — second fallback (60 calls/min free)
# ---------------------------------------------------------------------------


class FinnhubStockSource(DataSource):
    """Finnhub API — second fallback for stock data."""

    @property
    def name(self) -> str:
        return "finnhub-stocks"

    @property
    def asset_class(self) -> str:
        return "stocks"

    @with_retry
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[dict]:
        if not FINNHUB_API_KEY:
            raise ValueError("FINNHUB_API_KEY not set")

        resolution_map = {
            "1m": "1", "5m": "5", "15m": "15", "30m": "30",
            "1h": "60", "1d": "D", "1w": "W",
        }
        resolution = resolution_map.get(timeframe, "60")

        now = int(time.time())
        hours_map = {"1h": 1, "4h": 4, "1d": 24, "15m": 0.25}
        hours_per_bar = hours_map.get(timeframe, 1)
        from_ts = now - int(limit * hours_per_bar * 3600)

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://finnhub.io/api/v1/stock/candle",
                params={
                    "symbol": symbol,
                    "resolution": resolution,
                    "from": from_ts,
                    "to": now,
                    "token": FINNHUB_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        status = data.get("s", "")
        if status != "ok":
            raise ValueError(f"Finnhub: status={status} for {symbol}")

        candles = []
        timestamps = data.get("t", [])
        opens = data.get("o", [])
        highs = data.get("h", [])
        lows = data.get("l", [])
        closes = data.get("c", [])
        volumes = data.get("v", [])

        for i in range(len(timestamps)):
            candles.append({
                "timestamp": timestamps[i],
                "open": opens[i],
                "high": highs[i],
                "low": lows[i],
                "close": closes[i],
                "volume": volumes[i] if i < len(volumes) else 0,
            })
        return candles

    async def health_check(self) -> bool:
        if not FINNHUB_API_KEY:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    "https://finnhub.io/api/v1/stock/symbol",
                    params={"exchange": "US", "token": FINNHUB_API_KEY},
                )
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 4. Alpha Vantage — emergency (25 calls/day free)
# ---------------------------------------------------------------------------


class AlphaVantageStockSource(DataSource):
    """Alpha Vantage — absolute last resort for stock data."""

    @property
    def name(self) -> str:
        return "alphavantage-stocks"

    @property
    def asset_class(self) -> str:
        return "stocks"

    @with_retry
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[dict]:
        if not ALPHAVANTAGE_API_KEY:
            raise ValueError("ALPHAVANTAGE_API_KEY not set")

        interval_map = {
            "1m": "1min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1h": "60min",
        }
        av_interval = interval_map.get(timeframe, "60min")

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "TIME_SERIES_INTRADAY",
                    "symbol": symbol,
                    "interval": av_interval,
                    "outputsize": "compact",
                    "apikey": ALPHAVANTAGE_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if "Error Message" in data or "Note" in data:
            msg = data.get("Error Message") or data.get("Note", "rate limited")
            raise ValueError(f"Alpha Vantage: {msg}")

        series_key = f"Time Series ({av_interval})"
        time_series = data.get(series_key, {})
        if not time_series:
            raise ValueError(f"Alpha Vantage: no data for {symbol}")

        candles = []
        for ts, values in sorted(time_series.items()):
            candles.append({
                "timestamp": ts,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": int(values.get("5. volume", 0)),
            })
        return candles[-limit:]

    async def health_check(self) -> bool:
        if not ALPHAVANTAGE_API_KEY:
            return False
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    "https://www.alphavantage.co/query",
                    params={
                        "function": "TIME_SERIES_INTRADAY",
                        "symbol": "AAPL",
                        "interval": "60min",
                        "outputsize": "compact",
                        "apikey": ALPHAVANTAGE_API_KEY,
                    },
                )
                data = resp.json()
                return "Time Series (60min)" in data
        except Exception:
            return False

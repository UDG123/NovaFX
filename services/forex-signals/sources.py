"""
NovaFX Forex DataSource implementations.

Four-tier failover:
  1. TwelveData (primary, Grow 55 plan)
  2. Finnhub (free, 60 calls/min)
  3. Financial Modeling Prep (free, 250 calls/day)
  4. Alpha Vantage (free, 25 calls/day — last resort)
"""

import logging
import os
import time

import httpx

from shared.resilience import DataSource, with_retry

logger = logging.getLogger("novafx.forex.sources")

TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FMP_API_KEY = os.getenv("FMP_API_KEY", "")
ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")


# ---------------------------------------------------------------------------
# 1. TwelveData — primary (55 credits/min, no daily cap)
# ---------------------------------------------------------------------------


class TwelveDataForexSource(DataSource):
    """TwelveData API — primary forex data source."""

    @property
    def name(self) -> str:
        return "twelvedata-forex"

    @property
    def asset_class(self) -> str:
        return "forex"

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
            raise ValueError(f"TwelveData: no values in response for {symbol}")

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
# 2. Finnhub — first fallback (60 calls/min free)
# ---------------------------------------------------------------------------


class FinnhubForexSource(DataSource):
    """Finnhub API — first fallback for forex data."""

    @property
    def name(self) -> str:
        return "finnhub-forex"

    @property
    def asset_class(self) -> str:
        return "forex"

    @staticmethod
    def _to_finnhub_symbol(symbol: str) -> str:
        """Convert EUR/USD -> OANDA:EUR_USD."""
        return f"OANDA:{symbol.replace('/', '_')}"

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

        fh_symbol = self._to_finnhub_symbol(symbol)

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://finnhub.io/api/v1/forex/candles",
                params={
                    "symbol": fh_symbol,
                    "resolution": resolution,
                    "from": from_ts,
                    "to": now,
                    "token": FINNHUB_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        status = data.get("s", "")
        if status != "ok" or status == "no_data":
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
                    "https://finnhub.io/api/v1/forex/symbol",
                    params={"exchange": "oanda", "token": FINNHUB_API_KEY},
                )
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 3. Financial Modeling Prep — second fallback (250 calls/day free)
# ---------------------------------------------------------------------------


class FMPForexSource(DataSource):
    """Financial Modeling Prep — second fallback, limited free tier."""

    @property
    def name(self) -> str:
        return "fmp-forex"

    @property
    def asset_class(self) -> str:
        return "forex"

    @staticmethod
    def _to_fmp_symbol(symbol: str) -> str:
        """Convert EUR/USD -> EURUSD."""
        return symbol.replace("/", "")

    @with_retry
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[dict]:
        if not FMP_API_KEY:
            raise ValueError("FMP_API_KEY not set")

        interval_map = {
            "1h": "1hour", "4h": "4hour", "15m": "15min",
            "5m": "5min", "30m": "30min", "1d": "1day",
        }
        fmp_interval = interval_map.get(timeframe, "1hour")
        fmp_symbol = self._to_fmp_symbol(symbol)

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"https://financialmodelingprep.com/api/v3/historical-chart/{fmp_interval}/{fmp_symbol}",
                params={"apikey": FMP_API_KEY},
            )
            resp.raise_for_status()
            data = resp.json()

        if not isinstance(data, list) or not data:
            raise ValueError(f"FMP: empty response for {symbol}")

        # FMP returns newest first — reverse to chronological
        data = list(reversed(data[:limit]))

        candles = []
        for row in data:
            candles.append({
                "timestamp": row.get("date", ""),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row.get("volume", 0)),
            })
        return candles

    async def health_check(self) -> bool:
        if not FMP_API_KEY:
            return False
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    "https://financialmodelingprep.com/api/v3/is-the-market-open",
                    params={"apikey": FMP_API_KEY},
                )
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 4. Alpha Vantage — emergency only (25 calls/day free)
# ---------------------------------------------------------------------------


class AlphaVantageForexSource(DataSource):
    """Alpha Vantage — absolute last resort, 25 calls/day."""

    @property
    def name(self) -> str:
        return "alphavantage-forex"

    @property
    def asset_class(self) -> str:
        return "forex"

    @with_retry
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[dict]:
        if not ALPHAVANTAGE_API_KEY:
            raise ValueError("ALPHAVANTAGE_API_KEY not set")

        parts = symbol.split("/")
        if len(parts) != 2:
            raise ValueError(f"Invalid forex symbol format: {symbol}")
        from_sym, to_sym = parts

        interval_map = {
            "1m": "1min", "5m": "5min", "15m": "15min",
            "30m": "30min", "1h": "60min",
        }
        av_interval = interval_map.get(timeframe, "60min")

        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "FX_INTRADAY",
                    "from_symbol": from_sym,
                    "to_symbol": to_sym,
                    "interval": av_interval,
                    "outputsize": "compact",
                    "apikey": ALPHAVANTAGE_API_KEY,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        # Check for error messages
        if "Error Message" in data or "Note" in data:
            msg = data.get("Error Message") or data.get("Note", "rate limited")
            raise ValueError(f"Alpha Vantage: {msg}")

        series_key = f"Time Series FX (Intraday)"
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
                "volume": 0,
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
                        "function": "FX_INTRADAY",
                        "from_symbol": "EUR",
                        "to_symbol": "USD",
                        "interval": "60min",
                        "outputsize": "compact",
                        "apikey": ALPHAVANTAGE_API_KEY,
                    },
                )
                data = resp.json()
                return "Time Series FX (Intraday)" in data
        except Exception:
            return False

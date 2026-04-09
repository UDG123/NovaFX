"""
NovaFX Crypto Fallback Scanner.

Activates when Freqtrade's exchange connection fails.
Uses shared DataSourceManager with CCXT, TwelveData, and CryptoCompare
as ranked data sources with automatic failover.
"""

import asyncio
import logging
import os
import sys
import time
from typing import Optional

import httpx
import numpy as np
import redis as redis_client

# ---------------------------------------------------------------------------
# Redis signal deduplication
# ---------------------------------------------------------------------------

REDIS_URL = os.getenv("REDIS_URL")
_redis = redis_client.from_url(REDIS_URL) if REDIS_URL else None
SIGNAL_COOLDOWN_SECONDS = 3600  # 1 hour cooldown per pair+direction


async def write_signal_to_stream(symbol: str, direction: str, price: float, confidence: float) -> None:
    """Write signal to Redis stream for paper trader to pick up."""
    if not _redis:
        return
    try:
        import time
        _redis.xadd("novafx:signals:crypto", {
            "symbol": symbol,
            "direction": direction,
            "entry": str(price),
            "confidence": str(confidence),
            "timestamp": str(time.time()),
        }, maxlen=100)
    except Exception as e:
        logging.getLogger("novafx.crypto.fallback").warning(f"Failed to write to Redis stream: {e}")


def is_duplicate_signal(symbol: str, direction: str) -> bool:
    """Check if signal was sent recently; if not, mark it as sent."""
    if not _redis:
        return False
    key = f"novafx:signal_sent:{symbol}:{direction}"
    if _redis.exists(key):
        return True
    _redis.setex(key, SIGNAL_COOLDOWN_SECONDS, "1")
    return False

sys.path.insert(0, "/freqtrade")

from shared.models import AssetClass, Signal, SignalAction
from shared.resilience import DataSource, DataSourceManager, with_retry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("novafx.crypto.fallback")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DISPATCHER_URL = os.getenv("DISPATCHER_URL", "http://localhost:8000")
TWELVEDATA_API_KEY = os.getenv("TWELVEDATA_API_KEY", "")
CRYPTOCOMPARE_API_KEY = os.getenv("CRYPTOCOMPARE_API_KEY", "")
SCAN_INTERVAL = int(os.getenv("SCAN_INTERVAL", "300"))
FREQTRADE_API = "http://localhost:8080/api/v1/ping"

# Telegram direct send config (fallback when confluence doesn't fire)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "-1003614474777")
TELEGRAM_DIRECT_SEND = os.getenv("TELEGRAM_DIRECT_SEND", "true").lower() == "true"

WATCHLIST = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT",
    "ADA/USDT", "AVAX/USDT", "DOGE/USDT", "LINK/USDT",
]

# TwelveData uses /USD not /USDT
TD_SYMBOL_MAP = {
    "BTC/USDT": "BTC/USD", "ETH/USDT": "ETH/USD",
    "SOL/USDT": "SOL/USD", "XRP/USDT": "XRP/USD",
    "ADA/USDT": "ADA/USD", "AVAX/USDT": "AVAX/USD",
    "DOGE/USDT": "DOGE/USD", "LINK/USDT": "LINK/USD",
}

# CryptoCompare uses base/quote split
CC_SYMBOL_MAP = {
    "BTC/USDT": ("BTC", "USD"), "ETH/USDT": ("ETH", "USD"),
    "SOL/USDT": ("SOL", "USD"), "XRP/USDT": ("XRP", "USD"),
    "ADA/USDT": ("ADA", "USD"), "AVAX/USDT": ("AVAX", "USD"),
    "DOGE/USDT": ("DOGE", "USD"), "LINK/USDT": ("LINK", "USD"),
}


# ---------------------------------------------------------------------------
# DataSource implementations
# ---------------------------------------------------------------------------


class CCXTSource(DataSource):
    """Multi-exchange CCXT source with internal failover and proper session management."""

    EXCHANGES = ["binance", "kraken", "okx", "kucoin", "gateio"]

    def __init__(self) -> None:
        self._exchanges: dict = {}  # Reusable exchange instances

    @property
    def name(self) -> str:
        return "ccxt-multi"

    @property
    def asset_class(self) -> str:
        return "crypto"

    async def _get_exchange(self, exchange_id: str):
        """Get or create a reusable exchange instance."""
        import ccxt.async_support as ccxt

        if exchange_id not in self._exchanges:
            exchange_cls = getattr(ccxt, exchange_id)
            self._exchanges[exchange_id] = exchange_cls({"enableRateLimit": True})
        return self._exchanges[exchange_id]

    async def close(self) -> None:
        """Close all open exchange sessions."""
        for exchange_id, exchange in list(self._exchanges.items()):
            try:
                await exchange.close()
            except Exception as exc:
                logger.debug("Failed to close %s: %s", exchange_id, exc)
        self._exchanges.clear()

    @with_retry
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[dict]:
        for exchange_id in self.EXCHANGES:
            try:
                exchange = await self._get_exchange(exchange_id)
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

                return [
                    {
                        "timestamp": row[0],
                        "open": row[1],
                        "high": row[2],
                        "low": row[3],
                        "close": row[4],
                        "volume": row[5],
                    }
                    for row in ohlcv
                ]

            except Exception as exc:
                logger.debug(
                    "CCXT %s failed for %s: %s", exchange_id, symbol, exc
                )
                # Remove failed exchange from cache to force reconnect
                if exchange_id in self._exchanges:
                    try:
                        await self._exchanges[exchange_id].close()
                    except Exception:
                        pass
                    del self._exchanges[exchange_id]
                continue

        raise ConnectionError(f"All CCXT exchanges failed for {symbol}")

    async def health_check(self) -> bool:
        try:
            exchange = await self._get_exchange("kraken")
            await exchange.fetch_ticker("BTC/USD")
            return True
        except Exception:
            return False


class TwelveDataCryptoSource(DataSource):
    """TwelveData API source for crypto candles."""

    @property
    def name(self) -> str:
        return "twelvedata-crypto"

    @property
    def asset_class(self) -> str:
        return "crypto"

    @with_retry
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[dict]:
        if not TWELVEDATA_API_KEY:
            raise ValueError("TWELVEDATA_API_KEY not set")

        td_symbol = TD_SYMBOL_MAP.get(symbol, symbol)

        interval_map = {
            "1h": "1h", "4h": "4h", "1d": "1day",
            "15m": "15min", "5m": "5min",
        }
        td_interval = interval_map.get(timeframe, "1h")

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol": td_symbol,
                    "interval": td_interval,
                    "outputsize": str(limit),
                    "apikey": TWELVEDATA_API_KEY,
                    "format": "JSON",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        if "values" not in data:
            raise ValueError(f"TwelveData error: {data.get('message', 'unknown')}")

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
                    "https://api.twelvedata.com/time_series",
                    params={
                        "symbol": "BTC/USD",
                        "interval": "1h",
                        "outputsize": "1",
                        "apikey": TWELVEDATA_API_KEY,
                    },
                )
                return resp.status_code == 200
        except Exception:
            return False


class CryptoCompareSource(DataSource):
    """CryptoCompare CCCAGG source — free tier, multi-exchange weighted average."""

    @property
    def name(self) -> str:
        return "cryptocompare"

    @property
    def asset_class(self) -> str:
        return "crypto"

    @with_retry
    async def fetch_candles(
        self, symbol: str, timeframe: str, limit: int
    ) -> list[dict]:
        fsym, tsym = CC_SYMBOL_MAP.get(symbol, ("BTC", "USD"))

        endpoint_map = {
            "1h": "histohour", "4h": "histohour",
            "1d": "histoday", "15m": "histominute",
            "5m": "histominute",
        }
        endpoint = endpoint_map.get(timeframe, "histohour")

        params = {"fsym": fsym, "tsym": tsym, "limit": limit}
        if CRYPTOCOMPARE_API_KEY:
            params["api_key"] = CRYPTOCOMPARE_API_KEY

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"https://min-api.cryptocompare.com/data/v2/{endpoint}",
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        if data.get("Response") != "Success":
            raise ValueError(f"CryptoCompare error: {data.get('Message', 'unknown')}")

        candles = []
        for row in data["Data"]["Data"]:
            candles.append({
                "timestamp": row["time"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
                "volume": row.get("volumeto", 0),
            })
        return candles

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(
                    "https://min-api.cryptocompare.com/data/v2/histohour",
                    params={"fsym": "BTC", "tsym": "USD", "limit": 1},
                )
                return resp.status_code == 200
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------


def _compute_rsi(closes: list[float], period: int = 14) -> float:
    """Compute RSI from close prices."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[-period:])
    avg_loss = np.mean(losses[-period:])
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_ema(values: list[float], period: int) -> float:
    """Compute latest EMA value."""
    if len(values) < period:
        return values[-1] if values else 0.0
    arr = np.array(values, dtype=float)
    multiplier = 2.0 / (period + 1)
    ema = arr[0]
    for val in arr[1:]:
        ema = (val - ema) * multiplier + ema
    return float(ema)


def _compute_atr(candles: list[dict], period: int = 14) -> float | None:
    """Compute ATR from candle dicts."""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return None
    return float(np.mean(trs[-period:]))


def analyze_candles(symbol: str, candles: list[dict]) -> Optional[Signal]:
    """Run RSI-only strategy on candle data (no EMA crossover required)."""
    if len(candles) < 30:
        return None

    closes = [c["close"] for c in candles]
    rsi = _compute_rsi(closes)
    ema_fast = _compute_ema(closes, 12)
    ema_slow = _compute_ema(closes, 26)
    price = closes[-1]

    if any(np.isnan(v) for v in [rsi, ema_fast, ema_slow]):
        return None

    atr = _compute_atr(candles)
    metadata = {"rsi": round(rsi, 2), "ema_fast": round(ema_fast, 4), "ema_slow": round(ema_slow, 4)}

    # Buy signal: RSI < 40 (RSI-only, no EMA crossover required)
    if rsi < 40:
        confidence = min(0.4 + (40 - rsi) / 50, 1.0)
        sl = round(price - atr * 2.0, 2) if atr else None
        tp = [round(price + atr * 4.0, 2)] if atr else None
        return Signal(
            source="ccxt-crypto-fallback",
            action=SignalAction.BUY,
            symbol=symbol.replace("/", ""),
            asset_class=AssetClass.CRYPTO,
            confidence=round(confidence, 3),
            price=price,
            stop_loss=sl,
            take_profit=tp,
            timeframe="1h",
            strategy="RSI-EMA-Fallback",
            metadata=metadata,
        )

    # Sell signal: RSI > 60 (RSI-only, no EMA crossover required)
    if rsi > 60:
        confidence = min(0.4 + (rsi - 60) / 50, 1.0)
        sl = round(price + atr * 2.0, 2) if atr else None
        tp = [round(price - atr * 4.0, 2)] if atr else None
        return Signal(
            source="ccxt-crypto-fallback",
            action=SignalAction.SELL,
            symbol=symbol.replace("/", ""),
            asset_class=AssetClass.CRYPTO,
            confidence=round(confidence, 3),
            price=price,
            stop_loss=sl,
            take_profit=tp,
            timeframe="1h",
            strategy="RSI-EMA-Fallback",
            metadata=metadata,
        )

    return None


# ---------------------------------------------------------------------------
# Telegram direct send (fallback)
# ---------------------------------------------------------------------------


def format_telegram_signal(signal: Signal) -> str:
    """Format signal as Telegram message (HTML)."""
    from datetime import datetime, timezone

    action_emoji = "\U0001f7e2" if signal.action.value == "BUY" else "\U0001f534"
    direction = f"{action_emoji} {signal.action.value.upper()}"

    # Calculate percentages for SL/TP display
    sl_pct = abs(signal.stop_loss - signal.price) / signal.price * 100 if signal.stop_loss else 0
    tp_pct = abs(signal.take_profit[0] - signal.price) / signal.price * 100 if signal.take_profit else 0

    # Get RSI from metadata
    rsi = signal.metadata.get("rsi", "N/A") if signal.metadata else "N/A"

    conf_pct = round(signal.confidence * 100)
    ts = datetime.now(timezone.utc).strftime("%d %b %Y  %H:%M UTC")

    return (
        f"\u26a1 <b>NOVAFX CRYPTO SIGNAL</b>\n\n"
        f"{direction}  <b>{signal.symbol}</b>\n"
        f"\U0001f4b0 Crypto\n\n"
        f"\U0001f4cd <b>Entry:</b> ${signal.price:.2f}\n"
        f"\U0001f534 <b>Stop Loss:</b> ${signal.stop_loss:.2f}  ({sl_pct:.2f}%)\n"
        f"\u2705 <b>Take Profit:</b> ${signal.take_profit[0]:.2f}  ({tp_pct:.2f}%)\n\n"
        f"\U0001f3af <b>Confidence:</b> {conf_pct}%\n"
        f"\U0001f4ca <b>RSI:</b> {rsi}\n"
        f"\U0001f9e0 <b>Strategy:</b> {signal.strategy}\n\n"
        f"\U0001f4c5 <i>{ts}</i>\n"
        f"\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n"
        f"\u26a0\ufe0f <i>Risk max 1-2% per trade. Not financial advice.</i>"
    )


async def send_telegram_direct(signal: Signal) -> bool:
    """Send signal directly to Telegram (bypasses confluence filter)."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_DIRECT_SEND:
        logger.debug("Telegram direct send disabled or no token")
        return False

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    text = format_telegram_signal(signal)

    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                logger.info(
                    "Telegram direct send SUCCESS: %s %s @ $%.2f",
                    signal.action.value,
                    signal.symbol,
                    signal.price,
                )
                return True
            else:
                logger.error(
                    "Telegram direct send FAILED: %s — %s",
                    resp.status_code,
                    resp.text[:200],
                )
                return False
    except Exception as exc:
        logger.error("Telegram direct send ERROR: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Health monitor + main loop
# ---------------------------------------------------------------------------


async def is_freqtrade_healthy() -> bool:
    """Check if Freqtrade API is responsive."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(FREQTRADE_API)
            return resp.status_code == 200
    except Exception:
        return False


async def post_signal(signal: Signal) -> None:
    """POST signal to the dispatcher."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"{DISPATCHER_URL}/signals/ingest",
                json=signal.model_dump(mode="json"),
            )
            resp.raise_for_status()
            logger.info(
                "Signal dispatched: %s %s (confidence=%.2f)",
                signal.action.value, signal.symbol, signal.confidence,
            )
    except Exception:
        logger.exception("Failed to dispatch signal for %s", signal.symbol)


async def run_scanner(manager: DataSourceManager) -> None:
    """Scan all watchlist symbols and generate signals."""
    for symbol in WATCHLIST:
        try:
            result = await manager.get_candles(symbol, "1h", 250)
            candles = result["candles"]
            signal = analyze_candles(symbol, candles)
            if signal:
                # Check for duplicate signal (same pair+direction within cooldown)
                signal_direction = signal.action.value
                if is_duplicate_signal(symbol, signal_direction):
                    logger.info(f"Skipping duplicate signal: {signal_direction} {symbol}")
                    continue
                signal.metadata["data_source"] = result["source"]
                signal.metadata["data_confidence"] = result["confidence"]
                # Send to dispatcher for confluence tracking
                await post_signal(signal)
                # Also send directly to Telegram (bypasses confluence filter)
                await send_telegram_direct(signal)
                # Write to Redis stream for paper trader
                await write_signal_to_stream(
                    symbol.replace("/", ""),
                    signal.action.value,
                    signal.price,
                    signal.confidence,
                )
        except Exception:
            logger.exception("Scanner failed for %s", symbol)


async def main() -> None:
    """Main loop: monitor Freqtrade health, activate scanner when needed."""
    ccxt_source = CCXTSource()
    manager = DataSourceManager([
        ccxt_source,
        TwelveDataCryptoSource(),
        CryptoCompareSource(),
    ])

    await manager.start_health_checks()
    scanner_active = False

    logger.info("Crypto fallback scanner started (interval=%ds)", SCAN_INTERVAL)

    try:
        while True:
            ft_healthy = await is_freqtrade_healthy()

            if ft_healthy and scanner_active:
                logger.info("Freqtrade healthy, scanner dormant")
                scanner_active = False
            elif not ft_healthy and not scanner_active:
                logger.warning("Freqtrade down, scanner active")
                scanner_active = True

            if scanner_active:
                await run_scanner(manager)

            await asyncio.sleep(SCAN_INTERVAL)
    except asyncio.CancelledError:
        pass
    finally:
        await manager.stop_health_checks()
        # Properly close CCXT exchange sessions to avoid unclosed client warnings
        await ccxt_source.close()
        logger.info("Crypto fallback scanner stopped")


if __name__ == "__main__":
    asyncio.run(main())

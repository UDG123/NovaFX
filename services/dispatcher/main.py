"""
NovaFX Dispatcher Service.

Central hub that receives signals from all trading bots,
stores them in Redis Streams, applies confluence filtering,
and publishes actionable signals via Redis pub/sub.
"""

import json
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware

from confluence import ConfluenceEngine
from shared.models import AssetClass, ConfluenceResult, Signal, SignalAction

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("novafx.dispatcher")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CONFLUENCE_WINDOW = int(os.getenv("CONFLUENCE_WINDOW_SEC", "300"))
CONFLUENCE_MIN_SOURCES = int(os.getenv("CONFLUENCE_MIN_SOURCES", "2"))
CONFLUENCE_MIN_CONFIDENCE = float(os.getenv("CONFLUENCE_MIN_CONFIDENCE", "0.6"))
STREAM_MAXLEN = 10000

# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

redis_client: Optional[redis.Redis] = None
confluence_engine: Optional[ConfluenceEngine] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect to Redis on startup, close on shutdown."""
    global redis_client, confluence_engine

    redis_client = redis.from_url(
        REDIS_URL, decode_responses=True, socket_connect_timeout=5
    )
    try:
        await redis_client.ping()
        logger.info("Redis connected: %s", REDIS_URL.split("@")[-1])
    except Exception:
        logger.error("Redis connection failed — dispatcher will not function")
        redis_client = None

    if redis_client:
        confluence_engine = ConfluenceEngine(
            redis_client=redis_client,
            window_sec=CONFLUENCE_WINDOW,
            min_sources=CONFLUENCE_MIN_SOURCES,
            min_confidence=CONFLUENCE_MIN_CONFIDENCE,
        )

    yield

    if redis_client:
        await redis_client.aclose()
        logger.info("Redis connection closed")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NovaFX Dispatcher",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _infer_asset_class(symbol: str) -> AssetClass:
    """Infer asset class from symbol format."""
    s = symbol.upper()
    if "/" in s:
        quote = s.split("/")[1]
        if quote in ("USD", "USDT", "USDC", "BTC", "ETH", "BUSD"):
            return AssetClass.CRYPTO
        if len(quote) == 3:
            return AssetClass.FOREX
    if s in ("XAUUSD", "XAGUSD"):
        return AssetClass.COMMODITIES
    if s in ("SPX500", "NAS100", "US30"):
        return AssetClass.FUTURES
    return AssetClass.STOCKS


async def _ingest_signal(signal: Signal) -> dict:
    """Core ingest logic: store in Redis stream + sorted set, run confluence."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    signal_json = signal.model_dump_json()
    signal_dict = signal.model_dump(mode="json")

    # 1. Store in Redis stream for audit trail
    stream_key = f"signals:{signal.asset_class.value}"
    await redis_client.xadd(
        stream_key,
        {"data": signal_json},
        maxlen=STREAM_MAXLEN,
    )

    # 2. Add to confluence sorted set (scored by timestamp)
    confluence_key = f"confluence:{signal.symbol}"
    score = signal.timestamp.timestamp()
    await redis_client.zadd(confluence_key, {signal_json: score})

    # 3. Prune entries older than 2x confluence window
    cutoff = time.time() - (CONFLUENCE_WINDOW * 2)
    await redis_client.zremrangebyscore(confluence_key, "-inf", cutoff)

    # 4. Run confluence check
    confluence_result: Optional[ConfluenceResult] = None
    if confluence_engine:
        confluence_result = await confluence_engine.evaluate(
            signal.symbol, signal.asset_class
        )

    # 5. Publish if consensus reached
    if confluence_result and redis_client:
        await redis_client.publish(
            "telegram:signals",
            confluence_result.model_dump_json(),
        )
        logger.info(
            "Published confluence hit: %s %s (confidence=%.2f)",
            confluence_result.consensus_action.value,
            confluence_result.symbol,
            confluence_result.weighted_confidence,
        )

    return {
        "status": "accepted",
        "signal_id": signal.signal_id,
        "confluence": confluence_result.model_dump(mode="json") if confluence_result else None,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "novafx-dispatcher",
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/signals/ingest")
async def ingest_signal(signal: Signal):
    """Receive a normalized Signal and process through confluence pipeline."""
    return await _ingest_signal(signal)


@app.post("/webhook/freqtrade")
async def webhook_freqtrade(request: Request):
    """
    Receive a Freqtrade webhook and normalize to Signal schema.

    Expected payload:
        {strategy, pair, side/type, timeframe, current_rate,
         stoploss, confidence}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    pair = body.get("pair", "")
    symbol = pair.replace("/", "")
    side = body.get("side") or body.get("type", "buy")
    strategy = body.get("strategy", "unknown")

    action_map = {"buy": SignalAction.BUY, "sell": SignalAction.SELL}
    action = action_map.get(side.lower(), SignalAction.HOLD)

    signal = Signal(
        source=f"freqtrade-{strategy}",
        action=action,
        symbol=symbol,
        asset_class=AssetClass.CRYPTO,
        confidence=float(body.get("confidence", 0.7)),
        price=float(body.get("current_rate", 0)),
        stop_loss=float(body.get("stoploss", 0)) if body.get("stoploss") else None,
        timeframe=body.get("timeframe", "5m"),
        strategy=strategy,
        metadata={"raw_freqtrade": body},
    )

    logger.info(
        "Freqtrade webhook: %s %s @ %.4f [%s]",
        action.value, symbol, signal.price, strategy,
    )
    return await _ingest_signal(signal)


@app.post("/webhook/tradingview")
async def webhook_tradingview(request: Request):
    """
    Receive a TradingView alert and normalize to Signal schema.

    Expected payload:
        {ticker, action, price/close, confidence, stop_loss,
         take_profit, timeframe, strategy}
    """
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    ticker = body.get("ticker", "")
    symbol = ticker.replace("/", "")
    action_str = body.get("action", "buy").lower()

    action_map = {
        "buy": SignalAction.BUY,
        "sell": SignalAction.SELL,
        "close": SignalAction.CLOSE,
    }
    action = action_map.get(action_str, SignalAction.HOLD)

    price = float(body.get("price") or body.get("close", 0))
    asset_class = _infer_asset_class(ticker)

    tp_raw = body.get("take_profit")
    take_profit = None
    if tp_raw:
        if isinstance(tp_raw, list):
            take_profit = [float(t) for t in tp_raw]
        else:
            take_profit = [float(tp_raw)]

    signal = Signal(
        source="tradingview",
        action=action,
        symbol=symbol,
        asset_class=asset_class,
        confidence=float(body.get("confidence", 0.7)),
        price=price,
        stop_loss=float(body.get("stop_loss")) if body.get("stop_loss") else None,
        take_profit=take_profit,
        timeframe=body.get("timeframe", "15m"),
        strategy=body.get("strategy", "tradingview-alert"),
        metadata={"raw_tradingview": body},
    )

    logger.info(
        "TradingView webhook: %s %s @ %.4f [%s]",
        action.value, symbol, price, asset_class.value,
    )
    return await _ingest_signal(signal)


@app.get("/signals/recent/{asset_class}")
async def recent_signals(asset_class: str, count: int = Query(default=50, ge=1, le=500)):
    """Fetch recent signals from a Redis stream by asset class."""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")

    try:
        ac = AssetClass(asset_class)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid asset_class. Valid: {[e.value for e in AssetClass]}",
        )

    stream_key = f"signals:{ac.value}"
    entries = await redis_client.xrevrange(stream_key, count=count)

    signals = []
    for entry_id, fields in entries:
        try:
            data = json.loads(fields.get("data", "{}"))
            data["_stream_id"] = entry_id
            signals.append(data)
        except Exception:
            continue

    return {
        "asset_class": ac.value,
        "count": len(signals),
        "signals": signals,
    }


@app.get("/signals/confluence/{symbol}")
async def check_confluence(symbol: str):
    """Manually check confluence status for a symbol."""
    if not confluence_engine:
        raise HTTPException(status_code=503, detail="Confluence engine not available")

    asset_class = _infer_asset_class(symbol)
    result = await confluence_engine.evaluate(symbol, asset_class)

    return {
        "symbol": symbol,
        "asset_class": asset_class.value,
        "confluence": result.model_dump(mode="json") if result else None,
    }


# ---------------------------------------------------------------------------
# Scan Endpoints (for n8n workflows)
# ---------------------------------------------------------------------------


async def _get_recent_signals(asset_class: str, count: int = 20) -> list[dict]:
    """Helper to fetch recent signals from Redis stream."""
    if not redis_client:
        return []

    stream_key = f"signals:{asset_class}"
    try:
        entries = await redis_client.xrevrange(stream_key, count=count)
        signals = []
        for entry_id, fields in entries:
            try:
                data = json.loads(fields.get("data", "{}"))
                signals.append(data)
            except Exception:
                continue
        return signals
    except Exception:
        return []


@app.get("/scan/forex")
async def scan_forex():
    """
    Fetch latest forex signals from Redis.
    Used by n8n workflows to poll for new signals.
    """
    signals = await _get_recent_signals("forex", count=20)
    # Filter to only recent signals (last 5 minutes)
    cutoff = time.time() - 300
    recent = [
        s for s in signals
        if datetime.fromisoformat(s.get("timestamp", "1970-01-01T00:00:00+00:00").replace("Z", "+00:00")).timestamp() > cutoff
    ]
    return {
        "asset_class": "forex",
        "signals": recent,
        "count": len(recent),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/scan/crypto")
async def scan_crypto():
    """
    Fetch latest crypto signals from Redis.
    Used by n8n workflows to poll for new signals.
    """
    signals = await _get_recent_signals("crypto", count=20)
    cutoff = time.time() - 300
    recent = [
        s for s in signals
        if datetime.fromisoformat(s.get("timestamp", "1970-01-01T00:00:00+00:00").replace("Z", "+00:00")).timestamp() > cutoff
    ]
    return {
        "asset_class": "crypto",
        "signals": recent,
        "count": len(recent),
        "ts": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/scan/stocks")
async def scan_stocks():
    """
    Fetch latest stock signals from Redis.
    Used by n8n workflows to poll for new signals.
    """
    signals = await _get_recent_signals("stocks", count=20)
    cutoff = time.time() - 300
    recent = [
        s for s in signals
        if datetime.fromisoformat(s.get("timestamp", "1970-01-01T00:00:00+00:00").replace("Z", "+00:00")).timestamp() > cutoff
    ]
    return {
        "asset_class": "stocks",
        "signals": recent,
        "count": len(recent),
        "ts": datetime.now(timezone.utc).isoformat(),
    }

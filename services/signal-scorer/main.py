"""
NovaFX v2 Signal Scorer
Reads signals from streams, applies regime modifiers, and publishes scored signals.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import redis.asyncio as redis
from aiohttp import web

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
PORT = int(os.getenv("PORT", 8000))
BLOCK_MS = int(os.getenv("BLOCK_MS", 2000))

# Redis keys
STREAM_FOREX = "novafx:signals:forex"
STREAM_CRYPTO = "novafx:signals:crypto"
STREAM_STOCKS = "novafx:signals:stocks"
STREAM_SCORED = "novafx:signals:scored"
REGIME_KEY = "novafx:regime"

# Track last read IDs
last_ids = {
    STREAM_FOREX: "0",
    STREAM_CRYPTO: "0",
    STREAM_STOCKS: "0",
}


async def get_regime(redis_client: redis.Redis) -> Optional[dict]:
    """Fetch current regime from Redis."""
    try:
        data = await redis_client.get(REGIME_KEY)
        if data:
            return json.loads(data)
    except Exception as e:
        logger.error(f"Error fetching regime: {e}")
    return None


def apply_regime_modifier(signal: dict, regime_data: Optional[dict]) -> tuple[int, dict]:
    """
    Apply regime modifiers to signal score.
    Returns (final_score, enriched_signal).

    Modifiers:
    - risk_off: score -> 0 (skip)
    - trending: +10
    - ranging: -10
    - out of prime session (forex): -15
    """
    base_score = signal.get("score", 0)
    desk = signal.get("desk", "unknown")
    modifiers = []

    if not regime_data:
        # No regime data, use base score
        return base_score, {**signal, "regime": "unknown", "prime_session": None, "modifiers": []}

    # Get desk-specific regime
    desk_regime = regime_data.get(desk, {})
    regime = desk_regime.get("regime", "mild_trend")
    prime_session = regime_data.get("prime_session", True)

    # Risk-off: skip entirely
    if regime == "risk_off":
        modifiers.append("risk_off: -100")
        return 0, {**signal, "regime": regime, "prime_session": prime_session, "modifiers": modifiers}

    final_score = base_score

    # Trending bonus
    if regime == "trending":
        final_score += 10
        modifiers.append("trending: +10")

    # Ranging penalty
    elif regime == "ranging":
        final_score -= 10
        modifiers.append("ranging: -10")

    # Out of prime session penalty (forex only)
    if desk == "forex" and not prime_session:
        final_score -= 15
        modifiers.append("out_of_prime_session: -15")

    return final_score, {
        **signal,
        "base_score": base_score,
        "final_score": final_score,
        "regime": regime,
        "prime_session": prime_session,
        "modifiers": modifiers,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


async def process_signals(redis_client: redis.Redis) -> None:
    """Process signals from all streams."""
    global last_ids

    try:
        # Read from all streams with blocking
        streams = {
            STREAM_FOREX: last_ids[STREAM_FOREX],
            STREAM_CRYPTO: last_ids[STREAM_CRYPTO],
            STREAM_STOCKS: last_ids[STREAM_STOCKS],
        }

        results = await redis_client.xread(streams, block=BLOCK_MS)

        if not results:
            return

        # Get current regime
        regime_data = await get_regime(redis_client)

        for stream_name, messages in results:
            stream_name = stream_name if isinstance(stream_name, str) else stream_name.decode()

            for msg_id, msg_data in messages:
                msg_id = msg_id if isinstance(msg_id, str) else msg_id.decode()
                last_ids[stream_name] = msg_id

                try:
                    # Parse signal data
                    data_str = msg_data.get("data") or msg_data.get(b"data", b"").decode()
                    signal = json.loads(data_str)

                    # Apply regime modifiers
                    final_score, enriched_signal = apply_regime_modifier(signal, regime_data)

                    logger.info(
                        f"Processing {signal.get('symbol')} {signal.get('direction')}: "
                        f"base={signal.get('score')} -> final={final_score}"
                    )

                    # Only publish if final score >= 65
                    if final_score >= 65:
                        await redis_client.xadd(STREAM_SCORED, {"data": json.dumps(enriched_signal)})
                        logger.info(
                            f"Published scored signal: {enriched_signal['symbol']} "
                            f"{enriched_signal['direction']} final_score={final_score}"
                        )
                    else:
                        logger.info(f"Skipped {signal.get('symbol')}: final_score {final_score} < 65")

                except Exception as e:
                    logger.error(f"Error processing message {msg_id}: {e}")

    except Exception as e:
        logger.error(f"Error in process_signals: {e}")


async def run_scorer() -> None:
    """Main scorer loop."""
    redis_client = redis.from_url(REDIS_URL, decode_responses=True)

    # Initialize last IDs to latest
    for stream in [STREAM_FOREX, STREAM_CRYPTO, STREAM_STOCKS]:
        try:
            info = await redis_client.xinfo_stream(stream)
            last_ids[stream] = info.get("last-generated-id", "0")
        except Exception:
            # Stream doesn't exist yet
            last_ids[stream] = "0"

    logger.info(f"Starting from stream positions: {last_ids}")

    while True:
        try:
            await process_signals(redis_client)
        except Exception as e:
            logger.error(f"Scorer error: {e}")
            await asyncio.sleep(5)


# Health server
async def health_handler(request: web.Request) -> web.Response:
    return web.Response(text='{"status":"ok"}', content_type="application/json")


async def stats_handler(request: web.Request) -> web.Response:
    """Return scorer stats."""
    stats = {
        "last_ids": last_ids,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    return web.Response(text=json.dumps(stats), content_type="application/json")


async def start_health_server() -> None:
    app = web.Application()
    app.router.add_get("/health", health_handler)
    app.router.add_get("/", health_handler)
    app.router.add_get("/stats", stats_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    await site.start()
    logger.info(f"Health server started on port {PORT}")


async def main() -> None:
    logger.info("NovaFX v2 Signal Scorer starting...")
    await asyncio.gather(
        start_health_server(),
        run_scorer(),
    )


if __name__ == "__main__":
    asyncio.run(main())

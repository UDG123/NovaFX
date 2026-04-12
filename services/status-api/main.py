"""
STATUS-API: Live status endpoint for all NovaFX floors
Reads scan logs from Redis and returns them as JSON.
"""
import os
import json
from datetime import datetime, timezone
from aiohttp import web
import redis.asyncio as redis
import httpx

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
PORT = int(os.environ.get("PORT", "8080"))
TELEGRAM_SYSTEM_CHAT_ID = os.environ.get("TELEGRAM_SYSTEM_CHAT_ID", "-1003710749613")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")

# Floor name mapping
FLOOR_NAMES = {
    "floor1": "TITAN",
    "floor2": "DRAGON",
    "floor3": "CIPHER",
    "floor4": "VERTEX",
    "floor5": "APEX",
    "floor6": "BULLION",
}


async def handle_status(request):
    """Return status of all 6 floors plus cross-floor intel."""
    r = redis.from_url(REDIS_URL, decode_responses=True)

    try:
        status = {}

        # Get all floor logs
        for i in range(1, 7):
            key = f"novafx:log:floor{i}"
            val = await r.get(key)
            floor_key = f"floor{i}"
            if val:
                try:
                    data = json.loads(val)
                    data["name"] = FLOOR_NAMES.get(floor_key, f"Floor {i}")
                    status[floor_key] = data
                except json.JSONDecodeError:
                    status[floor_key] = {"status": "invalid_json", "name": FLOOR_NAMES.get(floor_key)}
            else:
                status[floor_key] = {"status": "no_data", "name": FLOOR_NAMES.get(floor_key)}

        # Get cross-floor intel
        dxy = await r.get("novafx:cross:dxy_state")
        vix = await r.get("novafx:cross:vix_regime")
        status["intel"] = {
            "dxy_state": dxy if dxy else None,
            "vix_regime": vix if vix else None,
        }

        return web.json_response(status)

    finally:
        await r.aclose()


async def handle_floor(request):
    """Return status of a specific floor."""
    floor_num = request.match_info.get("floor", "1")

    if not floor_num.isdigit() or int(floor_num) < 1 or int(floor_num) > 6:
        return web.json_response({"error": "Invalid floor number (1-6)"}, status=400)

    r = redis.from_url(REDIS_URL, decode_responses=True)

    try:
        key = f"novafx:log:floor{floor_num}"
        val = await r.get(key)

        if val:
            try:
                data = json.loads(val)
                data["name"] = FLOOR_NAMES.get(f"floor{floor_num}", f"Floor {floor_num}")
                return web.json_response(data)
            except json.JSONDecodeError:
                return web.json_response({"error": "Invalid JSON in Redis"}, status=500)
        else:
            return web.json_response({
                "status": "no_data",
                "name": FLOOR_NAMES.get(f"floor{floor_num}", f"Floor {floor_num}"),
            })

    finally:
        await r.aclose()


async def handle_health(request):
    """Health check endpoint with floor status."""
    r = redis.from_url(REDIS_URL, decode_responses=True)

    try:
        floors = {}
        for i in range(1, 7):
            key = f"novafx:log:floor{i}"
            val = await r.get(key)
            floors[f"floor{i}"] = "ok" if val else "no_data"

        return web.json_response({
            "status": "healthy",
            "service": "novafx-status-api",
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "floors": floors,
        })

    finally:
        await r.aclose()


async def handle_alert(request):
    """Send test alert to system Telegram channel."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    message = f"🔔 NovaFX Alert Test | Pipeline OK | {timestamp}"

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_SYSTEM_CHAT_ID,
        "text": message,
        "parse_mode": "HTML",
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 200:
                return web.json_response({
                    "status": "sent",
                    "message": message,
                    "chat_id": TELEGRAM_SYSTEM_CHAT_ID,
                })
            else:
                return web.json_response({
                    "status": "error",
                    "code": resp.status_code,
                    "response": resp.text,
                }, status=500)
    except Exception as e:
        return web.json_response({
            "status": "error",
            "error": str(e),
        }, status=500)


def create_app():
    """Create and configure the aiohttp application."""
    app = web.Application()
    app.router.add_get("/status", handle_status)
    app.router.add_get("/status/{floor}", handle_floor)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/alert", handle_alert)
    return app


if __name__ == "__main__":
    print(f"[STATUS-API] Starting on port {PORT}")
    app = create_app()
    web.run_app(app, port=PORT, print=lambda s: print(f"[STATUS-API] {s}"))

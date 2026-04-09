"""
STATUS-API: Live status endpoint for all NovaFX floors
Reads scan logs from Redis and returns them as JSON.
"""
import os
import json
from aiohttp import web
import redis.asyncio as redis

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
PORT = int(os.environ.get("PORT", "8080"))

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
    """Health check endpoint."""
    return web.json_response({"ok": True})


def create_app():
    """Create and configure the aiohttp application."""
    app = web.Application()
    app.router.add_get("/status", handle_status)
    app.router.add_get("/status/{floor}", handle_floor)
    app.router.add_get("/health", handle_health)
    return app


if __name__ == "__main__":
    print(f"[STATUS-API] Starting on port {PORT}")
    app = create_app()
    web.run_app(app, port=PORT, print=lambda s: print(f"[STATUS-API] {s}"))

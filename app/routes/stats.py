import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.db.signal_store import get_weekly_stats

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/signals/stats")
async def signal_stats():
    stats = await get_weekly_stats()
    if not stats:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "reason": "database not configured"},
        )
    return {
        "status": "ok",
        "period": "last_7_days",
        "total_signals": stats["total"],
        "buys": stats["buys"],
        "sells": stats["sells"],
        "by_desk": stats["by_desk"],
    }

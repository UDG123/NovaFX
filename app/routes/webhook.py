import hmac
import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.signals import IncomingSignal
from app.services.bot_state import BotState
from app.services.signal_processor import process_signal
from app.services.telegram import send_signal

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health():
    state = BotState.get()

    if not state.scheduler_running():
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "reason": "scheduler not running",
                "service": "NovaFX Signal Bot",
            },
        )

    next_run = state.get_next_run_time()
    return {
        "status": "healthy",
        "service": "NovaFX Signal Bot",
        "uptime_seconds": state.uptime_seconds(),
        "active_strategy": state.active_strategy,
        "scheduler_next_run": next_run.isoformat() if next_run else None,
        "last_fetch_per_symbol": {
            sym: ts.isoformat() for sym, ts in state.last_fetch_times.items()
        },
    }


@router.post("/webhook")
async def receive_webhook(signal: IncomingSignal):
    if settings.WEBHOOK_SECRET:
        if not signal.secret or not hmac.compare_digest(signal.secret, settings.WEBHOOK_SECRET):
            raise HTTPException(status_code=401, detail="Invalid webhook secret")

    logger.info("Webhook received: %s %s @ %s", signal.action, signal.symbol, signal.price)

    processed = process_signal(signal)
    sent = await send_signal(processed)

    return {
        "status": "processed",
        "telegram_sent": sent,
        "signal": processed.model_dump(),
    }

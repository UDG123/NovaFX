import hmac
import logging

from fastapi import APIRouter, HTTPException

from app.config import settings
from app.models.signals import IncomingSignal
from app.services.signal_processor import process_signal
from app.services.telegram import send_signal

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health():
    return {"status": "healthy"}


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

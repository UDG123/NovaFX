import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.models.signals import IncomingSignal
from app.services.api_tracker import APITracker
from app.services.bot_state import BotState
from app.services.htf_bias import get_htf_bias
from app.services.signal_processor import process_signal
from app.services.telegram import send_signal

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def health():
    state = BotState.get()
    api = APITracker.get()
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "signal_engine_enabled": settings.SIGNAL_ENGINE_ENABLED,
        "signals_sent_session": state.signals_sent,
        "twelvedata_api": api.status(),
    }


@router.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    secret = body.get("secret", "")
    if secret != settings.WEBHOOK_SECRET:
        raise HTTPException(status_code=403, detail="Invalid secret")

    try:
        signal = IncomingSignal(**body)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    processed = process_signal(signal)
    bias = get_htf_bias(signal.symbol, signal.action)
    sent = await send_signal(processed, htf_bias=bias)

    state = BotState.get()
    state.record_signal(signal)

    return {
        "status": "ok",
        "symbol": processed.symbol,
        "action": processed.action,
        "entry_price": processed.entry_price,
        "stop_loss": processed.stop_loss,
        "take_profit_1": processed.take_profit_1,
        "take_profit_2": processed.take_profit_2,
        "take_profit_3": processed.take_profit_3,
        "risk_reward": processed.risk_reward,
        "risk_amount": processed.risk_amount,
        "telegram_sent": sent,
    }

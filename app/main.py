import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from app.config import settings
from app.db.database import init_db
from app.routes.stats import router as stats_router
from app.routes.webhook import router
from app.services.bot_commands import poll_telegram_commands
from app.services.bot_state import BotState
from app.services.signal_engine import run_signal_engine
from app.services.signal_processor import process_signal
from app.services.telegram import send_signal
from app.db.signal_store import save_signal
from app.services.htf_bias import get_htf_bias
from app.services.outcome_engine import register_signal, run_outcome_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress httpx logs to prevent bot token exposure in Railway logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


async def scheduled_signal_engine():
    state = BotState.get()
    signals = await run_signal_engine()
    for raw_signal in signals:
        try:
            state.record_signal(raw_signal)
            processed = process_signal(raw_signal)
            bias = get_htf_bias(raw_signal.symbol, raw_signal.action)
            await send_signal(processed, htf_bias=bias)
            await save_signal(processed)
            await register_signal(processed)
        except Exception:
            logger.exception(
                "Failed to process engine signal: %s %s",
                raw_signal.action,
                raw_signal.symbol,
            )
    logger.info("Scheduled run complete - %d signals processed", len(signals))


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    scheduler = AsyncIOScheduler()
    state = BotState.get()
    state.scheduler = scheduler

    if settings.SIGNAL_ENGINE_ENABLED:
        scheduler.add_job(
            scheduled_signal_engine,
            "interval",
            minutes=settings.SIGNAL_ENGINE_INTERVAL_MINUTES,
            id="signal_engine",
            name="NovaFX Signal Engine",
        )
        logger.info(
            "Signal engine scheduled every %d minutes",
            settings.SIGNAL_ENGINE_INTERVAL_MINUTES,
        )

    if settings.TELEGRAM_BOT_TOKEN:
        scheduler.add_job(
            poll_telegram_commands,
            "interval",
            seconds=5,
            id="telegram_poller",
            name="Telegram Command Poller",
        )
        logger.info("Telegram command poller started (every 5s)")

    scheduler.add_job(
        run_outcome_engine,
        "interval",
        seconds=60,
        id="outcome_engine",
        name="NovaFX Outcome Monitor",
    )
    logger.info("Outcome engine scheduled every 60 seconds")

    scheduler.start()

    yield

    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down")


app = FastAPI(
    title="NovaFX Signal Bot",
    description="Trading signal processor with TradingView webhook + built-in signal engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(stats_router)


@app.get("/")
async def root():
    return {
        "service": "NovaFX Signal Bot",
        "version": "1.0.0",
        "signal_engine_enabled": settings.SIGNAL_ENGINE_ENABLED,
        "signal_engine_interval": f"{settings.SIGNAL_ENGINE_INTERVAL_MINUTES}m",
        "endpoints": ["/", "/health", "/webhook", "/signals/stats"],
    }

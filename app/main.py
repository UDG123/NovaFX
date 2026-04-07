import logging
from contextlib import asynccontextmanager

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from app.config import settings
from app.db.database import Base, engine, init_db
from app.db.signal_store import save_signal
from app.db.trade_monitor import TradePosition
from app.routes.stats import router as stats_router
from app.routes.webhook import router
from app.services.bot_commands import poll_telegram_commands
from app.services.bot_state import BotState
from app.services.outcome_engine import register_signal, run_outcome_engine
from app.services.signal_engine import run_signal_engine
from app.services.signal_processor import process_signal
from app.services.telegram import send_signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("yfinance").setLevel(logging.WARNING)
logging.getLogger("peewee").setLevel(logging.WARNING)


async def scheduled_signal_engine():
    state = BotState.get()
    results = await run_signal_engine()
    for raw_signal, htf_bias, df in results:
        try:
            state.record_signal(raw_signal)
            processed = process_signal(raw_signal, df=df)
            await send_signal(processed, htf_bias=htf_bias)
            await save_signal(processed)
            await register_signal(processed)
        except Exception:
            logger.exception(
                "Failed to process signal: %s %s",
                raw_signal.action, raw_signal.symbol,
            )
    logger.info("Scheduled run complete - %d signals processed", len(results))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Init all DB tables including new TradePosition
    await init_db()
    if engine:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

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

        # Outcome engine runs every 60 seconds
        scheduler.add_job(
            run_outcome_engine,
            "interval",
            seconds=60,
            id="outcome_engine",
            name="NovaFX Outcome Engine",
        )
        logger.info("Outcome engine scheduled every 60 seconds")

    if settings.TELEGRAM_BOT_TOKEN:
        scheduler.add_job(
            poll_telegram_commands,
            "interval",
            seconds=5,
            id="telegram_poller",
            name="Telegram Command Poller",
        )
        logger.info("Telegram command poller started (every 5s)")

    scheduler.start()
    yield

    if scheduler.running:
        scheduler.shutdown(wait=False)
        logger.info("Scheduler shut down")


app = FastAPI(
    title="NovaFX Signal Bot",
    version="1.1.0",
    lifespan=lifespan,
)

app.include_router(router)
app.include_router(stats_router)


@app.get("/")
async def root():
    return {
        "service": "NovaFX Signal Bot",
        "version": "1.1.0",
        "signal_engine": f"every {settings.SIGNAL_ENGINE_INTERVAL_MINUTES}m",
        "outcome_engine": "every 60s",
        "endpoints": ["/", "/health", "/webhook", "/signals/stats"],
    }

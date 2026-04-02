import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal
from app.db.models import SignalHistory
from app.models.signals import ProcessedSignal

logger = logging.getLogger(__name__)

DESK_MAP = {
    "EURUSD": "TG_DESK1", "GBPUSD": "TG_DESK1", "USDJPY": "TG_DESK1",
    "AUDUSD": "TG_DESK1", "USDCAD": "TG_DESK1", "USDCHF": "TG_DESK1",
    "NZDUSD": "TG_DESK1",
    "EURGBP": "TG_DESK2", "EURJPY": "TG_DESK2", "GBPJPY": "TG_DESK2",
    "BTCUSDT": "TG_DESK3", "ETHUSDT": "TG_DESK3", "SOLUSDT": "TG_DESK3",
    "BNBUSDT": "TG_DESK3", "XRPUSDT": "TG_DESK3",
    "AAPL": "TG_DESK4", "MSFT": "TG_DESK4", "NVDA": "TG_DESK4",
    "TSLA": "TG_DESK4", "SPY": "TG_DESK4", "QQQ": "TG_DESK4",
    "XAUUSD": "TG_DESK5", "XAGUSD": "TG_DESK5",
    "SPX500": "TG_DESK6", "NAS100": "TG_DESK6", "US30": "TG_DESK6",
}


async def save_signal(signal: ProcessedSignal, regime: str | None = None) -> None:
    if AsyncSessionLocal is None:
        return

    desk = DESK_MAP.get(signal.symbol.upper())

    try:
        async with AsyncSessionLocal() as session:
            record = SignalHistory(
                symbol=signal.symbol,
                action=signal.action,
                entry_price=signal.entry_price,
                stop_loss=signal.stop_loss,
                take_profit_1=signal.take_profit_1,
                take_profit_2=signal.take_profit_2,
                take_profit_3=signal.take_profit_3,
                risk_reward=signal.risk_reward,
                risk_amount=signal.risk_amount,
                timeframe=signal.timeframe,
                source=signal.source,
                indicator=signal.indicator,
                regime=regime,
                desk=desk,
            )
            session.add(record)
            await session.commit()
            logger.info("Signal saved to DB: %s %s", signal.action, signal.symbol)
    except Exception:
        logger.exception("Failed to save signal to DB: %s %s", signal.action, signal.symbol)


async def get_weekly_stats() -> dict:
    if AsyncSessionLocal is None:
        return {}

    try:
        async with AsyncSessionLocal() as session:
            week_ago = datetime.now(timezone.utc) - timedelta(days=7)

            total = await session.scalar(
                select(func.count()).where(SignalHistory.created_at >= week_ago)
            )
            buys = await session.scalar(
                select(func.count()).where(
                    SignalHistory.created_at >= week_ago,
                    SignalHistory.action == "BUY",
                )
            )
            sells = await session.scalar(
                select(func.count()).where(
                    SignalHistory.created_at >= week_ago,
                    SignalHistory.action == "SELL",
                )
            )

            by_desk = {}
            for desk in ["TG_DESK1","TG_DESK2","TG_DESK3","TG_DESK4","TG_DESK5","TG_DESK6"]:
                count = await session.scalar(
                    select(func.count()).where(
                        SignalHistory.created_at >= week_ago,
                        SignalHistory.desk == desk,
                    )
                )
                by_desk[desk] = count or 0

            return {
                "total": total or 0,
                "buys": buys or 0,
                "sells": sells or 0,
                "by_desk": by_desk,
            }
    except Exception:
        logger.exception("Failed to get weekly stats")
        return {}

"""
NovaFX Outcome Engine.

Monitors all open trade positions every 60 seconds.
Detects TP1/TP2/TP3/SL hits and posts results to Telegram.

State machine:
  PENDING -> OPEN (entry price hit)
  OPEN    -> TP1_HIT (1/3 closed, SL -> breakeven)
  TP1_HIT -> TP2_HIT (2/3 closed)
  TP2_HIT -> TP3_HIT (fully closed -- full win)
  Any active state -> SL_HIT (loss or breakeven after TP1)
  PENDING -> EXPIRED (entry not hit within 24h)
"""
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update

from app.config import settings
from app.db.database import AsyncSessionLocal
from app.db.trade_monitor import SignalStatus, TradePosition
from app.models.signals import ProcessedSignal
from app.services.pnl_calculator import calculate_pnl, format_pnl_display
from app.services.price_monitor import get_current_price
from app.services.telegram import _post_to_channel

logger = logging.getLogger(__name__)

DESK_MAP = {
    "EURUSD": "TG_DESK1", "GBPUSD": "TG_DESK1", "USDJPY": "TG_DESK1",
    "AUDUSD": "TG_DESK1", "USDCAD": "TG_DESK1", "USDCHF": "TG_DESK1",
    "NZDUSD": "TG_DESK1",
    "EURGBP": "TG_DESK2", "EURJPY": "TG_DESK2", "GBPJPY": "TG_DESK2",
    "BTCUSDT": "TG_DESK3", "ETHUSDT": "TG_DESK3", "SOLUSDT": "TG_DESK3",
    "BNBUSDT": "TG_DESK3", "XRPUSDT": "TG_DESK3",
    "BTCUSD": "TG_DESK3", "ETHUSD": "TG_DESK3", "SOLUSD": "TG_DESK3",
    "BNBUSD": "TG_DESK3", "XRPUSD": "TG_DESK3",
    "AAPL": "TG_DESK4", "MSFT": "TG_DESK4", "NVDA": "TG_DESK4",
    "TSLA": "TG_DESK4", "SPY": "TG_DESK4", "QQQ": "TG_DESK4",
    "XAUUSD": "TG_DESK5", "XAGUSD": "TG_DESK5",
    "SPX500": "TG_DESK6", "NAS100": "TG_DESK6", "US30": "TG_DESK6",
}


async def register_signal(signal: ProcessedSignal) -> None:
    """
    Register a new signal as a PENDING trade position.
    Called immediately after send_signal() in the engine loop.
    """
    if AsyncSessionLocal is None:
        return

    desk = DESK_MAP.get(signal.symbol.upper().replace("/", "").replace("-", ""))
    signal_id = f"{signal.symbol}-{int(datetime.now(timezone.utc).timestamp())}"

    position = TradePosition(
        signal_id=signal_id,
        symbol=signal.symbol,
        desk=desk,
        action=signal.action,
        entry_price=signal.entry_price,
        stop_loss=signal.stop_loss,
        current_sl=signal.stop_loss,
        tp1=signal.take_profit_1,
        tp2=signal.take_profit_2,
        tp3=signal.take_profit_3,
        risk_amount=signal.risk_amount,
        risk_reward=signal.risk_reward,
        timeframe=signal.timeframe,
        indicator=signal.indicator,
        status=SignalStatus.PENDING,
        expires_at=datetime.now(timezone.utc) + timedelta(hours=24),
    )

    try:
        async with AsyncSessionLocal() as session:
            session.add(position)
            await session.commit()
            logger.info("Registered position: %s %s", signal.action, signal.symbol)
    except Exception:
        logger.exception("Failed to register position for %s", signal.symbol)


async def run_outcome_engine() -> None:
    """
    Main monitoring loop -- called every 60 seconds by APScheduler.
    Checks all active positions against current market prices.
    """
    if AsyncSessionLocal is None:
        return

    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(TradePosition).where(
                TradePosition.status.in_([
                    SignalStatus.PENDING,
                    SignalStatus.OPEN,
                    SignalStatus.TP1_HIT,
                    SignalStatus.TP2_HIT,
                ])
            )
        )
        positions = result.scalars().all()

    if not positions:
        return

    logger.info("Outcome engine: monitoring %d active positions", len(positions))

    for position in positions:
        try:
            await _evaluate_position(position)
        except Exception:
            logger.exception("Error evaluating position %s", position.signal_id)


async def _evaluate_position(position: TradePosition) -> None:
    """Evaluate a single position against current price."""
    now = datetime.now(timezone.utc)

    # Check expiry for PENDING positions
    if position.status == SignalStatus.PENDING:
        if position.expires_at and now > position.expires_at:
            await _close_position(position, SignalStatus.EXPIRED, None)
            return

    current_price = await get_current_price(position.symbol)
    if current_price is None:
        logger.warning("No price for %s \u2014 skipping this cycle", position.symbol)
        return

    is_buy = position.action == "BUY"

    # PENDING: check if entry price has been hit
    if position.status == SignalStatus.PENDING:
        entry_hit = (
            current_price <= position.entry_price if is_buy
            else current_price >= position.entry_price
        )
        if entry_hit:
            await _update_status(position, SignalStatus.OPEN,
                                 entry_hit_at=now)
            logger.info("Entry hit: %s %s @ %.5f",
                        position.action, position.symbol, current_price)
        return  # Don't evaluate TP/SL until entry is confirmed

    # Use current_sl (may have moved to breakeven after TP1)
    sl = position.current_sl or position.stop_loss

    # Check SL first (always takes priority)
    sl_hit = (current_price <= sl) if is_buy else (current_price >= sl)
    if sl_hit:
        pnl = _calculate_outcome_pnl(position, sl)
        await _close_position(position, SignalStatus.SL_HIT, sl, pnl)
        await _send_result(position, SignalStatus.SL_HIT, sl, pnl)
        return

    # Check TPs in order
    if position.status == SignalStatus.OPEN:
        tp1_hit = (current_price >= position.tp1) if is_buy else (current_price <= position.tp1)
        if tp1_hit:
            await _update_status(
                position, SignalStatus.TP1_HIT,
                tp1_hit_at=now,
                current_sl=position.entry_price,
                sl_moved_to_be=True,
            )
            await _send_tp_notification(position, level=1, price=position.tp1)
            return

    elif position.status == SignalStatus.TP1_HIT:
        tp2_hit = (current_price >= position.tp2) if is_buy else (current_price <= position.tp2)
        if tp2_hit:
            await _update_status(position, SignalStatus.TP2_HIT, tp2_hit_at=now)
            await _send_tp_notification(position, level=2, price=position.tp2)
            return

    elif position.status == SignalStatus.TP2_HIT:
        tp3_hit = (current_price >= position.tp3) if is_buy else (current_price <= position.tp3)
        if tp3_hit:
            pnl = _calculate_outcome_pnl(position, position.tp3)
            await _close_position(position, SignalStatus.TP3_HIT, position.tp3, pnl)
            await _send_result(position, SignalStatus.TP3_HIT, position.tp3, pnl)
            return


def _calculate_outcome_pnl(position: TradePosition, exit_price: float) -> dict:
    return calculate_pnl(
        symbol=position.symbol,
        action=position.action,
        entry_price=position.entry_price,
        exit_price=exit_price,
        risk_amount=position.risk_amount,
        stop_loss=position.stop_loss,
    )


async def _update_status(position: TradePosition, status: SignalStatus, **kwargs) -> None:
    if AsyncSessionLocal is None:
        return
    async with AsyncSessionLocal() as session:
        await session.execute(
            update(TradePosition)
            .where(TradePosition.id == position.id)
            .values(status=status, **kwargs)
        )
        await session.commit()
    position.status = status
    for k, v in kwargs.items():
        setattr(position, k, v)


async def _close_position(
    position: TradePosition,
    status: SignalStatus,
    exit_price: float | None,
    pnl: dict | None = None,
) -> None:
    if AsyncSessionLocal is None:
        return
    now = datetime.now(timezone.utc)
    duration = None
    if position.entry_hit_at:
        duration = int((now - position.entry_hit_at).total_seconds() / 60)

    updates = {
        "status": status,
        "closed_at": now,
        "exit_price": exit_price,
        "duration_minutes": duration,
    }
    if pnl:
        updates["pnl_pips"] = pnl["pnl_pips"]
        updates["pnl_dollars"] = pnl["pnl_dollars"]

    async with AsyncSessionLocal() as session:
        await session.execute(
            update(TradePosition)
            .where(TradePosition.id == position.id)
            .values(**updates)
        )
        await session.commit()

    logger.info(
        "Position closed: %s %s \u2192 %s | P&L: %s pips / $%s",
        position.action, position.symbol, status.value,
        pnl["pnl_pips"] if pnl else "N/A",
        pnl["pnl_dollars"] if pnl else "N/A",
    )


async def _send_tp_notification(
    position: TradePosition, level: int, price: float
) -> None:
    """Send TP hit notification to the desk channel."""
    emoji_map = {1: "\U0001f3af", 2: "\U0001f3af\U0001f3af", 3: "\U0001f3c6"}
    emoji = emoji_map.get(level, "\u2705")

    remaining = {1: "TP2 & TP3 still active \u2014 SL moved to breakeven",
                 2: "TP3 still active \u2014 riding to final target",
                 3: ""}

    msg = (
        f"{emoji} <b>TP{level} HIT \u2014 {position.symbol} {position.action}</b>\n\n"
        f"\U0001f4ca <b>{position.symbol}</b>  |  {position.action}\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        f"\U0001f4b0 Entry: <code>{position.entry_price:.5g}</code>\n"
        f"\U0001f3af TP{level}: <code>{price:.5g}</code> \u2705\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        f"\U0001f4dd <i>{remaining.get(level, '')}</i>"
    )

    await _send_to_desk(position.desk, msg)


async def _send_result(
    position: TradePosition,
    status: SignalStatus,
    exit_price: float,
    pnl: dict,
) -> None:
    """Send final trade result to desk + portfolio channels."""
    duration_str = ""
    if position.entry_hit_at:
        now = datetime.now(timezone.utc)
        mins = int((now - position.entry_hit_at).total_seconds() / 60)
        hours, mins = divmod(mins, 60)
        duration_str = f"\u23f1 Duration: {hours}h {mins}m\n"

    pnl_display = format_pnl_display(
        position.symbol, pnl["pnl_pips"], pnl["pnl_dollars"]
    )
    pnl_r = pnl.get("pnl_r", 0)

    if status == SignalStatus.TP3_HIT:
        header = f"\U0001f3c6 <b>FULL WIN \u2014 {position.symbol} {position.action}</b>"
        outcome_line = f"\U0001f3af All 3 targets hit\n\U0001f4b0 <b>{pnl_display}</b>\n\U0001f4c8 <b>+{pnl_r}R</b>"
    elif status == SignalStatus.SL_HIT:
        if position.sl_moved_to_be:
            header = f"\u2696\ufe0f <b>BREAKEVEN \u2014 {position.symbol} {position.action}</b>"
            outcome_line = f"\U0001f6d1 SL hit at breakeven after TP1\n\U0001f4b0 <b>Partial profit secured</b>"
        else:
            header = f"\u274c <b>STOPPED OUT \u2014 {position.symbol} {position.action}</b>"
            outcome_line = f"\U0001f6d1 Stop loss hit\n\U0001f4b8 <b>{pnl_display}</b>\n\U0001f4c9 <b>{pnl_r}R</b>"
    else:
        header = f"\u23f0 <b>EXPIRED \u2014 {position.symbol} {position.action}</b>"
        outcome_line = "Entry price never reached within 24h"

    msg = (
        f"{header}\n\n"
        f"\U0001f4ca <b>{position.symbol}</b>  |  {position.action}\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        f"\U0001f4b0 Entry: <code>{position.entry_price:.5g}</code>\n"
        f"\U0001f6aa Exit: <code>{exit_price:.5g}</code>\n"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        f"{outcome_line}\n"
        f"{duration_str}"
        f"\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\u2501\n"
        f"<i>NovaFX Simulated P&L \u2014 not financial advice</i>"
    )

    await _send_to_desk(position.desk, msg)

    # Mirror to portfolio
    portfolio_id = getattr(settings, "TG_PORTFOLIO", None)
    if portfolio_id and settings.TELEGRAM_BOT_TOKEN:
        await _post_to_channel(portfolio_id, msg, settings.TELEGRAM_BOT_TOKEN)


async def _send_to_desk(desk: str | None, message: str) -> None:
    if not desk or not settings.TELEGRAM_BOT_TOKEN:
        return
    chat_id = getattr(settings, desk, None)
    if chat_id:
        await _post_to_channel(chat_id, message, settings.TELEGRAM_BOT_TOKEN)

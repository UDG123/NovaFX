"""
Trade outcome monitoring for NovaFX.

After a signal fires, this module:
1. Saves it as an open position in PostgreSQL
2. Polls live prices every 60 seconds via yfinance / CCXT
3. Detects TP1 / TP2 / TP3 / SL hits
4. Posts result back to the same Telegram desk channel
5. Updates running P&L stats
"""
import enum
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from sqlalchemy import (
    Boolean, Column, DateTime, Enum, Float,
    Integer, String, func, select
)
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import AsyncSessionLocal, Base

logger = logging.getLogger(__name__)


class SignalStatus(str, enum.Enum):
    PENDING   = "PENDING"    # Signal fired, waiting for entry price
    OPEN      = "OPEN"       # Entry hit, trade active
    TP1_HIT   = "TP1_HIT"   # 1/3 closed at TP1, SL moved to breakeven
    TP2_HIT   = "TP2_HIT"   # 2/3 closed at TP2
    TP3_HIT   = "TP3_HIT"   # Fully closed at TP3 — full win
    SL_HIT    = "SL_HIT"    # Stop loss triggered — loss or breakeven
    EXPIRED   = "EXPIRED"   # Entry never reached within 24h


class TradePosition(Base):
    __tablename__ = "trade_positions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    signal_id       = Column(String(50), unique=True, index=True, nullable=False)
    symbol          = Column(String(20), nullable=False, index=True)
    desk            = Column(String(20))           # TG_DESK1 etc
    action          = Column(String(4), nullable=False)   # BUY / SELL
    entry_price     = Column(Float, nullable=False)
    stop_loss       = Column(Float, nullable=False)
    current_sl      = Column(Float)                # Updated to BE after TP1
    tp1             = Column(Float, nullable=False)
    tp2             = Column(Float, nullable=False)
    tp3             = Column(Float, nullable=False)
    risk_amount     = Column(Float, default=100.0)
    risk_reward     = Column(Float)
    timeframe       = Column(String(10))
    indicator       = Column(String(200))

    status          = Column(
        Enum(SignalStatus),
        default=SignalStatus.PENDING,
        nullable=False,
        index=True
    )

    sl_moved_to_be  = Column(Boolean, default=False)

    # Timestamps
    created_at      = Column(DateTime(timezone=True),
                             default=lambda: datetime.now(timezone.utc))
    entry_hit_at    = Column(DateTime(timezone=True))
    tp1_hit_at      = Column(DateTime(timezone=True))
    tp2_hit_at      = Column(DateTime(timezone=True))
    tp3_hit_at      = Column(DateTime(timezone=True))
    sl_hit_at       = Column(DateTime(timezone=True))
    closed_at       = Column(DateTime(timezone=True))
    expires_at      = Column(DateTime(timezone=True))

    # Outcome
    exit_price      = Column(Float)
    pnl_pips        = Column(Float)
    pnl_dollars     = Column(Float)
    duration_minutes = Column(Integer)

    # Telegram message IDs for editing/replying
    signal_message_id = Column(Integer)

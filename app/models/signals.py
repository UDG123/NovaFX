from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class IncomingSignal(BaseModel):
    symbol: str
    action: Literal["BUY", "SELL"]
    price: float
    timeframe: str = "15m"
    source: str = "tradingview"
    secret: Optional[str] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    indicator: Optional[str] = None


class ProcessedSignal(BaseModel):
    symbol: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    position_size: float
    risk_amount: float
    timeframe: str
    source: str
    indicator: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

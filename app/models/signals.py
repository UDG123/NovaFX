from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class IncomingSignal(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    action: Literal["BUY", "SELL"]
    price: float = Field(..., gt=0)
    timeframe: str = Field(default="15m", pattern=r"^\d+[mhd]$")
    source: str = Field(default="tradingview", min_length=1, max_length=50)
    secret: Optional[str] = Field(default=None, exclude=True)
    sl: Optional[float] = Field(default=None, gt=0)
    tp: Optional[float] = Field(default=None, gt=0)
    indicator: Optional[str] = Field(default=None, max_length=200)
    confluence_count: Optional[int] = Field(default=None, ge=0)

    @field_validator("symbol")
    @classmethod
    def symbol_alphanumeric(cls, v: str) -> str:
        cleaned = v.replace("/", "").replace("-", "")
        if not cleaned.isalnum():
            raise ValueError("symbol must be alphanumeric (may contain / or -)")
        return v


class ProcessedSignal(BaseModel):
    symbol: str
    action: str
    entry_price: float = Field(..., gt=0)
    stop_loss: float = Field(..., gt=0)
    take_profit: float = Field(..., gt=0)
    risk_reward: float = Field(..., ge=0)
    position_size: float = Field(..., ge=0)
    risk_amount: float = Field(..., ge=0)
    timeframe: str
    source: str
    indicator: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

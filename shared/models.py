"""Shared Pydantic models for all NovaFX services."""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class SignalAction(str, Enum):
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    HOLD = "hold"


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    FOREX = "forex"
    STOCKS = "stocks"
    OPTIONS = "options"
    FUTURES = "futures"
    COMMODITIES = "commodities"


class Signal(BaseModel):
    signal_id: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    source: str
    action: SignalAction
    symbol: str
    asset_class: AssetClass
    confidence: float = Field(ge=0.0, le=1.0)
    price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[list[float]] = None
    timeframe: str
    strategy: str
    metadata: dict = Field(default_factory=dict)


class ConfluenceResult(BaseModel):
    symbol: str
    asset_class: AssetClass
    consensus_action: SignalAction
    weighted_confidence: float = Field(ge=0.0, le=1.0)
    contributing_signals: list[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

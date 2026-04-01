from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class Trade(BaseModel):
    symbol: str
    action: Literal["BUY", "SELL"]
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    pnl_pct: float
    commission_pct: float
    net_pnl_pct: float
    result: Literal["WIN", "LOSS"]


class PhaseResult(BaseModel):
    phase: str
    symbol: str
    strategy: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_pnl_pct: float = 0.0
    total_commission_pct: float = 0.0
    net_pnl_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    mtf_filtered: int = 0
    trades: list[Trade] = Field(default_factory=list)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None


class BacktestSummary(BaseModel):
    id: Optional[int] = None
    strategy_name: str
    composite_score: float
    backtest_win_rate: float
    forward_win_rate: Optional[float] = None
    total_trades: int
    symbols_tested: str
    ran_at: datetime

"""
Realistic slippage and fill modeling for NovaFX backtester.

Models 4 components of execution cost:
  1. Base spread (asset-specific, wider for alts)
  2. Volatility adjustment (higher vol = wider spread)
  3. Market impact (square-root model of trade size vs volume)
  4. Time-of-day adjustment (session-dependent spread widening)
"""
import math
from dataclasses import dataclass, field

import numpy as np

# Base spreads in basis points per asset
BASE_SPREADS_BPS = {
    # Crypto
    "BTC-USD": 3.0,
    "ETH-USD": 4.0,
    "BNB-USD": 6.0,
    "SOL-USD": 8.0,
    "XRP-USD": 7.0,
    # Forex majors
    "EURUSD": 1.0,
    "GBPUSD": 1.2,
    "USDJPY": 1.0,
    "AUDUSD": 1.5,
    "USDCAD": 1.5,
    "USDCHF": 1.5,
    "NZDUSD": 2.0,
    "EURGBP": 1.5,
    # Stocks
    "AAPL": 1.0,
    "MSFT": 1.0,
    "NVDA": 2.0,
    "TSLA": 2.0,
    "SPY": 0.5,
    "QQQ": 0.5,
    # Commodities
    "XAUUSD": 2.5,
    "XAGUSD": 4.0,
}

# Time-of-day spread multipliers (UTC hours)
DEFAULT_TIME_ADJUSTMENTS = {
    0: 0.8, 1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8,   # Asia quiet
    5: 0.9, 6: 1.0,                               # Asia active
    7: 1.3, 8: 1.3,                               # London open
    9: 1.0, 10: 1.0, 11: 1.0, 12: 1.0,           # London session
    13: 1.5, 14: 1.5,                             # NY open overlap
    15: 1.3, 16: 1.3,                             # NY session
    17: 1.5,                                       # Peak volatility
    18: 1.2, 19: 1.0, 20: 1.0, 21: 0.9,          # NY wind-down
    22: 0.9, 23: 0.8,                             # Transition
}


@dataclass
class SlippageRecord:
    """Record of slippage applied to a single trade."""
    intended_price: float
    actual_price: float
    slippage_pct: float
    spread_component: float
    volatility_component: float
    impact_component: float
    time_component: float
    direction: str


@dataclass
class SlippageStats:
    """Aggregate slippage statistics across trades."""
    total_cost_pct: float = 0.0
    avg_slippage_pct: float = 0.0
    max_slippage_pct: float = 0.0
    n_trades: int = 0
    records: list[SlippageRecord] = field(default_factory=list)
    by_hour: dict = field(default_factory=dict)

    def add(self, record: SlippageRecord, hour: int | None = None):
        self.records.append(record)
        self.total_cost_pct += record.slippage_pct
        self.n_trades += 1
        self.avg_slippage_pct = self.total_cost_pct / self.n_trades
        self.max_slippage_pct = max(self.max_slippage_pct, record.slippage_pct)
        if hour is not None:
            if hour not in self.by_hour:
                self.by_hour[hour] = {"count": 0, "total": 0.0}
            self.by_hour[hour]["count"] += 1
            self.by_hour[hour]["total"] += record.slippage_pct

    def rr_impact(self, intended_rr: float) -> float:
        """Return actual R:R after average slippage reduces TP distance."""
        if self.avg_slippage_pct == 0 or intended_rr <= 0:
            return intended_rr
        # Slippage reduces reward side proportionally
        return intended_rr * (1 - self.avg_slippage_pct / 100 * 2)


class SlippageModel:
    """Realistic slippage model with spread, volatility, impact, and time components."""

    def __init__(self, base_spread_bps: float | None = None,
                 impact_coefficient: float = 10.0,
                 time_adjustments: dict | None = None,
                 symbol: str = ""):
        if base_spread_bps is not None:
            self.base_spread_bps = base_spread_bps
        else:
            self.base_spread_bps = BASE_SPREADS_BPS.get(symbol, 5.0)
        self.impact_coefficient = impact_coefficient
        self.time_adjustments = time_adjustments or DEFAULT_TIME_ADJUSTMENTS
        self.symbol = symbol
        self.stats = SlippageStats()

    def calculate_slippage(self, trade_size_usd: float = 1000.0,
                           avg_volume_usd: float = 1e8,
                           volatility: float = 0.01,
                           hour_utc: int | None = None,
                           direction: str = "BUY") -> tuple[float, SlippageRecord]:
        """Calculate slippage as decimal fraction (0.001 = 0.1%).

        Returns (slippage_decimal, SlippageRecord).
        """
        # 1. Base spread: half-spread cost (we cross the spread)
        spread = self.base_spread_bps / 10000 / 2

        # 2. Volatility adjustment: higher vol widens effective spread
        # Normalize: 1% vol = 1x, 2% vol = 1.4x, 3% vol = 1.7x
        vol_mult = math.sqrt(max(volatility, 0.001) / 0.01)
        vol_component = spread * (vol_mult - 1) * 0.5

        # 3. Market impact: square-root model
        # impact = coefficient * sqrt(trade_size / avg_volume)
        if avg_volume_usd > 0:
            participation = trade_size_usd / avg_volume_usd
            impact = self.impact_coefficient * math.sqrt(participation) / 10000
        else:
            impact = 0.0

        # 4. Time-of-day adjustment
        if hour_utc is not None:
            time_mult = self.time_adjustments.get(hour_utc, 1.0)
        else:
            time_mult = 1.0
        time_component = spread * (time_mult - 1)

        total_slippage = spread + vol_component + impact + time_component
        total_slippage = max(0.0, total_slippage)

        record = SlippageRecord(
            intended_price=0.0,  # Filled by apply_slippage
            actual_price=0.0,
            slippage_pct=total_slippage * 100,
            spread_component=spread * 100,
            volatility_component=vol_component * 100,
            impact_component=impact * 100,
            time_component=time_component * 100,
            direction=direction,
        )

        return total_slippage, record

    def apply_slippage(self, entry_price: float, direction: str,
                       slippage: float, record: SlippageRecord | None = None,
                       hour_utc: int | None = None) -> float:
        """Return adjusted entry price after slippage.

        BUY: price moves up (worse fill)
        SELL: price moves down (worse fill)
        """
        if direction == "BUY":
            actual = entry_price * (1 + slippage)
        else:
            actual = entry_price * (1 - slippage)

        if record is not None:
            record.intended_price = entry_price
            record.actual_price = actual
            self.stats.add(record, hour_utc)

        return actual

    def calculate_and_apply(self, entry_price: float, direction: str,
                            trade_size_usd: float = 1000.0,
                            avg_volume_usd: float = 1e8,
                            volatility: float = 0.01,
                            hour_utc: int | None = None) -> tuple[float, SlippageRecord]:
        """Convenience: calculate slippage and apply in one call."""
        slippage, record = self.calculate_slippage(
            trade_size_usd, avg_volume_usd, volatility, hour_utc, direction
        )
        actual = self.apply_slippage(entry_price, direction, slippage, record, hour_utc)
        return actual, record

    def report(self) -> str:
        """Generate slippage report string."""
        s = self.stats
        lines = [
            f"  Total trades: {s.n_trades}",
            f"  Avg slippage: {s.avg_slippage_pct:.4f}%",
            f"  Max slippage: {s.max_slippage_pct:.4f}%",
            f"  Total cost:   {s.total_cost_pct:.4f}%",
        ]
        if s.by_hour:
            lines.append("  By hour (UTC):")
            for h in sorted(s.by_hour):
                d = s.by_hour[h]
                avg = d["total"] / d["count"] if d["count"] else 0
                lines.append(f"    {h:02d}:00  n={d['count']:>3}  avg={avg:.4f}%")
        return "\n".join(lines)

"""
Realistic slippage and fill modeling for NovaFX backtester.

Models 4 components of execution cost:
  1. Base spread (asset-specific, wider for alts)
  2. Volatility adjustment (higher vol = wider spread)
  3. Market impact (square-root model of trade size vs volume)
  4. Time-of-day adjustment (session-dependent spread widening)

Includes forex volume synthesis (Yahoo forex volume is unreliable)
and clamped market impact to prevent blowups on thin data.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

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


FOREX_CODES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD", "SEK", "NOK"]

# Default daily volumes for forex (very liquid)
FOREX_DEFAULT_VOLUME_USD = {
    "major": 50_000_000_000,   # EUR/USD, GBP/USD — $50B/day
    "minor": 10_000_000_000,   # EUR/GBP, AUD/NZD — $10B/day
    "exotic": 1_000_000_000,   # USD/TRY, USD/ZAR — $1B/day
}

FOREX_MAJORS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD"]


def is_forex(symbol: str) -> bool:
    """Detect if symbol is a forex pair."""
    s = symbol.upper().replace("-", "").replace("/", "").replace("_", "")
    return sum(1 for c in FOREX_CODES if c in s) >= 2


def get_forex_volume_usd(symbol: str, timeframe_hours: float = 1.0) -> float:
    """Get reasonable forex volume estimate per bar.

    Major pairs: $50B/day, minor: $10B/day, exotic: $1B/day.
    Scaled to the requested timeframe.
    """
    s_clean = symbol.upper().replace("-", "").replace("/", "").replace("_", "").replace("=X", "")

    if any(m in s_clean or s_clean in m for m in FOREX_MAJORS):
        daily_vol = FOREX_DEFAULT_VOLUME_USD["major"]
    elif any(c in symbol.upper() for c in ["EUR", "GBP", "USD", "JPY"]):
        daily_vol = FOREX_DEFAULT_VOLUME_USD["minor"]
    else:
        daily_vol = FOREX_DEFAULT_VOLUME_USD["exotic"]

    bars_per_day = 24.0 / timeframe_hours
    return daily_vol / bars_per_day


def fix_forex_volume(volume: pd.Series, close: pd.Series, symbol: str) -> pd.Series:
    """Replace unreliable Yahoo forex volume with synthetic estimate.

    Non-forex: clamp raw volume to min 1000.
    Forex: use get_forex_volume_usd() scaled by close price.
    """
    if not is_forex(symbol):
        return volume.clip(lower=1000)

    hourly_usd = get_forex_volume_usd(symbol, timeframe_hours=1.0)
    return (hourly_usd / close).clip(lower=1000)


def safe_market_impact(trade_size: float, avg_volume_usd: float) -> float:
    """Square-root market impact with floor/cap.

    Returns impact as decimal fraction.
    Floor: 0.1 bps (0.00001), Cap: 50 bps (0.005).
    Volume floor: $1M to prevent blowups on thin data.
    """
    avg_volume_usd = max(avg_volume_usd, 1_000_000)
    participation = trade_size / avg_volume_usd
    impact = 0.1 * math.sqrt(participation)
    return max(0.00001, min(0.005, impact))


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
                           direction: str = "BUY",
                           symbol: str | None = None) -> tuple[float, SlippageRecord]:
        """Calculate slippage as decimal fraction (0.001 = 0.1%).

        Uses multiplicative model: (spread + impact) * vol_adj * time_adj.
        Forex volumes auto-corrected via get_forex_volume_usd().
        Final clamp for forex: 0.5-10 pips (0.00005 - 0.001).

        Returns (slippage_decimal, SlippageRecord).
        """
        sym = symbol or self.symbol

        # Forex fix: override broken Yahoo volume with realistic estimate
        if sym and is_forex(sym):
            avg_volume_usd = get_forex_volume_usd(sym, timeframe_hours=1.0)

        # Clamp minimum volume to prevent blowup
        avg_volume_usd = max(avg_volume_usd, 1_000_000)

        # 1. Base spread (full spread in decimal)
        spread = self.base_spread_bps / 10000

        # 2. Market impact: clamped square-root model
        impact = safe_market_impact(trade_size_usd, avg_volume_usd)

        # 3. Volatility adjustment (multiplicative)
        vol_adjustment = 1.0 + max(volatility, 0.0) * 5

        # 4. Time-of-day adjustment (multiplicative)
        if hour_utc is not None:
            tod_mult = self.time_adjustments.get(hour_utc, 1.0)
        else:
            tod_mult = 1.0

        # Total slippage: multiplicative model
        total_slippage = (spread + impact) * vol_adjustment * tod_mult

        # Final forex clamp: 0.5 - 10 pips (0.00005 - 0.001)
        if sym and is_forex(sym):
            total_slippage = max(0.00005, min(0.001, total_slippage))

        # Decompose for logging
        spread_part = spread * vol_adjustment * tod_mult
        impact_part = impact * vol_adjustment * tod_mult
        vol_part = (spread + impact) * (vol_adjustment - 1.0) * tod_mult
        time_part = (spread + impact) * vol_adjustment * (tod_mult - 1.0) if tod_mult != 1.0 else 0.0

        record = SlippageRecord(
            intended_price=0.0,  # Filled by apply_slippage
            actual_price=0.0,
            slippage_pct=total_slippage * 100,
            spread_component=spread_part * 100,
            volatility_component=vol_part * 100,
            impact_component=impact_part * 100,
            time_component=time_part * 100,
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
                            hour_utc: int | None = None,
                            symbol: str | None = None) -> tuple[float, SlippageRecord]:
        """Convenience: calculate slippage and apply in one call."""
        slippage, record = self.calculate_slippage(
            trade_size_usd, avg_volume_usd, volatility, hour_utc, direction,
            symbol=symbol,
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

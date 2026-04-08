"""
Volume handling for forex and low-volume assets.

Forex volume from Yahoo Finance is notional and unreliable.
This module provides:
1. Asset-type detection (granular: forex_major/minor/exotic, crypto tiers, stocks)
2. Synthetic volume estimation using True Range as activity proxy
3. Market impact clamping with per-asset volume floors
4. Asset-aware time-of-day slippage multipliers
"""

import math
from typing import Optional

import numpy as np
import pandas as pd

# Asset classification patterns
FOREX_PATTERNS = [
    "USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD",
    "SEK", "NOK", "DKK", "SGD", "HKD", "MXN", "ZAR", "TRY",
    "PLN", "CZK", "HUF", "ILS", "THB", "INR", "KRW", "TWD",
]

CRYPTO_PATTERNS = [
    "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "MATIC", "DOT", "AVAX",
]

# Default daily volumes (USD notional) for impact estimation
DEFAULT_VOLUMES = {
    "forex_major": 50_000_000_000,   # EUR/USD, GBP/USD, USD/JPY — $50B daily
    "forex_minor": 10_000_000_000,   # EUR/GBP, AUD/NZD — $10B daily
    "forex_exotic": 1_000_000_000,   # USD/TRY, USD/ZAR — $1B daily
    "crypto_large": 5_000_000_000,   # BTC, ETH — $5B daily
    "crypto_medium": 500_000_000,    # SOL, BNB — $500M daily
    "crypto_small": 50_000_000,      # Altcoins — $50M daily
    "stock_large": 1_000_000_000,    # AAPL, MSFT — $1B daily
    "stock_medium": 100_000_000,     # Mid-cap — $100M daily
    "stock_small": 10_000_000,       # Small-cap — $10M daily
    "default": 100_000_000,          # Fallback — $100M daily
}

# Market impact limits (as fraction of price)
IMPACT_FLOOR = 0.00001   # 0.1 bps minimum
IMPACT_CAP = 0.01        # 100 bps maximum (1%)

# Base spreads by asset type (in decimal, full spread)
BASE_SPREADS = {
    "forex_major": 0.00005,    # 0.5 pips
    "forex_minor": 0.00015,    # 1.5 pips
    "forex_exotic": 0.0005,    # 5 pips
    "crypto_large": 0.0003,    # 3 bps
    "crypto_medium": 0.0006,   # 6 bps
    "crypto_small": 0.001,     # 10 bps
    "stock_large": 0.0001,     # 1 bp
    "stock_medium": 0.0003,    # 3 bps
    "stock_small": 0.001,      # 10 bps
}


def classify_asset(symbol: str) -> str:
    """Classify asset type from symbol.

    Returns one of: forex_major, forex_minor, forex_exotic,
    crypto_large, crypto_medium, crypto_small,
    stock_large, stock_medium, stock_small.
    """
    s = symbol.upper().replace("-", "").replace("/", "").replace("_", "").replace("=X", "")

    # Forex detection: 2+ currency codes
    forex_count = sum(1 for p in FOREX_PATTERNS if p in s)
    if forex_count >= 2:
        g7 = ["EUR", "USD", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]
        g7_count = sum(1 for m in g7 if m in s)
        if g7_count >= 2:
            # Major if both are G7 majors (EUR, USD, GBP, JPY)
            top4 = ["EUR", "USD", "GBP", "JPY"]
            top4_count = sum(1 for m in top4 if m in s)
            if top4_count >= 2:
                return "forex_major"
            return "forex_minor"
        else:
            return "forex_exotic"

    # Crypto detection
    for pattern in CRYPTO_PATTERNS[:3]:  # BTC, ETH, BNB
        if pattern in s:
            return "crypto_large"
    for pattern in CRYPTO_PATTERNS[3:6]:  # SOL, XRP, ADA
        if pattern in s:
            return "crypto_medium"
    for pattern in CRYPTO_PATTERNS[6:]:
        if pattern in s:
            return "crypto_small"

    # Large-cap stocks
    large_caps = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "SPY", "QQQ"}
    if s in large_caps:
        return "stock_large"

    return "stock_medium"


def is_forex(symbol: str) -> bool:
    """Check if symbol is a forex pair."""
    return classify_asset(symbol).startswith("forex")


def get_default_volume(symbol: str) -> float:
    """Get default daily volume (USD) for an asset."""
    asset_type = classify_asset(symbol)
    return DEFAULT_VOLUMES.get(asset_type, DEFAULT_VOLUMES["default"])


def estimate_synthetic_volume(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    symbol: str,
    timeframe_hours: float = 1.0,
) -> pd.Series:
    """Estimate synthetic volume for forex using True Range as activity proxy.

    Higher-volatility bars get proportionally more volume, reflecting
    the real-world correlation between activity and price range.
    """
    daily_vol_usd = get_default_volume(symbol)
    bars_per_day = 24.0 / timeframe_hours
    base_vol_per_bar = daily_vol_usd / bars_per_day

    # True Range as activity proxy
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    # Normalize TR to create volume multiplier (mean = 1.0)
    tr_ma = tr.rolling(20, min_periods=1).mean()
    vol_multiplier = (tr / tr_ma).clip(0.5, 2.0)

    # Convert USD volume to asset units
    synthetic_vol = base_vol_per_bar * vol_multiplier / close
    return synthetic_vol.fillna(base_vol_per_bar / close.mean())


def normalize_volume(
    volume: pd.Series,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    symbol: str,
    min_volume_usd: float = 1_000_000,
    timeframe_hours: float = 1.0,
) -> pd.Series:
    """Normalize volume, replacing near-zero forex volume with synthetic estimates."""
    vol_usd = volume * close
    median_vol_usd = vol_usd.median()

    if is_forex(symbol) or median_vol_usd < min_volume_usd:
        return estimate_synthetic_volume(close, high, low, symbol, timeframe_hours)

    # Volume looks OK, just clamp minimum
    min_vol = min_volume_usd / close
    return volume.clip(lower=min_vol)


def calculate_market_impact(
    trade_size_usd: float,
    volume: pd.Series,
    close: pd.Series,
    symbol: str,
    impact_coefficient: float = 0.1,
) -> pd.Series:
    """Calculate market impact with proper clamping (series-based).

    Uses square-root market impact model:
        impact = coefficient * sqrt(trade_size / volume_usd)
    """
    vol_usd = volume * close
    min_vol = get_default_volume(symbol) / 24  # Minimum hourly volume
    vol_usd_safe = vol_usd.clip(lower=min_vol)

    impact = impact_coefficient * np.sqrt(trade_size_usd / vol_usd_safe)
    return impact.clip(lower=IMPACT_FLOOR, upper=IMPACT_CAP)


def get_time_of_day_multiplier(hour: pd.Series, asset_type: str) -> pd.Series:
    """Get time-of-day slippage multiplier, asset-type aware."""
    if asset_type.startswith("forex"):
        multipliers = {
            0: 1.3, 1: 1.3, 2: 1.3, 3: 1.2, 4: 1.1,
            5: 1.0, 6: 1.0,
            7: 0.9, 8: 0.8, 9: 0.8, 10: 0.8, 11: 0.8,
            12: 0.9,
            13: 0.8, 14: 0.8, 15: 0.8,  # London + NY overlap = best
            16: 0.9, 17: 1.0, 18: 1.1,
            19: 1.2, 20: 1.2, 21: 1.3, 22: 1.4, 23: 1.4,
        }
    elif asset_type.startswith("crypto"):
        multipliers = {
            0: 1.2, 1: 1.2, 2: 1.2, 3: 1.1, 4: 1.1,
            5: 1.0, 6: 1.0, 7: 1.0,
            8: 0.9, 9: 0.9, 10: 0.9, 11: 0.9, 12: 0.9,
            13: 0.9, 14: 0.8, 15: 0.8,  # US active = best crypto liquidity
            16: 0.9, 17: 1.0, 18: 1.0, 19: 1.0,
            20: 1.1, 21: 1.1, 22: 1.2, 23: 1.2,
        }
    else:
        # Stocks: only good during market hours (14-21 UTC for US)
        multipliers = {h: 1.5 for h in range(24)}  # Default: bad
        for h in [14, 15, 16, 17, 18, 19, 20]:
            multipliers[h] = {14: 1.2, 15: 0.9, 16: 0.8, 17: 0.8,
                              18: 0.8, 19: 0.9, 20: 1.0}.get(h, 1.0)
        multipliers[21] = 1.3  # After hours

    return hour.map(multipliers).fillna(1.0)


def calculate_slippage_series(
    trade_size_usd: float,
    volume: pd.Series,
    close: pd.Series,
    volatility: pd.Series,
    symbol: str,
    hour_utc: Optional[pd.Series] = None,
) -> pd.Series:
    """Calculate total slippage for every bar in a series.

    Combines spread, impact, volatility adjustment, and time-of-day.
    """
    asset_type = classify_asset(symbol)
    base_spread = BASE_SPREADS.get(asset_type, 0.0005)

    # Market impact (series)
    impact = calculate_market_impact(trade_size_usd, volume, close, symbol)

    # Volatility adjustment (multiplicative, capped at 1.5x at 5% vol)
    vol_adjustment = 1.0 + volatility.clip(0, 0.05) * 10

    # Time-of-day adjustment
    if hour_utc is not None:
        tod_mult = get_time_of_day_multiplier(hour_utc, asset_type)
    else:
        tod_mult = 1.0

    total = (base_spread + impact) * vol_adjustment * tod_mult
    return total.clip(lower=IMPACT_FLOOR, upper=IMPACT_CAP)


class VolumeHandler:
    """Unified volume handling for backtesting.

    Usage:
        handler = VolumeHandler(symbol='EURUSD', timeframe='1h')
        norm_vol = handler.normalize(data['volume'], data['close'], data['high'], data['low'])
        slippage = handler.get_slippage(10000, data)
    """

    def __init__(self, symbol: str, timeframe: str = "1h"):
        self.symbol = symbol
        self.asset_type = classify_asset(symbol)
        self.is_fx = is_forex(symbol)

        tf_map = {"1m": 1/60, "5m": 5/60, "15m": 0.25, "30m": 0.5,
                  "1h": 1, "4h": 4, "1d": 24, "1w": 168}
        self.timeframe_hours = tf_map.get(timeframe.lower(), 1.0)
        self.default_volume = get_default_volume(symbol)

    def normalize(self, volume: pd.Series, close: pd.Series,
                  high: pd.Series, low: pd.Series) -> pd.Series:
        """Normalize volume, using synthetic if needed."""
        return normalize_volume(
            volume, close, high, low,
            self.symbol,
            timeframe_hours=self.timeframe_hours,
        )

    def get_impact(self, trade_size_usd: float, volume: pd.Series,
                   close: pd.Series) -> pd.Series:
        """Calculate market impact (series)."""
        return calculate_market_impact(
            trade_size_usd, volume, close, self.symbol,
        )

    def get_slippage(self, trade_size_usd: float,
                     data: pd.DataFrame,
                     direction: str = "buy") -> pd.Series:
        """Calculate total slippage per bar.

        Args:
            trade_size_usd: Trade size in USD
            data: DataFrame with columns: close, high, low, volume
            direction: 'buy' or 'sell' (currently symmetric)
        """
        vol = self.normalize(
            data["volume"], data["close"],
            data["high"], data["low"],
        )

        # ATR-based volatility (pure pandas, no external deps)
        prev_close = data["close"].shift(1)
        tr = pd.concat([
            data["high"] - data["low"],
            (data["high"] - prev_close).abs(),
            (data["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.ewm(alpha=1.0 / 14, adjust=False).mean()
        volatility = atr / data["close"]

        # Get hour if available
        if isinstance(data.index, pd.DatetimeIndex):
            hour = pd.Series(data.index.hour, index=data.index)
        elif "timestamp" in data.columns:
            hour = data["timestamp"].dt.hour
        else:
            hour = None

        return calculate_slippage_series(
            trade_size_usd, vol, data["close"], volatility,
            self.symbol, hour,
        )

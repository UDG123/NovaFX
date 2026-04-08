"""
Weighted signal scoring for NovaFX.

Replaces binary emit/block with a 0-1 confidence score computed from
multiple factors. Only signals above a configurable threshold are emitted.

Supports:
  - Per-factor weights (default + per-asset/per-strategy overrides)
  - Confidence-based position sizing (higher confidence = larger position)
  - Adaptive weight learning from trade outcomes
  - Weight persistence to JSON for production use
"""
import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WEIGHTS = {
    "trend_strength": 0.25,
    "momentum_confirm": 0.20,
    "vol_surge": 0.15,
    "atr_percentile": 0.15,
    "multi_strat_agree": 0.15,
    "time_filter": 0.10,
}

# Weights learned from backtest optimization (backtest_scored.py results)
LEARNED_WEIGHTS = {
    "trend_strength": 0.24,
    "momentum_confirm": 0.20,
    "vol_surge": 0.13,
    "atr_percentile": 0.15,
    "multi_strat_agree": 0.18,
    "time_filter": 0.10,
}

# Hours considered high-noise (NY open volatility spike)
HIGH_NOISE_HOURS = {17, 18}

# Weight clamp bounds for learning
WEIGHT_MIN = 0.05
WEIGHT_MAX = 0.40

# Position sizing tiers
SIZING_TIERS = [
    (0.85, 1.50),  # >= 0.85 confidence: 1.5x base risk
    (0.75, 1.25),  # >= 0.75 confidence: 1.25x base risk
    (0.65, 1.00),  # >= 0.65 confidence: 1.0x base risk (default)
]


def _calc_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


class SignalScorer:
    """Weighted multi-factor signal confidence scorer."""

    def __init__(self, weights: dict | None = None, threshold: float = 0.65,
                 asset_overrides: dict | None = None,
                 strategy_overrides: dict | None = None):
        self.weights = weights if weights is not None else deepcopy(DEFAULT_WEIGHTS)
        self.threshold = threshold
        self.asset_overrides = asset_overrides or {}
        self.strategy_overrides = strategy_overrides or {}

        # Trade outcome tracking for weight learning
        self._trade_log: list[dict] = []
        self._learn_interval = 50

    def _get_weights(self, asset_class: str = "", strategy: str = "") -> dict:
        """Get effective weights, applying overrides in order."""
        w = deepcopy(self.weights)
        if asset_class in self.asset_overrides:
            w.update(self.asset_overrides[asset_class])
        if strategy in self.strategy_overrides:
            w.update(self.strategy_overrides[strategy])
        # Normalize to sum to 1.0
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
        return w

    def score(self, ctx: dict) -> float:
        """Compute 0-1 confidence score from signal context.

        These factors are designed to add information BEYOND the existing
        pipeline filters (which already gate on trend direction, regime,
        volume threshold, and choppiness). Each factor here measures the
        *degree* or *quality* of a condition, not just its presence.

        ctx keys:
            direction: "BUY" or "SELL"
            df: pd.DataFrame with OHLCV
            strategy: str (strategy that fired)
            asset_class: str
            sl_dist: float (stop loss distance)
            tp_dist: float (take profit distance)
            hour_utc: int | None (signal hour in UTC)
            volume_ratio: float (current vol / avg vol)
            n_strategies_agree: int (how many strategies fired same direction)
        """
        weights = self._get_weights(
            ctx.get("asset_class", ""),
            ctx.get("strategy", ""),
        )
        factors = {}

        df = ctx.get("df")
        direction = ctx.get("direction", "BUY")

        # 1. Trend strength: how far EMA20 is from EMA50 (slope, not just cross)
        if df is not None and len(df) >= 50:
            close = df["close"]
            ema20 = _calc_ema(close, 20)
            ema50 = _calc_ema(close, 50)
            current_price = close.iloc[-1]
            e20 = ema20.iloc[-1]
            e50 = ema50.iloc[-1]
            if not (np.isnan(e20) or np.isnan(e50)) and current_price > 0:
                # Normalized spread: how far apart are the EMAs as % of price
                spread = abs(e20 - e50) / current_price
                # Also check EMA20 slope over last 5 bars
                e20_5 = ema20.iloc[-6] if len(ema20) >= 6 and not np.isnan(ema20.iloc[-6]) else e20
                slope = (e20 - e20_5) / current_price

                # Spread scoring: wider = stronger trend
                if spread > 0.01:       # >1% spread — very strong
                    spread_score = 1.0
                elif spread > 0.005:    # 0.5-1% — moderate
                    spread_score = 0.7
                elif spread > 0.002:    # 0.2-0.5% — weak
                    spread_score = 0.4
                else:                   # <0.2% — barely trending
                    spread_score = 0.1

                # Slope must agree with direction
                if direction == "BUY" and slope > 0:
                    factors["trend_strength"] = spread_score
                elif direction == "SELL" and slope < 0:
                    factors["trend_strength"] = spread_score
                else:
                    factors["trend_strength"] = spread_score * 0.3  # Penalize counter-slope
            else:
                factors["trend_strength"] = 0.3
        else:
            factors["trend_strength"] = 0.3

        # 2. Momentum confirmation: RSI in favorable zone
        if df is not None and len(df) >= 20:
            rsi = _calc_rsi(df["close"]).iloc[-1]
            if not np.isnan(rsi):
                if direction == "BUY":
                    if 40 <= rsi <= 65:
                        factors["momentum_confirm"] = 1.0
                    elif rsi < 40:
                        factors["momentum_confirm"] = 0.6
                    else:
                        factors["momentum_confirm"] = max(0.0, 1.0 - (rsi - 65) / 35)
                else:
                    if 35 <= rsi <= 60:
                        factors["momentum_confirm"] = 1.0
                    elif rsi > 60:
                        factors["momentum_confirm"] = 0.6
                    else:
                        factors["momentum_confirm"] = max(0.0, 1.0 - (35 - rsi) / 35)
            else:
                factors["momentum_confirm"] = 0.5
        else:
            factors["momentum_confirm"] = 0.5

        # 3. Volume surge: how much above average (graduated, not binary)
        vol_ratio = ctx.get("volume_ratio", 1.0)
        if vol_ratio >= 2.0:
            factors["vol_surge"] = 1.0
        elif vol_ratio >= 1.5:
            factors["vol_surge"] = 0.8
        elif vol_ratio >= 1.2:
            factors["vol_surge"] = 0.6
        elif vol_ratio >= 1.0:
            factors["vol_surge"] = 0.4
        else:
            factors["vol_surge"] = 0.1

        # 4. ATR percentile: mid-range volatility is best (not too quiet, not too wild)
        if df is not None and len(df) >= 50:
            close = df["close"]
            high = df["high"]
            low = df["low"]
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr_series = tr.ewm(alpha=1.0 / 14, adjust=False).mean().dropna()
            if len(atr_series) > 20:
                current_atr = atr_series.iloc[-1]
                pct_rank = float((atr_series < current_atr).sum() / len(atr_series))
                # Sweet spot: 30th-70th percentile (moderate volatility)
                if 0.30 <= pct_rank <= 0.70:
                    factors["atr_percentile"] = 1.0
                elif 0.20 <= pct_rank <= 0.80:
                    factors["atr_percentile"] = 0.6
                else:
                    factors["atr_percentile"] = 0.2  # Extreme low or high vol
            else:
                factors["atr_percentile"] = 0.5
        else:
            factors["atr_percentile"] = 0.5

        # 5. Multi-strategy agreement: more strategies = higher confidence
        n_agree = ctx.get("n_strategies_agree", 1)
        if n_agree >= 3:
            factors["multi_strat_agree"] = 1.0
        elif n_agree >= 2:
            factors["multi_strat_agree"] = 0.7
        else:
            factors["multi_strat_agree"] = 0.3

        # 6. Time filter: avoid high-noise hours
        hour = ctx.get("hour_utc")
        if hour is not None:
            if hour in HIGH_NOISE_HOURS:
                factors["time_filter"] = 0.0
            elif 8 <= hour <= 16:  # London session — best for most assets
                factors["time_filter"] = 1.0
            else:
                factors["time_filter"] = 0.5
        else:
            factors["time_filter"] = 0.5

        # Compute weighted score
        score = sum(weights.get(k, 0) * factors.get(k, 0.5) for k in weights)
        return round(min(1.0, max(0.0, score)), 4)

    def should_emit(self, ctx: dict) -> tuple[bool, float]:
        """Return (should_trade, confidence_score)."""
        confidence = self.score(ctx)
        return confidence >= self.threshold, confidence

    def position_size_multiplier(self, confidence: float) -> float:
        """Return position size multiplier based on confidence tier."""
        for min_conf, mult in SIZING_TIERS:
            if confidence >= min_conf:
                return mult
        return 0.0  # Below minimum threshold

    def record_outcome(self, ctx: dict, confidence: float, won: bool) -> None:
        """Record a trade outcome for weight learning."""
        self._trade_log.append({
            "factors": {
                k: self._evaluate_factor(k, ctx)
                for k in self.weights
            },
            "confidence": confidence,
            "won": won,
        })

        if len(self._trade_log) >= self._learn_interval:
            self.update_weights(self._trade_log)
            self._trade_log = []

    def _evaluate_factor(self, factor: str, ctx: dict) -> float:
        """Re-evaluate a single factor from context (for logging)."""
        # Build a mini-context and score just this factor
        dummy_weights = {k: (1.0 if k == factor else 0.0) for k in self.weights}
        scorer = SignalScorer(weights=dummy_weights, threshold=0.0)
        return scorer.score(ctx)

    def update_weights(self, trades: list[dict]) -> None:
        """Adjust weights toward factors that better predict winners.

        For each factor, compute the average value among winning trades
        vs losing trades. Factors with higher values in winners get
        weight increases.
        """
        if not trades:
            return

        winners = [t for t in trades if t["won"]]
        losers = [t for t in trades if not t["won"]]

        if not winners or not losers:
            return

        adjustments = {}
        for factor in self.weights:
            win_avg = np.mean([t["factors"].get(factor, 0.5) for t in winners])
            loss_avg = np.mean([t["factors"].get(factor, 0.5) for t in losers])
            # Positive delta means factor is more present in winners
            delta = win_avg - loss_avg
            adjustments[factor] = delta

        # Apply adjustments (small learning rate)
        lr = 0.1
        for factor, delta in adjustments.items():
            self.weights[factor] += lr * delta
            self.weights[factor] = max(WEIGHT_MIN, min(WEIGHT_MAX, self.weights[factor]))

        # Re-normalize
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        logger.info("Weights updated: %s", {k: round(v, 3) for k, v in self.weights.items()})

    def save_weights(self, filepath: str = "config/signal_weights.json") -> Path:
        """Persist current weights to JSON for production use."""
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.weights, f, indent=2)
        logger.info("Weights saved to %s", p)
        return p

    def load_weights(self, filepath: str = "config/signal_weights.json") -> bool:
        """Load weights from JSON. Returns True if loaded, False if file missing."""
        p = Path(filepath)
        if not p.exists():
            logger.warning("No weights file at %s, using defaults", p)
            return False
        with open(p) as f:
            loaded = json.load(f)
        self.weights = loaded
        logger.info("Weights loaded from %s: %s", p, self.weights)
        return True

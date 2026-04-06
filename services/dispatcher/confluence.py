"""
NovaFX Confluence Engine.

Weighted voting system that evaluates whether multiple independent
signal sources agree on a trade direction for a given symbol.
"""

import json
import logging
import time
from typing import Optional

import redis.asyncio as redis

from shared.models import AssetClass, ConfluenceResult, Signal, SignalAction

logger = logging.getLogger("novafx.dispatcher.confluence")

# Default source weights — higher = more trusted
DEFAULT_SOURCE_WEIGHTS: dict[str, float] = {
    "freqtrade": 1.0,
    "twelvedata-forex": 1.0,
    "alpaca-stocks": 1.0,
    "tradingview": 0.8,
    "finnhub": 0.9,
    "cryptocompare": 0.8,
}


class ConfluenceEngine:
    """Evaluate confluence across signal sources for a symbol."""

    def __init__(
        self,
        redis_client: redis.Redis,
        window_sec: int = 300,
        min_sources: int = 2,
        min_confidence: float = 0.6,
        source_weights: Optional[dict[str, float]] = None,
    ) -> None:
        self._redis = redis_client
        self._window_sec = window_sec
        self._min_sources = min_sources
        self._min_confidence = min_confidence
        self._weights = source_weights or DEFAULT_SOURCE_WEIGHTS

    def _source_group(self, source: str) -> str:
        """Extract source group prefix for deduplication.

        'freqtrade-ScalpEMA' -> 'freqtrade'
        'tradingview' -> 'tradingview'
        """
        for prefix in self._weights:
            if source.startswith(prefix):
                return prefix
        return source.split("-")[0] if "-" in source else source

    def _get_weight(self, source: str) -> float:
        """Look up weight for a source, defaulting to 0.7."""
        group = self._source_group(source)
        return self._weights.get(group, 0.7)

    async def evaluate(
        self, symbol: str, asset_class: AssetClass
    ) -> Optional[ConfluenceResult]:
        """
        Check if enough independent sources agree on a direction for `symbol`.

        Returns ConfluenceResult if consensus is reached, else None.
        """
        key = f"confluence:{symbol}"
        now = time.time()
        window_start = now - self._window_sec

        # Fetch entries within the time window
        raw_entries = await self._redis.zrangebyscore(
            key, min=window_start, max="+inf", withscores=True
        )

        if not raw_entries:
            return None

        # Deserialize and deduplicate by source group (keep latest)
        by_group: dict[str, Signal] = {}
        for raw, score in raw_entries:
            try:
                signal = Signal.model_validate_json(raw)
                group = self._source_group(signal.source)
                # Sorted set is ordered, so later entries override earlier ones
                by_group[group] = signal
            except Exception:
                logger.debug("Failed to deserialize signal entry")
                continue

        unique_sources = len(by_group)
        if unique_sources < self._min_sources:
            logger.debug(
                "Confluence %s: %d unique sources < %d required",
                symbol, unique_sources, self._min_sources,
            )
            return None

        # Weighted vote
        buy_weight = 0.0
        sell_weight = 0.0
        contributing_ids: list[str] = []

        for group, signal in by_group.items():
            weight = self._get_weight(signal.source) * signal.confidence

            if signal.action == SignalAction.BUY:
                buy_weight += weight
            elif signal.action == SignalAction.SELL:
                sell_weight += weight
            else:
                continue  # CLOSE/HOLD don't contribute to directional votes

            contributing_ids.append(signal.signal_id)

        total_weight = buy_weight + sell_weight
        if total_weight == 0:
            return None

        if buy_weight >= sell_weight:
            consensus_action = SignalAction.BUY
            winning_weight = buy_weight
        else:
            consensus_action = SignalAction.SELL
            winning_weight = sell_weight

        weighted_confidence = round(winning_weight / total_weight, 4)

        if weighted_confidence < self._min_confidence:
            logger.info(
                "Confluence %s: confidence %.2f < %.2f threshold",
                symbol, weighted_confidence, self._min_confidence,
            )
            return None

        result = ConfluenceResult(
            symbol=symbol,
            asset_class=asset_class,
            consensus_action=consensus_action,
            weighted_confidence=weighted_confidence,
            contributing_signals=contributing_ids,
        )

        logger.info(
            "CONFLUENCE HIT: %s %s (confidence=%.2f, sources=%d)",
            consensus_action.value.upper(),
            symbol,
            weighted_confidence,
            unique_sources,
        )

        return result

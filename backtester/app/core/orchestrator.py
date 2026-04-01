import logging

from backtester.app.models.backtest import PhaseResult

logger = logging.getLogger(__name__)


def composite_score(result: PhaseResult) -> float:
    """Score a PhaseResult by weighting win rate, net PnL, and drawdown.

    Score = (win_rate * 0.4) + (net_pnl_pct * 0.4) - (max_drawdown_pct * 0.2)
    """
    return (
        result.win_rate * 0.4
        + result.net_pnl_pct * 0.4
        - result.max_drawdown_pct * 0.2
    )


def pick_best_strategy(results: list[PhaseResult]) -> PhaseResult | None:
    """Return the PhaseResult with the highest composite score, or None if empty."""
    if not results:
        return None
    best = max(results, key=composite_score)
    logger.info(
        "Best strategy: %s (score=%.4f, win_rate=%.2f%%, net_pnl=%.4f%%)",
        best.strategy,
        composite_score(best),
        best.win_rate,
        best.net_pnl_pct,
    )
    return best

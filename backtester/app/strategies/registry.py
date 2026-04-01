"""Central registry of all backtester strategies.

Each strategy module must expose a ``generate_signals(df) -> list[dict]``
function that returns signal dicts with keys: action, entry_idx, exit_idx.
"""

from backtester.app.strategies import bollinger_reversion

STRATEGY_REGISTRY: dict[str, object] = {
    "BollingerBandReversion": bollinger_reversion,
}


def get_strategy(name: str):
    """Return the strategy module by name, or raise KeyError."""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(
            f"Unknown strategy '{name}'. "
            f"Available: {', '.join(STRATEGY_REGISTRY)}"
        )
    return STRATEGY_REGISTRY[name]


def list_strategies() -> list[str]:
    """Return all registered strategy names."""
    return list(STRATEGY_REGISTRY)

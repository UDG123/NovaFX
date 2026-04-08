"""
Strategy-asset rules for NovaFX.

Defines which strategies are allowed on which assets/asset classes.
Based on WFO results and MC validation.
"""

STRATEGY_ASSET_RULES = {
    "rsi_adaptive": {
        "allowed_classes": ["forex"],  # Only forex — negative OOS on all crypto
        "blocked_assets": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD"],
    },
    "rsi_divergence": {
        "allowed_regimes": ["ranging", "mean_reverting"],  # Never in trending
    },
    "bb_reversion": {
        "blocked_assets": ["ETH-USD", "SOL-USD"],  # Negative OOS from WFO
    },
    "momentum_breakout": {
        "blocked_assets": ["BTC-USD"],  # Negative OOS from WFO
    },
}


def is_strategy_allowed(strategy: str, asset: str, asset_class: str = "") -> bool:
    """Check if a strategy is allowed for a given asset."""
    rules = STRATEGY_ASSET_RULES.get(strategy)
    if rules is None:
        return True  # No rules = allowed everywhere

    # Check blocked assets
    if asset in rules.get("blocked_assets", []):
        return False

    # Check allowed classes (if specified, asset class must match)
    allowed_classes = rules.get("allowed_classes")
    if allowed_classes is not None and asset_class:
        if asset_class not in allowed_classes:
            return False

    return True

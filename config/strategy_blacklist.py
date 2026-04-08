"""
Strategy-asset blacklist based on Walk-Forward Optimization results.

These combos showed efficiency < 0.3 or negative OOS PnL across 30 windows.
Do NOT deploy with optimized params — use defaults or skip entirely.

Source: scripts/run_walk_forward.py output (2yr hourly, 30 windows)
"""

# (strategy, asset) tuples that should be blocked from production
OVERFITTING_COMBOS = [
    # rsi_adaptive: negative OOS on all crypto (weakest strategy)
    ("rsi_adaptive", "BTC-USD"),
    ("rsi_adaptive", "ETH-USD"),
    ("rsi_adaptive", "SOL-USD"),
    ("rsi_adaptive", "XRP-USD"),
    # bb_reversion: negative OOS on ETH and SOL
    ("bb_reversion", "ETH-USD"),
    ("bb_reversion", "SOL-USD"),
    # momentum_breakout: negative OOS on BTC
    ("momentum_breakout", "BTC-USD"),
]

# Combos with efficiency 0.0-0.5 (overfit warning, not blocked but flagged)
OVERFIT_WARNING_COMBOS = [
    ("ema_cross", "BTC-USD"),        # eff=0.26
    ("macd_trend", "ETH-USD"),       # eff=0.06
    ("rsi_adaptive", "SOL-USD"),     # eff=0.10
    ("macd_zero", "SOL-USD"),        # eff=0.15
    ("macd_trend", "SOL-USD"),       # eff=0.47
    ("bb_reversion", "XRP-USD"),     # eff=0.17
    ("momentum_breakout", "XRP-USD"),  # eff=0.42
]


def is_blacklisted(strategy: str, asset: str) -> bool:
    """Check if a strategy-asset combo is blacklisted."""
    return (strategy, asset) in OVERFITTING_COMBOS


def is_overfit_warning(strategy: str, asset: str) -> bool:
    """Check if a strategy-asset combo has an overfitting warning."""
    return (strategy, asset) in OVERFIT_WARNING_COMBOS


def get_safe_strategies(asset: str) -> list[str]:
    """Return list of strategies that are NOT blacklisted for this asset."""
    all_strategies = [
        "ema_cross", "rsi_adaptive", "macd_zero", "bb_reversion",
        "momentum_breakout", "donchian_breakout", "macd_trend",
    ]
    return [s for s in all_strategies if not is_blacklisted(s, asset)]

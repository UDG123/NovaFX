"""Canonical list of valid strategy names across NovaFX.

Used for validation when strategies are referenced by name in webhooks,
API requests, or configuration.
"""

VALID_STRATEGIES = [
    "EMA 9/21 Cross",
    "RSI 14 Reversal",
    "MACD Cross",
    "BollingerBandReversion",
]


def is_valid_strategy(name: str) -> bool:
    return name in VALID_STRATEGIES

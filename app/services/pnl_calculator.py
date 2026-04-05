"""
P&L calculation for NovaFX simulated trade outcomes.

Handles pip values and dollar P&L per asset class.
"""

# Pip sizes per asset class
PIP_SIZE = {
    "forex":       0.0001,   # Standard forex (4 decimal places)
    "jpy":         0.01,     # JPY pairs (2 decimal places)
    "gold":        0.01,     # XAUUSD — 1 pip = $0.01
    "silver":      0.001,    # XAGUSD
    "crypto":      1.0,      # Crypto tracked in dollars/percent
    "stocks":      0.01,     # Stocks tracked in dollars
    "indices":     1.0,      # Index points
}

# Pip value per 0.01 lot (micro lot) in USD
PIP_VALUE_PER_MICRO = {
    "forex":   0.10,    # $0.10 per pip per micro lot
    "jpy":     0.10,
    "gold":    1.00,    # $1.00 per pip ($0.01 move) per micro lot (1 oz)
    "silver":  0.50,
    "crypto":  1.00,    # $1 per $1 move
    "stocks":  0.01,    # $0.01 per $0.01 move
    "indices": 0.10,
}


def detect_asset_class(symbol: str) -> str:
    s = symbol.upper()
    if s in ("XAUUSD",):
        return "gold"
    if s in ("XAGUSD",):
        return "silver"
    if any(s.endswith(c) for c in ("JPY",)):
        return "jpy"
    if any(s.endswith(c) for c in ("USDT", "USD", "BTC", "ETH")) and len(s) > 6:
        return "crypto"
    if s in ("AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ"):
        return "stocks"
    if s in ("SPX500", "NAS100", "US30"):
        return "indices"
    return "forex"


def calculate_pnl(
    symbol: str,
    action: str,
    entry_price: float,
    exit_price: float,
    risk_amount: float,
    stop_loss: float,
) -> dict:
    """
    Calculate P&L for a closed trade.

    Returns:
        {
            "pnl_pips": float,
            "pnl_dollars": float,
            "pnl_r": float,          # P&L in units of R (risk multiples)
            "pip_size": float,
            "asset_class": str,
        }
    """
    asset = detect_asset_class(symbol)
    pip_size = PIP_SIZE.get(asset, 0.0001)

    is_buy = action == "BUY"
    raw_move = exit_price - entry_price if is_buy else entry_price - exit_price

    pnl_pips = round(raw_move / pip_size, 1)

    # Risk per pip based on risk_amount and SL distance
    sl_distance_pips = abs(entry_price - stop_loss) / pip_size
    if sl_distance_pips > 0:
        dollar_per_pip = risk_amount / sl_distance_pips
    else:
        dollar_per_pip = 0.0

    pnl_dollars = round(pnl_pips * dollar_per_pip, 2)
    pnl_r = round(pnl_dollars / risk_amount, 2) if risk_amount > 0 else 0.0

    return {
        "pnl_pips": pnl_pips,
        "pnl_dollars": pnl_dollars,
        "pnl_r": pnl_r,
        "pip_size": pip_size,
        "asset_class": asset,
    }


def format_pnl_display(symbol: str, pnl_pips: float, pnl_dollars: float) -> str:
    """Format P&L for Telegram display."""
    asset = detect_asset_class(symbol)
    pip_label = "pts" if asset in ("indices", "gold", "silver") else "pips"
    sign = "+" if pnl_dollars >= 0 else ""
    return f"{sign}{pnl_pips:.0f} {pip_label}  |  {sign}${pnl_dollars:.2f}"

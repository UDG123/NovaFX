from app.config import settings
from app.models.signals import IncomingSignal, ProcessedSignal

MARKET_CONFIG = {
    "forex": {
        "sl_pct": 0.3,
        "tp_pct": 0.6,
        "symbols": [
            "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD",
            "NZDUSD", "USDCHF", "EURGBP", "EURJPY", "GBPJPY",
        ],
    },
    "crypto": {
        "sl_pct": 1.5,
        "tp_pct": 3.0,
        "symbols": ["BTCUSD", "ETHUSD", "BTCUSDT", "ETHUSDT", "SOLUSD", "SOLUSDT"],
    },
    "indices": {
        "sl_pct": 0.5,
        "tp_pct": 1.0,
        "symbols": ["SPX500", "NAS100", "US30", "DE40", "UK100", "JP225"],
    },
    "commodities": {
        "sl_pct": 0.8,
        "tp_pct": 1.6,
        "symbols": ["XAUUSD", "XAGUSD", "USOIL", "UKOIL"],
    },
}


def detect_market(symbol: str) -> str:
    symbol_upper = symbol.upper().replace("/", "").replace("-", "")
    for market, cfg in MARKET_CONFIG.items():
        if symbol_upper in cfg["symbols"]:
            return market
    if any(symbol_upper.endswith(c) for c in ("USD", "JPY", "GBP", "EUR", "CHF", "AUD", "CAD", "NZD")):
        if len(symbol_upper) == 6:
            return "forex"
    if any(t in symbol_upper for t in ("BTC", "ETH", "SOL", "BNB", "USDT", "USDC")):
        return "crypto"
    return "forex"


def process_signal(signal: IncomingSignal) -> ProcessedSignal:
    market = detect_market(signal.symbol)
    cfg = MARKET_CONFIG.get(market, MARKET_CONFIG["forex"])
    sl_pct = cfg["sl_pct"] / 100
    tp_pct = cfg["tp_pct"] / 100

    if signal.action == "BUY":
        sl = signal.sl if signal.sl is not None else round(signal.price * (1 - sl_pct), 6)
        tp = signal.tp if signal.tp is not None else round(signal.price * (1 + tp_pct), 6)
    else:
        sl = signal.sl if signal.sl is not None else round(signal.price * (1 + sl_pct), 6)
        tp = signal.tp if signal.tp is not None else round(signal.price * (1 - tp_pct), 6)

    risk_per_unit = abs(signal.price - sl)
    reward_per_unit = abs(tp - signal.price)
    risk_reward = round(reward_per_unit / risk_per_unit, 2) if risk_per_unit > 0 else 0.0

    risk_amount = round(settings.ACCOUNT_BALANCE * (settings.DEFAULT_RISK_PCT / 100), 2)
    position_size = round(risk_amount / risk_per_unit, 4) if risk_per_unit > 0 else 0.0

    return ProcessedSignal(
        symbol=signal.symbol.upper(),
        action=signal.action,
        entry_price=signal.price,
        stop_loss=sl,
        take_profit=tp,
        risk_reward=risk_reward,
        position_size=position_size,
        risk_amount=risk_amount,
        timeframe=signal.timeframe,
        source=signal.source,
        indicator=signal.indicator,
    )

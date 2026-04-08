#!/usr/bin/env python3
"""Run paper trading loop.

Usage:
    python scripts/run_paper_trading.py --exchange local --symbols BTC-USD --interval 5
    python scripts/run_paper_trading.py --symbols BTC-USD,ETH-USD --strategies macd_trend,ema_cross
"""
import sys
import time
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_harness import fetch_ohlcv
from src.strategies import get_strategy, STRATEGY_REGISTRY
from src.vectorbt_adapters.data_utils import prepare_vbt_data
from app.services.paper_trader import (
    PaperTrader, LocalSimulator, Signal,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def fetch_latest(symbol: str, yf_ticker: str) -> tuple:
    """Fetch latest OHLCV and return (df, current_price)."""
    df = fetch_ohlcv(yf_ticker, "1h", "30d")
    df = df.set_index("timestamp")
    df = prepare_vbt_data(df, symbol)
    price = float(df["close"].iloc[-1])
    return df, price


def generate_signals(df, symbol: str, strategies: list[str]) -> list[Signal]:
    """Run strategies and return list of Signals."""
    signals = []
    for strat_name in strategies:
        try:
            strat = get_strategy(strat_name)
            result = strat.generate_signals(df)
            last = result.iloc[-1]
            if last["signal"] == 0:
                continue

            side = "BUY" if last["signal"] == 1 else "SELL"
            signals.append(Signal(
                strategy=strat_name,
                symbol=symbol,
                side=side,
                entry_price=float(last["entry_price"]),
                stop_loss=float(last["stop_loss"]),
                take_profit=float(last["take_profit"]),
                confidence=float(last["confidence"]),
            ))
        except Exception as e:
            logger.warning("Strategy %s failed on %s: %s", strat_name, symbol, e)
    return signals


# Yahoo ticker mappings
TICKER_MAP = {
    "BTC-USD": "BTC-USD", "ETH-USD": "ETH-USD",
    "SOL-USD": "SOL-USD", "XRP-USD": "XRP-USD",
    "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X", "AAPL": "AAPL", "SPY": "SPY",
    "XAUUSD": "GC=F", "XAGUSD": "SI=F",
}


def main():
    parser = argparse.ArgumentParser(description="NovaFX Paper Trading")
    parser.add_argument("--exchange", default="local", choices=["local"])
    parser.add_argument("--symbols", default="BTC-USD")
    parser.add_argument("--strategies", default=",".join(["macd_trend", "ema_cross", "donchian_breakout"]))
    parser.add_argument("--interval", type=int, default=60, help="Seconds between cycles")
    parser.add_argument("--risk", type=float, default=0.02, help="Risk per trade")
    parser.add_argument("--cash", type=float, default=10000.0)
    args = parser.parse_args()

    symbols = args.symbols.split(",")
    strategies = args.strategies.split(",")

    # Create exchange
    exchange = LocalSimulator(initial_cash=args.cash)
    trader = PaperTrader(exchange, risk_per_trade=args.risk)

    # Try to load previous state
    if trader.load_state():
        logger.info("Loaded previous state: %d trades", len(trader.trades))

    logger.info("Starting paper trading: symbols=%s strategies=%s interval=%ds",
                 symbols, strategies, args.interval)

    cycle = 0
    try:
        while True:
            cycle += 1
            logger.info("── Cycle %d ──", cycle)

            for symbol in symbols:
                ticker = TICKER_MAP.get(symbol, symbol)
                try:
                    df, price = fetch_latest(symbol, ticker)
                    exchange.set_price(symbol, price)
                    logger.info("%s current price: %.2f", symbol, price)
                except Exception as e:
                    logger.error("Failed to fetch %s: %s", symbol, e)
                    continue

                # Generate signals
                sigs = generate_signals(df, symbol, strategies)
                for sig in sigs:
                    logger.info("Signal: %s %s @ %.2f (SL=%.2f TP=%.2f) [%s]",
                                 sig.side, sig.symbol, sig.entry_price,
                                 sig.stop_loss, sig.take_profit, sig.strategy)
                    trader.process_signal(sig)

            # Update positions
            closed = trader.update_positions()
            for t in closed:
                logger.info("CLOSED: %s %s PnL=%.2f (%s)", t.side, t.symbol, t.pnl, t.exit_reason)

            # Print status
            stats = trader.get_stats()
            logger.info("Status: balance=$%.2f trades=%d open=%d WR=%.1f%% PnL=%.2f",
                         stats["balance"], stats["n_trades"], stats["n_open"],
                         stats["win_rate"] * 100, stats["total_pnl"])

            # Save state
            trader.save_state()

            if args.interval <= 0:
                break  # Single-shot mode
            time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("Shutting down...")
        trader.save_state()
        stats = trader.get_stats()
        print(f"\nFinal stats: {stats}")


if __name__ == "__main__":
    main()

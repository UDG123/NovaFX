"""Paper trading engine for NovaFX.

Supports local simulation and exchange adapters.
Tracks signals, orders, positions, trades with execution metrics.
"""
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/paper_trading/paper_trading_state.json")


@dataclass
class Signal:
    strategy: str
    symbol: str
    side: str  # "BUY" or "SELL"
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float = 1.0
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class Order:
    signal: Signal
    intended_price: float
    fill_price: float
    slippage_pct: float
    commission: float
    latency_ms: float
    quantity: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


@dataclass
class Position:
    symbol: str
    side: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    strategy: str
    open_time: str
    unrealized_pnl: float = 0.0
    order: Order | None = None


@dataclass
class Trade:
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    pnl_pct: float
    strategy: str
    exit_reason: str  # "stop_loss", "take_profit", "manual", "signal"
    expected_slippage: float
    actual_slippage: float
    latency_ms: float
    open_time: str
    close_time: str
    duration_s: float = 0.0


# ── Exchange Adapters ──────────────────────────────────────────────────


class ExchangeAdapter(ABC):
    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def get_balance(self) -> float:
        ...

    @abstractmethod
    def get_price(self, symbol: str) -> float:
        ...

    @abstractmethod
    def place_order(self, symbol: str, side: str, quantity: float,
                    price: float) -> Order:
        ...


class LocalSimulator(ExchangeAdapter):
    """Local fill simulator with configurable slippage/commission/latency."""

    def __init__(self, initial_cash: float = 10000.0,
                 slippage_pct: float = 0.0005,
                 commission_pct: float = 0.001,
                 latency_ms: float = 50.0):
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.slippage_pct = slippage_pct
        self.commission_pct = commission_pct
        self.latency_ms = latency_ms
        self._prices: dict[str, float] = {}

    def connect(self) -> bool:
        return True

    def get_balance(self) -> float:
        return self.cash

    def set_price(self, symbol: str, price: float) -> None:
        self._prices[symbol] = price

    def get_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    def place_order(self, symbol: str, side: str, quantity: float,
                    price: float) -> Order:
        # Simulate latency
        time.sleep(self.latency_ms / 1000)

        # Apply slippage
        if side == "BUY":
            fill_price = price * (1 + self.slippage_pct)
        else:
            fill_price = price * (1 - self.slippage_pct)

        slippage_actual = abs(fill_price - price) / price * 100
        commission = fill_price * quantity * self.commission_pct

        # Update cash
        if side == "BUY":
            self.cash -= fill_price * quantity + commission
        else:
            self.cash += fill_price * quantity - commission

        return Order(
            signal=Signal(strategy="", symbol=symbol, side=side,
                          entry_price=price, stop_loss=0, take_profit=0),
            intended_price=price,
            fill_price=fill_price,
            slippage_pct=slippage_actual,
            commission=commission,
            latency_ms=self.latency_ms,
            quantity=quantity,
        )


# ── Paper Trader ───────────────────────────────────────────────────────


class PaperTrader:
    """Paper trading engine with execution tracking."""

    def __init__(self, exchange: ExchangeAdapter,
                 risk_per_trade: float = 0.02,
                 max_positions: int = 5):
        self.exchange = exchange
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.positions: list[Position] = []
        self.trades: list[Trade] = []
        self.signals_processed: int = 0

    def process_signal(self, signal: Signal) -> Position | None:
        """Size position, place order, create Position."""
        if len(self.positions) >= self.max_positions:
            logger.info("Max positions reached, skipping %s %s", signal.side, signal.symbol)
            return None

        # Check for existing position in same symbol
        for p in self.positions:
            if p.symbol == signal.symbol:
                logger.info("Already have position in %s, skipping", signal.symbol)
                return None

        # Position sizing
        cash = self.exchange.get_balance()
        risk_amount = cash * self.risk_per_trade
        stop_distance = abs(signal.entry_price - signal.stop_loss)
        if stop_distance <= 0:
            return None

        quantity = risk_amount / stop_distance
        if quantity <= 0:
            return None

        # Place order
        order = self.exchange.place_order(
            signal.symbol, signal.side, quantity, signal.entry_price,
        )
        order.signal = signal
        self.signals_processed += 1

        pos = Position(
            symbol=signal.symbol,
            side=signal.side,
            entry_price=order.fill_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.strategy,
            open_time=signal.timestamp,
            order=order,
        )
        self.positions.append(pos)
        logger.info("Opened %s %s %.4f @ %.2f (SL=%.2f TP=%.2f)",
                     signal.side, signal.symbol, quantity,
                     order.fill_price, signal.stop_loss, signal.take_profit)
        return pos

    def update_positions(self) -> list[Trade]:
        """Check stops/targets, close positions, return completed trades."""
        closed: list[Trade] = []
        remaining: list[Position] = []

        for pos in self.positions:
            price = self.exchange.get_price(pos.symbol)
            if price <= 0:
                remaining.append(pos)
                continue

            # Update unrealized PnL
            if pos.side == "BUY":
                pos.unrealized_pnl = (price - pos.entry_price) * pos.quantity
            else:
                pos.unrealized_pnl = (pos.entry_price - price) * pos.quantity

            exit_reason = None
            exit_price = price

            if pos.side == "BUY":
                if price <= pos.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = pos.stop_loss
                elif price >= pos.take_profit:
                    exit_reason = "take_profit"
                    exit_price = pos.take_profit
            else:
                if price >= pos.stop_loss:
                    exit_reason = "stop_loss"
                    exit_price = pos.stop_loss
                elif price <= pos.take_profit:
                    exit_reason = "take_profit"
                    exit_price = pos.take_profit

            if exit_reason:
                # Close position
                close_order = self.exchange.place_order(
                    pos.symbol,
                    "SELL" if pos.side == "BUY" else "BUY",
                    pos.quantity, exit_price,
                )

                if pos.side == "BUY":
                    pnl = (close_order.fill_price - pos.entry_price) * pos.quantity
                    pnl_pct = (close_order.fill_price - pos.entry_price) / pos.entry_price
                else:
                    pnl = (pos.entry_price - close_order.fill_price) * pos.quantity
                    pnl_pct = (pos.entry_price - close_order.fill_price) / pos.entry_price

                now = datetime.now(timezone.utc).isoformat()
                expected_slip = pos.order.slippage_pct if pos.order else 0
                actual_slip = close_order.slippage_pct

                trade = Trade(
                    symbol=pos.symbol, side=pos.side,
                    entry_price=pos.entry_price,
                    exit_price=close_order.fill_price,
                    quantity=pos.quantity,
                    pnl=round(pnl, 4),
                    pnl_pct=round(pnl_pct, 6),
                    strategy=pos.strategy,
                    exit_reason=exit_reason,
                    expected_slippage=expected_slip,
                    actual_slippage=actual_slip,
                    latency_ms=close_order.latency_ms,
                    open_time=pos.open_time,
                    close_time=now,
                )
                closed.append(trade)
                self.trades.append(trade)
                logger.info("Closed %s %s: %s PnL=%.2f (%.2f%%)",
                             pos.side, pos.symbol, exit_reason, pnl, pnl_pct * 100)
            else:
                remaining.append(pos)

        self.positions = remaining
        return closed

    def get_stats(self) -> dict[str, Any]:
        """Return aggregate statistics."""
        n = len(self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        total_pnl = sum(t.pnl for t in self.trades)
        avg_exp_slip = sum(t.expected_slippage for t in self.trades) / n if n else 0
        avg_act_slip = sum(t.actual_slippage for t in self.trades) / n if n else 0
        avg_latency = sum(t.latency_ms for t in self.trades) / n if n else 0

        return {
            "n_trades": n,
            "n_open": len(self.positions),
            "signals_processed": self.signals_processed,
            "win_rate": len(wins) / n if n else 0,
            "total_pnl": round(total_pnl, 4),
            "avg_expected_slippage": round(avg_exp_slip, 6),
            "avg_actual_slippage": round(avg_act_slip, 6),
            "slippage_ratio": round(avg_act_slip / avg_exp_slip, 4) if avg_exp_slip else 0,
            "avg_latency_ms": round(avg_latency, 2),
            "balance": round(self.exchange.get_balance(), 2),
        }

    def save_state(self, path: Path | str | None = None) -> None:
        """Persist state to JSON."""
        p = Path(path or STATE_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "stats": self.get_stats(),
            "positions": [_pos_to_dict(pos) for pos in self.positions],
            "trades": [asdict(t) for t in self.trades],
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(p, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, path: Path | str | None = None) -> bool:
        """Load state from JSON. Returns True if loaded."""
        p = Path(path or STATE_PATH)
        if not p.exists():
            return False
        with open(p) as f:
            state = json.load(f)
        # Restore trades
        self.trades = [Trade(**t) for t in state.get("trades", [])]
        return True


def _pos_to_dict(pos: Position) -> dict:
    return {
        "symbol": pos.symbol, "side": pos.side,
        "entry_price": pos.entry_price, "quantity": pos.quantity,
        "stop_loss": pos.stop_loss, "take_profit": pos.take_profit,
        "strategy": pos.strategy, "open_time": pos.open_time,
        "unrealized_pnl": pos.unrealized_pnl,
    }

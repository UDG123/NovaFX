"""
NovaFX Paper Trader - Main Service Loop.

Monitors Redis for new signals, manages simulated positions,
and sends results to Telegram.
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone, time as dt_time
from typing import Optional

import redis
from dotenv import load_dotenv

from trader import (
    Position,
    create_position,
    check_position_exit,
    calculate_pnl,
    fetch_current_price,
    send_telegram,
    format_entry_message,
    format_exit_message,
    format_daily_summary,
    normalize_symbol,
    ACCOUNT_BALANCE,
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("novafx.paper-trader")

# Redis configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Polling intervals
SIGNAL_POLL_INTERVAL = 60  # Check for new signals every 60 seconds
POSITION_CHECK_INTERVAL = 300  # Check open positions every 5 minutes
DAILY_SUMMARY_HOUR = 17  # Send daily summary at 17:00 UTC

# Redis key prefixes
SIGNAL_KEY_PREFIX = "novafx:signal_sent:"
POSITION_KEY_PREFIX = "novafx:paper:positions:"
PROCESSED_SIGNALS_KEY = "novafx:paper:processed_signals"
DAILY_TRADES_KEY = "novafx:paper:daily_trades:"
ACCOUNT_BALANCE_KEY = "novafx:paper:account_balance"
STREAM_LAST_ID_KEY = "novafx:paper:stream_last_id"

# Redis stream for real-time signals
PAPER_TRADES_STREAM = "novafx:trades:paper"


class PaperTrader:
    """Paper trading service that simulates trades based on scanner signals."""

    def __init__(self):
        self.redis_client: Optional[redis.Redis] = None
        self.running = False
        self.account_balance = ACCOUNT_BALANCE
        self.last_daily_summary_date: Optional[str] = None

    def connect_redis(self) -> bool:
        """Connect to Redis."""
        try:
            self.redis_client = redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_URL}")

            # Load account balance from Redis if exists
            stored_balance = self.redis_client.get(ACCOUNT_BALANCE_KEY)
            if stored_balance:
                self.account_balance = float(stored_balance)
                logger.info(f"Loaded account balance: ${self.account_balance:,.2f}")
            else:
                self.redis_client.set(ACCOUNT_BALANCE_KEY, str(self.account_balance))
                logger.info(f"Initialized account balance: ${self.account_balance:,.2f}")

            return True
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            return False

    def get_processed_signals(self) -> set:
        """Get set of already processed signal keys."""
        if not self.redis_client:
            return set()
        members = self.redis_client.smembers(PROCESSED_SIGNALS_KEY)
        return set(members) if members else set()

    def mark_signal_processed(self, signal_key: str) -> None:
        """Mark a signal as processed."""
        if self.redis_client:
            self.redis_client.sadd(PROCESSED_SIGNALS_KEY, signal_key)
            # Keep processed signals for 24 hours
            self.redis_client.expire(PROCESSED_SIGNALS_KEY, 86400)

    def get_new_signals(self) -> list[tuple[str, str]]:
        """
        Poll Redis for new signals.

        Returns list of (symbol, direction) tuples for new signals.
        """
        if not self.redis_client:
            return []

        new_signals = []
        processed = self.get_processed_signals()

        # Scan for signal keys: novafx:signal_sent:{symbol}:{direction}
        cursor = 0
        pattern = f"{SIGNAL_KEY_PREFIX}*"

        while True:
            cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)
            for key in keys:
                if key not in processed:
                    # Parse symbol and direction from key
                    # Key format: novafx:signal_sent:EURUSD:sell
                    parts = key.replace(SIGNAL_KEY_PREFIX, "").split(":")
                    if len(parts) >= 2:
                        symbol = parts[0]
                        direction = parts[1]
                        new_signals.append((symbol, direction))
                        logger.info(f"Found new signal: {direction.upper()} {symbol}")
                        self.mark_signal_processed(key)

            if cursor == 0:
                break

        return new_signals

    def get_stream_signals(self) -> list[tuple[str, str, float]]:
        """
        Read new signals from Redis stream.

        Returns list of (symbol, direction, price) tuples for new signals.
        """
        if not self.redis_client:
            return []

        new_signals = []

        # Get last processed stream ID or start from now
        last_id = self.redis_client.get(STREAM_LAST_ID_KEY) or "0"

        try:
            # Read from stream (non-blocking with count limit)
            entries = self.redis_client.xread(
                {PAPER_TRADES_STREAM: last_id},
                count=50,
                block=0  # Non-blocking
            )

            if entries:
                for stream_name, messages in entries:
                    for msg_id, data in messages:
                        symbol = data.get("symbol", "")
                        direction = data.get("direction", "")
                        price_str = data.get("price", "0")

                        try:
                            price = float(price_str)
                        except ValueError:
                            price = 0.0

                        if symbol and direction and price > 0:
                            new_signals.append((symbol, direction, price))
                            logger.info(f"Stream signal: {direction.upper()} {symbol} @ {price}")

                        # Update last processed ID
                        self.redis_client.set(STREAM_LAST_ID_KEY, msg_id)

        except Exception as e:
            logger.error(f"Error reading stream: {e}")

        return new_signals

    def save_position(self, position: Position) -> None:
        """Save position to Redis."""
        if not self.redis_client:
            return

        key = f"{POSITION_KEY_PREFIX}{position.symbol}:{position.direction}"
        self.redis_client.set(key, json.dumps(position.to_dict()))
        logger.info(f"Saved position: {key}")

    def get_open_positions(self) -> list[Position]:
        """Get all open positions from Redis."""
        if not self.redis_client:
            return []

        positions = []
        cursor = 0
        pattern = f"{POSITION_KEY_PREFIX}*"

        while True:
            cursor, keys = self.redis_client.scan(cursor, match=pattern, count=100)
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    pos_dict = json.loads(data)
                    if pos_dict.get("status") == "open":
                        positions.append(Position.from_dict(pos_dict))

            if cursor == 0:
                break

        return positions

    def close_position(self, position: Position, exit_price: float, reason: str) -> dict:
        """Close a position and update Redis."""
        pnl_usd, pnl_percent = calculate_pnl(position, exit_price)

        # Update account balance
        self.account_balance += pnl_usd
        if self.redis_client:
            self.redis_client.set(ACCOUNT_BALANCE_KEY, str(self.account_balance))

        # Mark position as closed
        position.status = "closed"
        key = f"{POSITION_KEY_PREFIX}{position.symbol}:{position.direction}"
        if self.redis_client:
            self.redis_client.delete(key)

        # Record trade for daily summary
        trade_record = {
            "symbol": position.symbol,
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": exit_price,
            "entry_time": position.entry_time,
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "size_usd": position.size_usd,
            "pnl_usd": pnl_usd,
            "pnl_percent": pnl_percent,
            "reason": reason,
        }

        # Save to daily trades
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        daily_key = f"{DAILY_TRADES_KEY}{today}"
        if self.redis_client:
            self.redis_client.rpush(daily_key, json.dumps(trade_record))
            self.redis_client.expire(daily_key, 86400 * 7)  # Keep for 7 days

        logger.info(
            f"Closed position: {position.direction.upper()} {position.symbol} "
            f"P&L: ${pnl_usd:+.2f} ({pnl_percent:+.2f}%)"
        )

        return trade_record

    def get_daily_trades(self, date_str: str) -> list[dict]:
        """Get all trades for a specific date."""
        if not self.redis_client:
            return []

        daily_key = f"{DAILY_TRADES_KEY}{date_str}"
        trades_raw = self.redis_client.lrange(daily_key, 0, -1)

        return [json.loads(t) for t in trades_raw] if trades_raw else []

    async def process_new_signals(self) -> None:
        """Process any new signals and open positions."""
        # Get signals from key polling (legacy method)
        key_signals = self.get_new_signals()
        # Get signals from Redis stream (real-time method)
        stream_signals = self.get_stream_signals()

        # Process key-based signals (need to fetch price)
        for symbol, direction in key_signals:
            current_price = await fetch_current_price(symbol)
            if current_price is None:
                logger.warning(f"Could not fetch price for {symbol}, skipping signal")
                continue

            position = create_position(symbol, direction, current_price)
            self.save_position(position)

            message = format_entry_message(position)
            await send_telegram(message)

            logger.info(
                f"Opened paper position: {direction.upper()} {symbol} @ {current_price}"
            )

        # Process stream signals (already have price from scanner)
        for symbol, direction, price in stream_signals:
            position = create_position(symbol, direction, price)
            self.save_position(position)

            message = format_entry_message(position)
            await send_telegram(message)

            logger.info(
                f"Opened paper position (stream): {direction.upper()} {symbol} @ {price}"
            )

    async def check_open_positions(self) -> None:
        """Check all open positions for SL/TP/timeout."""
        positions = self.get_open_positions()

        for position in positions:
            # Fetch current price
            current_price = await fetch_current_price(position.symbol)
            if current_price is None:
                logger.warning(f"Could not fetch price for {position.symbol}")
                continue

            # Check if should exit
            should_close, reason, exit_price = check_position_exit(
                position, current_price
            )

            if should_close:
                # Close the position
                trade_record = self.close_position(position, exit_price, reason)

                # Send Telegram notification
                message = format_exit_message(
                    position,
                    exit_price,
                    reason,
                    trade_record["pnl_usd"],
                    trade_record["pnl_percent"],
                )
                await send_telegram(message)

    async def send_daily_summary(self) -> None:
        """Send daily summary at 17:00 UTC."""
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")

        # Only send once per day
        if self.last_daily_summary_date == today:
            return

        # Check if it's time (17:00 UTC)
        if now.hour != DAILY_SUMMARY_HOUR:
            return

        self.last_daily_summary_date = today

        # Get today's trades
        trades = self.get_daily_trades(today)

        if not trades:
            logger.info("No trades today, skipping daily summary")
            return

        # Calculate account values
        total_pnl = sum(t.get("pnl_usd", 0) for t in trades)
        account_start = self.account_balance - total_pnl
        account_end = self.account_balance

        # Format and send summary
        date_display = now.strftime("%b %d, %Y")
        message = format_daily_summary(trades, account_start, account_end, date_display)
        await send_telegram(message)

        logger.info(f"Sent daily summary: {len(trades)} trades, P&L: ${total_pnl:+.2f}")

    async def run(self) -> None:
        """Main service loop."""
        logger.info("Starting NovaFX Paper Trader")

        # Connect to Redis
        if not self.connect_redis():
            logger.error("Failed to connect to Redis, exiting")
            return

        self.running = True
        last_signal_check = 0
        last_position_check = 0

        logger.info(
            f"Paper Trader running - Signal poll: {SIGNAL_POLL_INTERVAL}s, "
            f"Position check: {POSITION_CHECK_INTERVAL}s"
        )

        while self.running:
            try:
                now = asyncio.get_event_loop().time()

                # Check for new signals every 60 seconds
                if now - last_signal_check >= SIGNAL_POLL_INTERVAL:
                    await self.process_new_signals()
                    last_signal_check = now

                # Check open positions every 5 minutes
                if now - last_position_check >= POSITION_CHECK_INTERVAL:
                    await self.check_open_positions()
                    last_position_check = now

                # Check for daily summary
                await self.send_daily_summary()

                # Sleep briefly to avoid tight loop
                await asyncio.sleep(10)

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(30)  # Wait before retrying

    def stop(self) -> None:
        """Stop the service."""
        self.running = False
        logger.info("Paper Trader stopping")


async def main():
    """Entry point."""
    trader = PaperTrader()

    try:
        await trader.run()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    finally:
        trader.stop()


if __name__ == "__main__":
    asyncio.run(main())

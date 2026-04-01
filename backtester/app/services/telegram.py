import logging

import httpx

from app.config import settings
from backtester.app.core.orchestrator import composite_score
from backtester.app.models.backtest import PhaseResult

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
TELEGRAM_MAX_LENGTH = 4096

# Unicode block characters for progress bars (empty to full in 8ths)
_BAR_CHARS = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
_FILLED = "\u2588"
_EMPTY = "\u2591"
BAR_WIDTH = 10


def _progress_bar(value: float, max_value: float = 100.0) -> str:
    """Build a Unicode progress bar of BAR_WIDTH characters."""
    if max_value <= 0:
        return _EMPTY * BAR_WIDTH
    ratio = max(0.0, min(1.0, value / max_value))
    filled = int(ratio * BAR_WIDTH)
    remainder = (ratio * BAR_WIDTH) - filled
    partial_idx = int(remainder * 8)

    bar = _FILLED * filled
    if filled < BAR_WIDTH:
        bar += _BAR_CHARS[partial_idx]
        bar += _EMPTY * (BAR_WIDTH - filled - 1)
    return bar


def _win_rate_indicator(win_rate: float) -> str:
    """Color-coded emoji for win rate."""
    if win_rate > 55:
        return "\U0001f7e2"  # green circle
    elif win_rate >= 45:
        return "\U0001f7e1"  # yellow circle
    else:
        return "\U0001f534"  # red circle


def _pnl_arrow(pnl: float) -> str:
    return "\u25b2" if pnl >= 0 else "\u25bc"


def _format_strategy_block(result: PhaseResult) -> str:
    """Format a single strategy result as a compact visual block."""
    wr_emoji = _win_rate_indicator(result.win_rate)
    wr_bar = _progress_bar(result.win_rate)
    pnl_bar = _progress_bar(abs(result.net_pnl_pct), max_value=20.0)
    pnl_arrow = _pnl_arrow(result.net_pnl_pct)
    score = composite_score(result)

    return (
        f"<b>{result.strategy}</b>  {wr_emoji}\n"
        f"  WR  <code>{wr_bar}</code> {result.win_rate:.1f}%"
        f"  ({result.wins}W/{result.losses}L)\n"
        f"  PnL <code>{pnl_bar}</code> {pnl_arrow}{result.net_pnl_pct:+.2f}%\n"
        f"  DD {result.max_drawdown_pct:.2f}%"
        f"  \u2022  Fee {result.total_commission_pct:.2f}%"
        f"  \u2022  Score {score:.1f}\n"
    )


def format_cycle_report(results: list[PhaseResult], cycle_label: str = "Backtest") -> str:
    """Format a full cycle report with visual progress bars per strategy.

    Guarantees the output fits within Telegram's 4096-char limit by
    truncating strategy blocks from the bottom if necessary.
    """
    if not results:
        return f"<b>\u2501\u2501\u2501 {cycle_label} Report \u2501\u2501\u2501</b>\n\nNo results."

    sorted_results = sorted(results, key=composite_score, reverse=True)
    best = sorted_results[0]

    # Build header
    header = (
        f"<b>\u2501\u2501\u2501 {cycle_label} Report \u2501\u2501\u2501</b>\n\n"
        f"\U0001f3c6 <b>Best: {best.strategy}</b> on <code>{best.symbol}</code>\n"
        f"   {_win_rate_indicator(best.win_rate)} {best.win_rate:.1f}% WR"
        f"  |  {_pnl_arrow(best.net_pnl_pct)}{best.net_pnl_pct:+.2f}% net\n\n"
        f"\u2500\u2500\u2500 Strategies ({len(sorted_results)}) \u2500\u2500\u2500\n\n"
    )

    # Build footer
    total_trades = sum(r.total_trades for r in sorted_results)
    total_net = sum(r.net_pnl_pct for r in sorted_results)
    total_comm = sum(r.total_commission_pct for r in sorted_results)
    footer = (
        f"\n\u2500\u2500\u2500 Summary \u2500\u2500\u2500\n"
        f"Strategies: {len(sorted_results)}"
        f"  |  Trades: {total_trades}"
        f"  |  Net: {_pnl_arrow(total_net)}{total_net:+.2f}%"
        f"  |  Fees: {total_comm:.2f}%"
    )

    # Truncation message (reserved space if needed)
    truncation_msg = "\n\n<i>... {remaining} more strategies truncated</i>"

    # Fill in strategy blocks, truncating if we'd exceed the limit
    budget = TELEGRAM_MAX_LENGTH - len(header) - len(footer) - len(truncation_msg) - 20
    blocks: list[str] = []
    for result in sorted_results:
        block = _format_strategy_block(result)
        if sum(len(b) for b in blocks) + len(block) > budget:
            remaining = len(sorted_results) - len(blocks)
            blocks.append(truncation_msg.format(remaining=remaining))
            break
        blocks.append(block)

    body = "\n".join(blocks)
    message = header + body + footer

    # Final safety trim (should never trigger given the budget logic above)
    if len(message) > TELEGRAM_MAX_LENGTH:
        message = message[: TELEGRAM_MAX_LENGTH - 3] + "..."

    return message


async def send_cycle_report(results: list[PhaseResult], cycle_label: str = "Backtest") -> bool:
    if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
        logger.warning("Telegram credentials not configured - skipping report")
        return False

    url = TELEGRAM_API.format(token=settings.TELEGRAM_BOT_TOKEN)
    payload = {
        "chat_id": settings.TELEGRAM_CHAT_ID,
        "text": format_cycle_report(results, cycle_label),
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            logger.info("Cycle report sent (%d strategies)", len(results))
            return True
    except httpx.HTTPError as exc:
        logger.error("Failed to send cycle report: %s", exc)
        return False

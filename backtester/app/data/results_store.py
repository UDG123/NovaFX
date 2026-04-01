"""SQLite persistence for backtest cycle results."""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backtester.app.models.backtest import BacktestSummary

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "results.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            strategy_name TEXT NOT NULL,
            composite_score REAL NOT NULL,
            backtest_win_rate REAL NOT NULL,
            forward_win_rate REAL,
            total_trades INTEGER NOT NULL,
            symbols_tested TEXT NOT NULL,
            ran_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


init_db()


def save_result(summary: BacktestSummary) -> int:
    """Persist a BacktestSummary and return the row id."""
    conn = _get_conn()
    cur = conn.execute(
        """
        INSERT INTO results
            (strategy_name, composite_score, backtest_win_rate,
             forward_win_rate, total_trades, symbols_tested, ran_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            summary.strategy_name,
            summary.composite_score,
            summary.backtest_win_rate,
            summary.forward_win_rate,
            summary.total_trades,
            summary.symbols_tested,
            summary.ran_at.isoformat(),
        ),
    )
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    logger.info("Saved result id=%d for %s", row_id, summary.strategy_name)
    return row_id


def get_recent_results(limit: int = 10) -> list[BacktestSummary]:
    """Return the most recent results, newest first."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT * FROM results ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()

    return [
        BacktestSummary(
            id=row["id"],
            strategy_name=row["strategy_name"],
            composite_score=row["composite_score"],
            backtest_win_rate=row["backtest_win_rate"],
            forward_win_rate=row["forward_win_rate"],
            total_trades=row["total_trades"],
            symbols_tested=row["symbols_tested"],
            ran_at=datetime.fromisoformat(row["ran_at"]),
        )
        for row in rows
    ]


def clear_results() -> None:
    """Delete all results (for testing)."""
    conn = _get_conn()
    conn.execute("DELETE FROM results")
    conn.commit()
    conn.close()

"""NovaFX Paper Trading Dashboard (Streamlit).

Usage:
    streamlit run app/dashboard/app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

try:
    import streamlit as st
except ImportError:
    raise SystemExit("pip install streamlit plotly")

STATE_PATH = Path("data/paper_trading/paper_trading_state.json")
VALIDATION_PATH = Path("results/extended_validation.json")

st.set_page_config(page_title="NovaFX Dashboard", layout="wide")
st.title("NovaFX Paper Trading Dashboard")


def load_state() -> dict | None:
    if not STATE_PATH.exists():
        return None
    with open(STATE_PATH) as f:
        return json.load(f)


def load_validation() -> dict | None:
    if not VALIDATION_PATH.exists():
        return None
    with open(VALIDATION_PATH) as f:
        return json.load(f)


# ── Load Data ──────────────────────────────────────────────────────────
state = load_state()
validation = load_validation()

if state is None:
    st.info("No paper trading data yet. Run: `python scripts/run_paper_trading.py`")
    st.stop()

stats = state.get("stats", {})
trades = state.get("trades", [])
positions = state.get("positions", [])

# ── Metric Cards ───────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Trades", stats.get("n_trades", 0))
c2.metric("Total PnL", f"${stats.get('total_pnl', 0):,.2f}")
c3.metric("Win Rate", f"{stats.get('win_rate', 0):.1%}")
c4.metric("Open Positions", stats.get("n_open", 0))

# ── Equity Curve ───────────────────────────────────────────────────────
if trades:
    st.subheader("Equity Curve")
    pnls = [t["pnl"] for t in trades]
    cum_pnl = [sum(pnls[:i+1]) for i in range(len(pnls))]
    equity = [10000 + c for c in cum_pnl]

    # Drawdown
    peak = equity[0]
    dd = []
    for e in equity:
        peak = max(peak, e)
        dd.append((e - peak) / peak * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(y=equity, name="Equity", line=dict(color="#2196F3")))
    fig.add_trace(go.Scatter(y=dd, name="Drawdown %", yaxis="y2",
                              fill="tozeroy", fillcolor="rgba(255,82,82,0.15)",
                              line=dict(color="rgba(255,82,82,0.5)")))
    fig.update_layout(
        yaxis=dict(title="Equity ($)"),
        yaxis2=dict(title="Drawdown %", overlaying="y", side="right", range=[-50, 5]),
        height=400, margin=dict(t=10),
    )
    st.plotly_chart(fig, use_container_width=True)

# ── Open Positions ─────────────────────────────────────────────────────
if positions:
    st.subheader("Open Positions")
    pos_df = pd.DataFrame(positions)
    st.dataframe(pos_df[["symbol", "side", "entry_price", "quantity",
                          "stop_loss", "take_profit", "strategy", "unrealized_pnl"]],
                 use_container_width=True)
else:
    st.info("No open positions")

# ── Recent Trades ──────────────────────────────────────────────────────
if trades:
    st.subheader("Recent Trades (Last 20)")
    trades_df = pd.DataFrame(trades[-20:])
    cols = ["symbol", "side", "entry_price", "exit_price", "pnl", "pnl_pct",
            "strategy", "exit_reason", "close_time"]
    display_cols = [c for c in cols if c in trades_df.columns]
    st.dataframe(trades_df[display_cols], use_container_width=True)

# ── Strategy Performance ──────────────────────────────────────────────
if trades:
    st.subheader("Strategy Performance")
    trades_df = pd.DataFrame(trades)
    strat_stats = trades_df.groupby("strategy").agg(
        trades=("pnl", "count"),
        total_pnl=("pnl", "sum"),
        win_rate=("pnl", lambda x: (x > 0).mean()),
        avg_pnl=("pnl", "mean"),
    ).round(4)

    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=strat_stats.index, y=strat_stats["total_pnl"],
                           name="Total PnL",
                           marker_color=["#4CAF50" if v > 0 else "#F44336"
                                          for v in strat_stats["total_pnl"]]))
    fig2.update_layout(height=300, margin=dict(t=10))
    st.plotly_chart(fig2, use_container_width=True)
    st.dataframe(strat_stats, use_container_width=True)

# ── Execution Metrics ──────────────────────────────────────────────────
st.subheader("Execution Metrics")
ec1, ec2, ec3 = st.columns(3)
ec1.metric("Avg Expected Slippage", f"{stats.get('avg_expected_slippage', 0):.4f}%")
ec2.metric("Avg Actual Slippage", f"{stats.get('avg_actual_slippage', 0):.4f}%")
ec3.metric("Avg Latency", f"{stats.get('avg_latency_ms', 0):.1f} ms")

slip_ratio = stats.get("slippage_ratio", 0)
if slip_ratio > 0:
    st.progress(min(slip_ratio, 2.0) / 2.0,
                text=f"Slippage Ratio: {slip_ratio:.2f}x (actual/expected)")

# ── Validation Status ─────────────────────────────────────────────────
if validation:
    st.subheader("Extended Validation Status")
    meta = validation.get("meta", {})
    results = validation.get("results", [])
    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("Combos Tested", meta.get("n_results", 0))
    vc2.metric("Statistically Significant",
               sum(1 for r in results if r.get("statistically_significant")))
    vc3.metric("Robust (MC>=70)",
               sum(1 for r in results if r.get("robust")))

    top = sorted(results, key=lambda x: x.get("sharpe", -99), reverse=True)[:5]
    if top:
        st.markdown("**Top 5 by Sharpe:**")
        st.dataframe(pd.DataFrame(top)[["strategy", "symbol", "sharpe",
                                         "total_return", "win_rate", "n_trades"]],
                     use_container_width=True)

# ── Auto-Refresh ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Settings")
    auto_refresh = st.checkbox("Auto-refresh", value=False)
    refresh_interval = st.slider("Interval (s)", 5, 120, 30)
    if auto_refresh:
        import time
        time.sleep(refresh_interval)
        st.rerun()

    st.markdown("---")
    st.markdown(f"Updated: {state.get('updated_at', 'N/A')}")
    st.markdown(f"Balance: ${stats.get('balance', 0):,.2f}")

"""
Walk-Forward Optimization framework for NovaFX.

Prevents overfitting by continuously re-optimizing on rolling windows:
  1. Optimize params on train_bars (in-sample)
  2. Test on test_bars (out-of-sample)
  3. Roll forward by step_bars
  4. Repeat until data exhausted

Includes purging (embargo) to prevent train/test data leakage.
"""
import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Callable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    bar_idx: int
    direction: str
    entry: float
    sl: float
    tp: float
    outcome: str  # "TP1", "SL", "OPEN"
    pnl_pct: float


@dataclass
class WindowResult:
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    best_params: dict
    is_sharpe: float      # In-sample Sharpe
    is_pnl: float         # In-sample PnL
    is_win_rate: float
    oos_sharpe: float     # Out-of-sample Sharpe
    oos_pnl: float
    oos_win_rate: float
    oos_trades: list[TradeResult] = field(default_factory=list)
    n_is_trades: int = 0
    n_oos_trades: int = 0


@dataclass
class WalkForwardResults:
    windows: list[WindowResult]
    oos_equity: pd.Series
    oos_sharpe: float
    oos_pnl: float
    oos_win_rate: float
    efficiency_ratio: float   # OOS sharpe / IS sharpe
    param_stability: dict     # How much params changed window-to-window
    total_oos_trades: int


# ─── Optimization Metrics ──────────────────────────────────────────────

def _sharpe(pnls: list[float]) -> float:
    if not pnls or len(pnls) < 2:
        return 0.0
    arr = np.array(pnls)
    std = arr.std()
    if std == 0:
        return 0.0
    return float(arr.mean() / std * np.sqrt(len(arr)))


def _sortino(pnls: list[float]) -> float:
    if not pnls or len(pnls) < 2:
        return 0.0
    arr = np.array(pnls)
    downside = arr[arr < 0]
    if len(downside) < 1:
        return float(arr.mean() * np.sqrt(len(arr)))
    dd_std = downside.std()
    if dd_std == 0:
        return 0.0
    return float(arr.mean() / dd_std * np.sqrt(len(arr)))


def _calmar(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    drawdown = peak - cum
    max_dd = drawdown.max()
    if max_dd == 0:
        return float(cum[-1])
    return float(cum[-1] / max_dd)


def _profit_factor(pnls: list[float]) -> float:
    if not pnls:
        return 0.0
    arr = np.array(pnls)
    gross_profit = arr[arr > 0].sum()
    gross_loss = abs(arr[arr < 0].sum())
    if gross_loss == 0:
        return float(gross_profit) if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


METRICS = {
    "sharpe": _sharpe,
    "sortino": _sortino,
    "calmar": _calmar,
    "profit_factor": _profit_factor,
}


# ─── Parameter Grids ───────────────────────────────────────────────────

PARAM_GRIDS = {
    "ema_cross": {
        "fast_period": [7, 9, 12],
        "slow_period": [18, 21, 26],
    },
    "rsi_adaptive": {
        "rsi_period": [10, 14, 20],
        "buy_offset": [25, 30, 35],
        "sell_offset": [65, 70, 75],
    },
    "macd_zero": {
        "fast": [8, 12, 16],
        "slow": [21, 26, 30],
        "signal": [7, 9, 12],
    },
    "bb_reversion": {
        "bb_period": [15, 20, 25],
        "bb_std": [1.5, 2.0, 2.5],
    },
    "momentum_breakout": {
        "lookback": [10, 15, 20, 25],
        "ema_fast": [15, 20, 25],
        "ema_slow": [40, 50, 60],
    },
    "donchian_breakout": {
        "entry_period": [15, 20, 25, 30],
        "min_width": [0.001, 0.002, 0.003],
    },
    "macd_trend": {
        "fast": [8, 12, 16],
        "slow": [21, 26, 30],
        "signal": [7, 9, 12],
        "trend_period": [40, 50, 60],
    },
}


# ─── Parameterized Strategy Runners ───────────────────────────────────
# Each returns a list of (bar_index, direction) tuples for a given window

def _calc_ema(close: pd.Series, period: int) -> pd.Series:
    return close.ewm(span=period, adjust=False).mean()


def _calc_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _calc_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = _calc_ema(close, fast)
    ema_slow = _calc_ema(close, slow)
    ml = ema_fast - ema_slow
    sl = _calc_ema(ml, signal)
    return ml, sl


def _calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def _run_parameterized(df: pd.DataFrame, strategy: str, params: dict,
                       scan_step: int = 4) -> list[tuple[int, str]]:
    """Run a strategy with specific params. Returns list of (bar_idx, direction)."""
    signals = []
    close = df["close"]
    high = df["high"]
    low = df["low"]
    warmup = 60  # Skip first bars for indicator warmup

    if strategy == "ema_cross":
        fp = params.get("fast_period", 9)
        sp = params.get("slow_period", 21)
        ef = _calc_ema(close, fp)
        es = _calc_ema(close, sp)
        for i in range(max(warmup, sp + 2), len(df), scan_step):
            pf, cf = ef.iloc[i - 1], ef.iloc[i]
            ps, cs = es.iloc[i - 1], es.iloc[i]
            if any(np.isnan(v) for v in [pf, cf, ps, cs]):
                continue
            if pf <= ps and cf > cs:
                signals.append((i, "BUY"))
            elif pf >= ps and cf < cs:
                signals.append((i, "SELL"))

    elif strategy == "rsi_adaptive":
        rp = params.get("rsi_period", 14)
        bo = params.get("buy_offset", 30)
        so = params.get("sell_offset", 70)
        rsi = _calc_rsi(close, rp)
        sma50 = close.rolling(50).mean()
        for i in range(warmup, len(df), scan_step):
            v = rsi.iloc[i]
            if np.isnan(v):
                continue
            s50 = sma50.iloc[i]
            p = close.iloc[i]
            if np.isnan(s50):
                bt, st = bo, so
            elif p > s50:
                bt, st = bo + 10, so + 10
            else:
                bt, st = bo - 10, so - 10
            if v < bt:
                signals.append((i, "BUY"))
            elif v > st:
                signals.append((i, "SELL"))

    elif strategy == "macd_zero":
        ml, sl = _calc_macd(close, params.get("fast", 12), params.get("slow", 26), params.get("signal", 9))
        for i in range(warmup, len(df), scan_step):
            pm, cm = ml.iloc[i - 1], ml.iloc[i]
            ps_, cs_ = sl.iloc[i - 1], sl.iloc[i]
            if any(np.isnan(v) for v in [pm, cm, ps_, cs_]):
                continue
            if pm <= ps_ and cm > cs_ and cm >= 0:
                signals.append((i, "BUY"))
            elif pm >= ps_ and cm < cs_ and cm <= 0:
                signals.append((i, "SELL"))

    elif strategy == "bb_reversion":
        per = params.get("bb_period", 20)
        sd = params.get("bb_std", 2.0)
        mid = close.rolling(per).mean()
        std = close.rolling(per).std()
        upper = mid + sd * std
        lower = mid - sd * std
        for i in range(warmup, len(df), scan_step):
            if any(np.isnan(v) for v in [upper.iloc[i], lower.iloc[i]]):
                continue
            if close.iloc[i] <= lower.iloc[i]:
                signals.append((i, "BUY"))
            elif close.iloc[i] >= upper.iloc[i]:
                signals.append((i, "SELL"))

    elif strategy == "momentum_breakout":
        lb = params.get("lookback", 20)
        ef = params.get("ema_fast", 20)
        es = params.get("ema_slow", 50)
        ema_f = _calc_ema(close, ef)
        ema_s = _calc_ema(close, es)
        rsi = _calc_rsi(close)
        for i in range(max(warmup, lb + 1), len(df), scan_step):
            p = close.iloc[i]
            h_lb = high.iloc[i - lb:i].max()
            l_lb = low.iloc[i - lb:i].min()
            e_f = ema_f.iloc[i]
            e_s = ema_s.iloc[i]
            r = rsi.iloc[i]
            if any(np.isnan(v) for v in [h_lb, l_lb, e_f, e_s, r]):
                continue
            if p > h_lb and e_f > e_s and r < 80:
                signals.append((i, "BUY"))
            elif p < l_lb and e_f < e_s and r > 20:
                signals.append((i, "SELL"))

    elif strategy == "donchian_breakout":
        ep = params.get("entry_period", 20)
        mw = params.get("min_width", 0.002)
        for i in range(max(warmup, ep + 1), len(df), scan_step):
            p = close.iloc[i]
            eh = high.iloc[i - ep:i].max()
            el = low.iloc[i - ep:i].min()
            if np.isnan(eh) or np.isnan(el) or el <= 0:
                continue
            if (eh - el) / el < mw:
                continue
            if p > eh:
                signals.append((i, "BUY"))
            elif p < el:
                signals.append((i, "SELL"))

    elif strategy == "macd_trend":
        ml, sl = _calc_macd(close, params.get("fast", 12), params.get("slow", 26), params.get("signal", 9))
        tp = params.get("trend_period", 50)
        sma = close.rolling(tp).mean()
        for i in range(warmup, len(df), scan_step):
            pm, cm = ml.iloc[i - 1], ml.iloc[i]
            ps_, cs_ = sl.iloc[i - 1], sl.iloc[i]
            s = sma.iloc[i]
            p = close.iloc[i]
            if any(np.isnan(v) for v in [pm, cm, ps_, cs_, s]):
                continue
            if pm <= ps_ and cm > cs_ and p > s:
                signals.append((i, "BUY"))
            elif pm >= ps_ and cm < cs_ and p < s:
                signals.append((i, "SELL"))

    return signals


# ─── Trade Simulator ───────────────────────────────────────────────────

def _simulate(df: pd.DataFrame, signals: list[tuple[int, str]],
              sl_mult: float = 1.5, tp_mult: float = 3.0,
              max_bars: int = 20, cooldown: int = 16) -> list[TradeResult]:
    """Simulate trades with ATR-based SL/TP."""
    H = df["high"].values.astype(float)
    L = df["low"].values.astype(float)
    C = df["close"].values.astype(float)
    atr = _calc_atr(df["high"], df["low"], df["close"]).values

    trades = []
    last_bar = -999

    for bar_idx, direction in signals:
        if bar_idx - last_bar < cooldown:
            continue
        if bar_idx + max_bars >= len(df):
            continue

        entry = C[bar_idx]
        a = atr[bar_idx]
        if np.isnan(a) or a <= 0:
            continue

        sl_d = a * sl_mult
        tp_d = a * tp_mult
        sl = entry - sl_d if direction == "BUY" else entry + sl_d
        tp = entry + tp_d if direction == "BUY" else entry - tp_d

        # Walk forward
        outcome = "OPEN"
        pnl = 0.0
        for j in range(bar_idx + 1, min(bar_idx + max_bars + 1, len(C))):
            if direction == "BUY":
                if L[j] <= sl:
                    outcome, pnl = "SL", (sl - entry) / entry * 100
                    break
                if H[j] >= tp:
                    outcome, pnl = "TP1", (tp - entry) / entry * 100
                    break
            else:
                if H[j] >= sl:
                    outcome, pnl = "SL", (entry - sl) / entry * 100
                    break
                if L[j] <= tp:
                    outcome, pnl = "TP1", (entry - tp) / entry * 100
                    break
        else:
            last_c = C[min(bar_idx + max_bars, len(C) - 1)]
            pnl = ((last_c - entry) / entry if direction == "BUY" else (entry - last_c) / entry) * 100

        trades.append(TradeResult(bar_idx, direction, entry, sl, tp, outcome, pnl))
        last_bar = bar_idx

    return trades


# ─── Walk-Forward Optimizer ───────────────────────────────────────────

class WalkForwardOptimizer:

    def __init__(self, train_bars: int = 2000, test_bars: int = 500,
                 step_bars: int = 500, embargo_bars: int = 50):
        self.train_bars = train_bars
        self.test_bars = test_bars
        self.step_bars = step_bars
        self.embargo_bars = embargo_bars

    def _expand_grid(self, param_grid: dict) -> list[dict]:
        """Expand param_grid dict into list of all combinations."""
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combos = list(product(*values))
        return [dict(zip(keys, c)) for c in combos]

    def run(self, df: pd.DataFrame, strategy: str,
            param_grid: dict | None = None,
            metric: str = "sharpe",
            sl_mult: float = 1.5, tp_mult: float = 3.0) -> WalkForwardResults:
        """Run walk-forward optimization."""
        if param_grid is None:
            param_grid = PARAM_GRIDS.get(strategy, {})
        if not param_grid:
            raise ValueError(f"No param grid for strategy '{strategy}'")

        metric_fn = METRICS.get(metric, _sharpe)
        grid = self._expand_grid(param_grid)
        total_bars = len(df)
        min_required = self.train_bars + self.test_bars + self.embargo_bars

        if total_bars < min_required:
            raise ValueError(f"Need {min_required} bars, got {total_bars}")

        windows: list[WindowResult] = []
        all_oos_trades: list[TradeResult] = []
        window_id = 0

        offset = 0
        while offset + self.train_bars + self.embargo_bars + self.test_bars <= total_bars:
            train_start = offset
            train_end = offset + self.train_bars
            # Embargo: skip embargo_bars between train and test
            test_start = train_end + self.embargo_bars
            test_end = min(test_start + self.test_bars, total_bars)

            if test_end - test_start < 50:
                break

            train_df = df.iloc[train_start:train_end].copy().reset_index(drop=True)
            test_df = df.iloc[test_start:test_end].copy().reset_index(drop=True)

            # Optimize on train set
            best_score = -np.inf
            best_params = grid[0]
            best_is_trades = []

            for params in grid:
                sigs = _run_parameterized(train_df, strategy, params)
                trades = _simulate(train_df, sigs, sl_mult, tp_mult)
                pnls = [t.pnl_pct for t in trades]
                score = metric_fn(pnls)
                if score > best_score:
                    best_score = score
                    best_params = params
                    best_is_trades = trades

            # Test on OOS with best params
            oos_sigs = _run_parameterized(test_df, strategy, best_params)
            oos_trades = _simulate(test_df, oos_sigs, sl_mult, tp_mult)
            oos_pnls = [t.pnl_pct for t in oos_trades]
            is_pnls = [t.pnl_pct for t in best_is_trades]

            is_sharpe = _sharpe(is_pnls)
            oos_sharpe = _sharpe(oos_pnls)
            is_wr = sum(1 for t in best_is_trades if t.outcome == "TP1") / max(len(best_is_trades), 1) * 100
            oos_wr = sum(1 for t in oos_trades if t.outcome == "TP1") / max(len(oos_trades), 1) * 100

            wr = WindowResult(
                window_id=window_id,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                best_params=best_params,
                is_sharpe=is_sharpe,
                is_pnl=sum(is_pnls),
                is_win_rate=is_wr,
                oos_sharpe=oos_sharpe,
                oos_pnl=sum(oos_pnls),
                oos_win_rate=oos_wr,
                oos_trades=oos_trades,
                n_is_trades=len(best_is_trades),
                n_oos_trades=len(oos_trades),
            )
            windows.append(wr)
            all_oos_trades.extend(oos_trades)

            window_id += 1
            offset += self.step_bars

        # Stitch OOS equity curve
        oos_pnls = [t.pnl_pct for t in all_oos_trades]
        oos_equity = pd.Series(np.cumsum(oos_pnls)) if oos_pnls else pd.Series(dtype=float)

        # Aggregate metrics
        total_oos_sharpe = _sharpe(oos_pnls)
        total_is_sharpe = np.mean([w.is_sharpe for w in windows]) if windows else 0.0
        efficiency = total_oos_sharpe / total_is_sharpe if total_is_sharpe != 0 else 0.0

        total_oos_wr = sum(1 for t in all_oos_trades if t.outcome == "TP1") / max(len(all_oos_trades), 1) * 100

        # Param stability: measure how much each param changed across windows
        param_stability = {}
        if len(windows) >= 2:
            all_params = [w.best_params for w in windows]
            for key in all_params[0]:
                vals = [p[key] for p in all_params]
                if all(isinstance(v, (int, float)) for v in vals):
                    param_stability[key] = {
                        "values": vals,
                        "std": float(np.std(vals)),
                        "cv": float(np.std(vals) / np.mean(vals)) if np.mean(vals) != 0 else 0.0,
                        "n_unique": len(set(vals)),
                    }

        return WalkForwardResults(
            windows=windows,
            oos_equity=oos_equity,
            oos_sharpe=total_oos_sharpe,
            oos_pnl=sum(oos_pnls),
            oos_win_rate=total_oos_wr,
            efficiency_ratio=efficiency,
            param_stability=param_stability,
            total_oos_trades=len(all_oos_trades),
        )

"""Tests for backtest engine: win/loss counting, commission, PnL, drawdown, MTF."""

import numpy as np
import pandas as pd
import pytest

from backtester.app.core.backtest_engine import (
    COMMISSION_RATES,
    detect_market,
    filter_signals_by_mtf,
    get_commission_rate,
    get_htf_bias,
    run_backtest,
)


class TestDetectMarket:
    @pytest.mark.parametrize("symbol,expected", [
        ("EURUSD", "forex"),
        ("GBPJPY", "forex"),
        ("BTCUSD", "crypto"),
        ("ETHUSDT", "crypto"),
        ("SPX500", "indices"),
        ("NAS100", "indices"),
        ("XAUUSD", "commodities"),
        ("USOIL", "commodities"),
    ])
    def test_known_symbols(self, symbol, expected):
        assert detect_market(symbol) == expected

    def test_unknown_defaults_to_forex(self):
        assert detect_market("ZZZZZ") == "forex"


class TestCommissionRates:
    def test_forex_rate(self):
        assert get_commission_rate("EURUSD") == 0.0002

    def test_crypto_rate(self):
        assert get_commission_rate("BTCUSD") == 0.001

    def test_indices_rate(self):
        assert get_commission_rate("SPX500") == 0.0005

    def test_commodities_rate(self):
        assert get_commission_rate("XAUUSD") == 0.0002


class TestRunBacktest:
    def test_winning_buy_trade(self, backtest_df):
        # Entry at idx 0 (close=100), exit at idx 5 (close=105) -> +5% gross
        signals = [{"action": "BUY", "entry_idx": 0, "exit_idx": 5}]
        result = run_backtest(backtest_df, signals, "EURUSD", "test_strat")

        assert result.total_trades == 1
        assert result.wins == 1
        assert result.losses == 0
        assert result.win_rate == 100.0
        assert result.trades[0].pnl_pct == pytest.approx(5.0, abs=0.01)
        assert result.trades[0].result == "WIN"

    def test_losing_buy_trade(self, backtest_df):
        # Entry at idx 2 (close=104), exit at idx 4 (close=101) -> -2.88% gross
        signals = [{"action": "BUY", "entry_idx": 2, "exit_idx": 4}]
        result = run_backtest(backtest_df, signals, "EURUSD", "test_strat")

        assert result.total_trades == 1
        assert result.wins == 0
        assert result.losses == 1
        assert result.trades[0].result == "LOSS"
        assert result.trades[0].pnl_pct < 0

    def test_winning_sell_trade(self, backtest_df):
        # Entry at idx 2 (close=104), exit at idx 4 (close=101) -> short wins
        signals = [{"action": "SELL", "entry_idx": 2, "exit_idx": 4}]
        result = run_backtest(backtest_df, signals, "EURUSD", "test_strat")

        assert result.wins == 1
        assert result.trades[0].pnl_pct > 0
        assert result.trades[0].result == "WIN"

    def test_multiple_trades_win_loss_count(self, backtest_df):
        signals = [
            {"action": "BUY", "entry_idx": 0, "exit_idx": 5},   # win
            {"action": "BUY", "entry_idx": 5, "exit_idx": 4},   # loss (105->101)
            {"action": "BUY", "entry_idx": 4, "exit_idx": 9},   # win (101->112)
        ]
        result = run_backtest(backtest_df, signals, "EURUSD", "test_strat")

        assert result.total_trades == 3
        assert result.wins == 2
        assert result.losses == 1
        assert result.win_rate == pytest.approx(66.67, abs=0.01)

    def test_commission_deducted_from_pnl(self, backtest_df):
        signals = [{"action": "BUY", "entry_idx": 0, "exit_idx": 5}]
        result = run_backtest(backtest_df, signals, "EURUSD", "test_strat")
        trade = result.trades[0]

        expected_commission = COMMISSION_RATES["forex"] * 100  # as percentage
        assert trade.commission_pct == pytest.approx(expected_commission, abs=0.0001)
        assert trade.net_pnl_pct == pytest.approx(trade.pnl_pct - trade.commission_pct, abs=0.0001)

    def test_total_commission_in_phase_result(self, backtest_df):
        signals = [
            {"action": "BUY", "entry_idx": 0, "exit_idx": 5},
            {"action": "BUY", "entry_idx": 4, "exit_idx": 9},
        ]
        result = run_backtest(backtest_df, signals, "EURUSD", "test_strat")

        per_trade_commission = COMMISSION_RATES["forex"] * 100
        assert result.total_commission_pct == pytest.approx(per_trade_commission * 2, abs=0.001)

    def test_crypto_commission_higher_than_forex(self, backtest_df):
        signals = [{"action": "BUY", "entry_idx": 0, "exit_idx": 5}]
        forex_result = run_backtest(backtest_df, signals, "EURUSD", "test")
        crypto_result = run_backtest(backtest_df, signals, "BTCUSD", "test")

        assert crypto_result.trades[0].commission_pct > forex_result.trades[0].commission_pct

    def test_net_pnl_less_than_gross(self, backtest_df):
        signals = [{"action": "BUY", "entry_idx": 0, "exit_idx": 9}]
        result = run_backtest(backtest_df, signals, "EURUSD", "test_strat")

        assert result.net_pnl_pct < result.gross_pnl_pct

    def test_out_of_bounds_signals_skipped(self, backtest_df):
        signals = [{"action": "BUY", "entry_idx": 0, "exit_idx": 999}]
        result = run_backtest(backtest_df, signals, "EURUSD", "test_strat")

        assert result.total_trades == 0

    def test_empty_signals_returns_zero_trades(self, backtest_df):
        result = run_backtest(backtest_df, [], "EURUSD", "test_strat")

        assert result.total_trades == 0
        assert result.wins == 0
        assert result.losses == 0
        assert result.gross_pnl_pct == 0.0
        assert result.total_commission_pct == 0.0
        assert result.net_pnl_pct == 0.0

    def test_max_drawdown_tracked(self, backtest_df):
        # Win then loss — should register some drawdown
        signals = [
            {"action": "BUY", "entry_idx": 0, "exit_idx": 6},   # 100->108 win
            {"action": "BUY", "entry_idx": 6, "exit_idx": 4},   # 108->101 loss
        ]
        result = run_backtest(backtest_df, signals, "EURUSD", "test_strat")

        assert result.max_drawdown_pct > 0


# ── HTF helper fixtures ──────────────────────────────────────────────────────


def _make_htf_df(prices: list[float]) -> pd.DataFrame:
    closes = np.array(prices, dtype=float)
    opens = np.roll(closes, 1)
    opens[0] = closes[0]
    return pd.DataFrame({
        "open": opens,
        "high": np.maximum(opens, closes) * 1.002,
        "low": np.minimum(opens, closes) * 0.998,
        "close": closes,
        "volume": [100] * len(closes),
    })


# ── Multi-Timeframe bias ─────────────────────────────────────────────────────


class TestGetHtfBias:
    def test_bullish_bias(self):
        # Uptrend: EMA9 > EMA21
        prices = [100.0 + i * 0.3 for i in range(30)]
        assert get_htf_bias(_make_htf_df(prices)) == "BUY"

    def test_bearish_bias(self):
        # Downtrend: EMA9 < EMA21
        prices = [100.0 - i * 0.3 for i in range(30)]
        assert get_htf_bias(_make_htf_df(prices)) == "SELL"

    def test_insufficient_data_returns_none(self):
        assert get_htf_bias(_make_htf_df([100.0] * 10)) is None

    def test_none_dataframe_returns_none(self):
        assert get_htf_bias(None) is None


class TestFilterSignalsByMtf:
    def test_bullish_bias_keeps_buys(self):
        signals = [
            {"action": "BUY", "entry_idx": 0, "exit_idx": 5},
            {"action": "SELL", "entry_idx": 2, "exit_idx": 7},
            {"action": "BUY", "entry_idx": 3, "exit_idx": 8},
        ]
        kept, removed = filter_signals_by_mtf(signals, "BUY")
        assert len(kept) == 2
        assert all(s["action"] == "BUY" for s in kept)
        assert removed == 1

    def test_bearish_bias_keeps_sells(self):
        signals = [
            {"action": "BUY", "entry_idx": 0, "exit_idx": 5},
            {"action": "SELL", "entry_idx": 2, "exit_idx": 7},
        ]
        kept, removed = filter_signals_by_mtf(signals, "SELL")
        assert len(kept) == 1
        assert kept[0]["action"] == "SELL"
        assert removed == 1

    def test_none_bias_passes_all(self):
        signals = [
            {"action": "BUY", "entry_idx": 0, "exit_idx": 5},
            {"action": "SELL", "entry_idx": 2, "exit_idx": 7},
        ]
        kept, removed = filter_signals_by_mtf(signals, None)
        assert len(kept) == 2
        assert removed == 0

    def test_empty_signals(self):
        kept, removed = filter_signals_by_mtf([], "BUY")
        assert kept == []
        assert removed == 0


class TestRunBacktestMTF:
    def test_mtf_filters_counter_trend_trades(self, backtest_df):
        signals = [
            {"action": "BUY", "entry_idx": 0, "exit_idx": 5},
            {"action": "SELL", "entry_idx": 5, "exit_idx": 4},
        ]
        # Bullish HTF: only BUY should survive
        htf = _make_htf_df([100.0 + i * 0.3 for i in range(30)])
        result = run_backtest(backtest_df, signals, "EURUSD", "test", df_htf=htf)

        assert result.total_trades == 1
        assert result.trades[0].action == "BUY"
        assert result.mtf_filtered == 1

    def test_mtf_filtered_field_in_result(self, backtest_df):
        signals = [
            {"action": "BUY", "entry_idx": 0, "exit_idx": 5},
            {"action": "SELL", "entry_idx": 2, "exit_idx": 4},
            {"action": "SELL", "entry_idx": 4, "exit_idx": 9},
        ]
        # Bullish HTF: 2 SELLs should be removed
        htf = _make_htf_df([100.0 + i * 0.3 for i in range(30)])
        result = run_backtest(backtest_df, signals, "EURUSD", "test", df_htf=htf)

        assert result.mtf_filtered == 2
        assert result.total_trades == 1

    def test_no_htf_means_no_filtering(self, backtest_df):
        signals = [
            {"action": "BUY", "entry_idx": 0, "exit_idx": 5},
            {"action": "SELL", "entry_idx": 5, "exit_idx": 4},
        ]
        result = run_backtest(backtest_df, signals, "EURUSD", "test")
        assert result.total_trades == 2
        assert result.mtf_filtered == 0

    def test_insufficient_htf_data_passes_all(self, backtest_df):
        signals = [
            {"action": "BUY", "entry_idx": 0, "exit_idx": 5},
            {"action": "SELL", "entry_idx": 5, "exit_idx": 4},
        ]
        short_htf = _make_htf_df([100.0] * 5)
        result = run_backtest(backtest_df, signals, "EURUSD", "test", df_htf=short_htf)
        assert result.total_trades == 2
        assert result.mtf_filtered == 0

    def test_bearish_htf_removes_buys(self, backtest_df):
        signals = [
            {"action": "BUY", "entry_idx": 0, "exit_idx": 5},
            {"action": "SELL", "entry_idx": 5, "exit_idx": 4},
        ]
        htf = _make_htf_df([100.0 - i * 0.3 for i in range(30)])
        result = run_backtest(backtest_df, signals, "EURUSD", "test", df_htf=htf)

        assert result.total_trades == 1
        assert result.trades[0].action == "SELL"
        assert result.mtf_filtered == 1

    def test_mtf_default_zero_without_htf(self, backtest_df):
        result = run_backtest(backtest_df, [], "EURUSD", "test")
        assert result.mtf_filtered == 0

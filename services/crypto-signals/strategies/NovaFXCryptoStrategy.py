"""
NovaFX Crypto Strategy for Freqtrade.

Multi-indicator confluence: RSI + EMA crossover + Bollinger Bands.
Designed for 1H timeframe on top crypto pairs.
"""

from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter
from pandas import DataFrame
import talib.abstract as ta


class NovaFXCryptoStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "1h"
    can_short = False

    # Stoploss
    stoploss = -0.05
    trailing_stop = True
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    # ROI
    minimal_roi = {
        "0": 0.10,
        "60": 0.05,
        "120": 0.03,
        "240": 0.01,
    }

    # Hyperoptable parameters
    rsi_buy_threshold = IntParameter(20, 40, default=30, space="buy")
    rsi_sell_threshold = IntParameter(60, 80, default=70, space="sell")
    ema_fast_period = IntParameter(8, 16, default=12, space="buy")
    ema_slow_period = IntParameter(20, 34, default=26, space="buy")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate all indicators."""
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=self.ema_fast_period.value)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=self.ema_slow_period.value)

        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2, nbdevdn=2)
        dataframe["bb_upper"] = bollinger["upperband"]
        dataframe["bb_middle"] = bollinger["middleband"]
        dataframe["bb_lower"] = bollinger["lowerband"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Entry: RSI oversold + EMA bullish cross + price near lower BB."""
        dataframe.loc[
            (
                (dataframe["rsi"] < self.rsi_buy_threshold.value)
                & (dataframe["ema_fast"] > dataframe["ema_slow"])
                & (dataframe["close"] > dataframe["bb_lower"])
            ),
            "enter_long",
        ] = 1
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit: RSI overbought + EMA bearish cross."""
        dataframe.loc[
            (
                (dataframe["rsi"] > self.rsi_sell_threshold.value)
                & (dataframe["ema_fast"] < dataframe["ema_slow"])
            ),
            "exit_long",
        ] = 1
        return dataframe

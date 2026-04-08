"""Data utilities for VectorBT."""

import pandas as pd
from pathlib import Path

from src.execution.volume_handler import VolumeHandler


def load_data(symbol: str, data_dir: str = "data", timeframe: str = "1h") -> pd.DataFrame:
    """Load OHLCV data from CSV."""
    symbol_clean = symbol.replace("-", "_").replace("/", "_").lower()
    patterns = [
        f"{symbol_clean}_{timeframe}.csv",
        f"{symbol_clean}.csv",
        f"{symbol.lower()}_{timeframe}.csv",
        f"{symbol.lower()}.csv",
    ]

    data_path = Path(data_dir)
    for pattern in patterns:
        filepath = data_path / pattern
        if filepath.exists():
            df = pd.read_csv(filepath)
            break
    else:
        raise FileNotFoundError(f"No data file for {symbol} in {data_dir}. Tried: {patterns}")

    df.columns = df.columns.str.lower()

    for col in ["timestamp", "datetime", "date", "time"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
            df = df.set_index(col)
            break

    required = ["open", "high", "low", "close", "volume"]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return df[required].sort_index()


def prepare_vbt_data(data: pd.DataFrame, symbol: str,
                     normalize_volume: bool = True) -> pd.DataFrame:
    """Clean and normalize data for VBT backtesting."""
    df = data.copy().dropna()

    if normalize_volume:
        handler = VolumeHandler(symbol)
        df["volume"] = handler.normalize(df["volume"], df["close"], df["high"], df["low"])

    return df


def infer_timeframe(data: pd.DataFrame) -> str:
    """Infer timeframe from DatetimeIndex spacing."""
    if not isinstance(data.index, pd.DatetimeIndex) or len(data) < 2:
        return "1h"
    delta = data.index[1] - data.index[0]
    if delta <= pd.Timedelta(minutes=5):
        return "5m"
    if delta <= pd.Timedelta(minutes=15):
        return "15m"
    if delta <= pd.Timedelta(hours=1):
        return "1h"
    if delta <= pd.Timedelta(hours=4):
        return "4h"
    return "1d"

"""
HMM-based market regime detection for NovaFX.

Uses a Gaussian Hidden Markov Model with 3 hidden states to classify
market conditions into bull, bear, or ranging regimes. Features used:
  - Log returns
  - Rolling volatility (20-bar)
  - ADX (optional, included when OHLC available)

States are auto-labeled after training by sorting on mean return:
  highest mean return -> bull
  lowest mean return  -> bear
  middle              -> ranging
"""
import logging
import pickle
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

logger = logging.getLogger(__name__)

RegimeLabel = Literal["bull", "bear", "ranging"]

# Strategy gating per HMM regime
HMM_REGIME_ALLOWED = {
    "bull": {"ema_cross", "macd_zero", "rsi_adaptive", "momentum_breakout", "donchian_breakout", "macd_trend"},
    "bear": {"ema_cross", "macd_zero", "rsi_adaptive", "momentum_breakout", "donchian_breakout", "macd_trend"},
    "ranging": {"rsi_adaptive", "bb_reversion", "rsi_divergence"},
}

MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "regime"


def _calc_adx_feature(high: pd.Series, low: pd.Series, close: pd.Series,
                       period: int = 14) -> pd.Series:
    """Lightweight ADX for feature extraction."""
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    plus_dm[plus_dm < minus_dm] = 0
    minus_dm[minus_dm < plus_dm] = 0

    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1.0 / period, adjust=False).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


class HMMRegimeDetector:
    """3-state Gaussian HMM regime classifier."""

    def __init__(self, n_states: int = 3, n_iter: int = 200,
                 vol_window: int = 20, use_adx: bool = True):
        self.n_states = n_states
        self.n_iter = n_iter
        self.vol_window = vol_window
        self.use_adx = use_adx

        self.model: GaussianHMM | None = None
        self.state_labels: dict[int, RegimeLabel] = {}
        self.state_stats: dict[int, dict] = {}
        self.symbol: str = ""
        self._trained = False

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Build feature matrix from OHLCV DataFrame.

        Returns (n_samples, n_features) array. Features:
          [0] log_return
          [1] rolling_vol_20
          [2] adx_14 (if use_adx and OHLC available)
        """
        close = df["close"].astype(float)
        log_ret = np.log(close / close.shift(1))
        roll_vol = log_ret.rolling(self.vol_window).std()

        features = pd.DataFrame({
            "log_return": log_ret,
            "roll_vol": roll_vol,
        })

        if self.use_adx and all(c in df.columns for c in ["high", "low", "close"]):
            adx = _calc_adx_feature(
                df["high"].astype(float),
                df["low"].astype(float),
                close,
            )
            # Normalize ADX to 0-1 range for comparable feature scale
            features["adx"] = adx / 100.0

        features = features.dropna()
        return features.values, features.index

    def _label_states(self) -> None:
        """Auto-label HMM states by mean return: bull > ranging > bear."""
        means = self.model.means_[:, 0]  # column 0 = log returns
        order = np.argsort(means)  # ascending: bear, ranging, bull

        labels: list[RegimeLabel] = ["bear", "ranging", "bull"]
        self.state_labels = {int(order[i]): labels[i] for i in range(self.n_states)}

        # Collect stats per state
        for state_idx in range(self.n_states):
            label = self.state_labels[state_idx]
            covars = self.model.covars_[state_idx]
            self.state_stats[state_idx] = {
                "label": label,
                "mean_return": float(means[state_idx]),
                "volatility": float(np.sqrt(covars[0, 0])) if covars.ndim == 2 else float(np.sqrt(covars[0])),
                "start_prob": float(self.model.startprob_[state_idx]),
            }

    def train(self, df: pd.DataFrame, symbol: str = "") -> dict:
        """Train the HMM on a DataFrame with OHLCV columns.

        Args:
            df: DataFrame with at least 'close' column (and optionally high/low for ADX)
            symbol: Symbol name for logging/saving

        Returns:
            dict with training stats per state
        """
        self.symbol = symbol
        X, idx = self._build_features(df)

        if len(X) < 100:
            raise ValueError(f"Not enough data to train HMM: {len(X)} rows (need 100+)")

        # Try full covariance first, fall back to diagonal if it fails
        # (low-volatility forex pairs can produce non-positive-definite matrices)
        for cov_type in ("full", "diag"):
            try:
                self.model = GaussianHMM(
                    n_components=self.n_states,
                    covariance_type=cov_type,
                    n_iter=self.n_iter,
                    random_state=42,
                    verbose=False,
                    min_covar=1e-5,
                )
                self.model.fit(X)
                break
            except ValueError as e:
                if cov_type == "diag":
                    raise
                logger.warning("Full covariance failed for %s, retrying with diag: %s",
                               symbol, e)
        self._label_states()

        # Compute time-in-state from training data
        states = self.model.predict(X)
        total = len(states)
        for state_idx in range(self.n_states):
            count = int(np.sum(states == state_idx))
            self.state_stats[state_idx]["time_pct"] = round(count / total * 100, 1)
            self.state_stats[state_idx]["count"] = count

        self._trained = True
        logger.info("HMM trained for %s: %d samples, %d features",
                     symbol, len(X), X.shape[1])

        return {self.state_labels[k]: v for k, v in self.state_stats.items()}

    def predict(self, df: pd.DataFrame) -> tuple[RegimeLabel, float]:
        """Predict current regime from recent data.

        Args:
            df: DataFrame with at least 'close' (and high/low for ADX)

        Returns:
            (regime_label, confidence) where confidence is the posterior
            probability of the predicted state.
        """
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")

        X, idx = self._build_features(df)
        if len(X) == 0:
            return "ranging", 0.0

        # Get posterior probabilities for the last observation
        posteriors = self.model.predict_proba(X)
        last_posterior = posteriors[-1]
        predicted_state = int(np.argmax(last_posterior))
        confidence = float(last_posterior[predicted_state])
        label = self.state_labels.get(predicted_state, "ranging")

        return label, confidence

    def predict_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict regime for every bar in the DataFrame.

        Returns DataFrame with columns: regime, confidence, state_id
        indexed to match the input (after warmup period).
        """
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained.")

        X, idx = self._build_features(df)
        if len(X) == 0:
            return pd.DataFrame()

        posteriors = self.model.predict_proba(X)
        states = self.model.predict(X)

        result = pd.DataFrame(index=idx)
        result["state_id"] = states
        result["regime"] = [self.state_labels.get(int(s), "ranging") for s in states]
        result["confidence"] = [float(posteriors[i, states[i]]) for i in range(len(states))]

        return result

    def detect_transition(self, df: pd.DataFrame, lookback: int = 5) -> dict | None:
        """Detect if a regime transition occurred in the last `lookback` bars.

        Returns dict with transition info or None if no change.
        """
        series = self.predict_series(df)
        if len(series) < lookback + 1:
            return None

        recent = series["regime"].iloc[-lookback:]
        prev = series["regime"].iloc[-(lookback + 1)]

        # Check if regime changed within the lookback window
        current = recent.iloc[-1]
        if current != prev:
            return {
                "from_regime": prev,
                "to_regime": current,
                "confidence": float(series["confidence"].iloc[-1]),
                "bars_ago": int((recent != current).sum()),
            }
        return None

    def save_model(self, path: Path | str | None = None) -> Path:
        """Serialize trained model to disk."""
        if not self._trained:
            raise RuntimeError("No trained model to save.")

        if path is None:
            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            path = MODEL_DIR / f"{self.symbol.replace('/', '_').replace('-', '_')}.pkl"
        else:
            path = Path(path)

        data = {
            "model": self.model,
            "state_labels": self.state_labels,
            "state_stats": self.state_stats,
            "symbol": self.symbol,
            "n_states": self.n_states,
            "vol_window": self.vol_window,
            "use_adx": self.use_adx,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info("Saved HMM model to %s", path)
        return path

    def load_model(self, path: Path | str | None = None, symbol: str = "") -> None:
        """Load a previously trained model from disk."""
        if path is None:
            safe = symbol.replace("/", "_").replace("-", "_")
            path = MODEL_DIR / f"{safe}.pkl"
        else:
            path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"No model at {path}")

        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.state_labels = data["state_labels"]
        self.state_stats = data["state_stats"]
        self.symbol = data["symbol"]
        self.n_states = data["n_states"]
        self.vol_window = data.get("vol_window", 20)
        self.use_adx = data.get("use_adx", True)
        self._trained = True

        logger.info("Loaded HMM model for %s from %s", self.symbol, path)

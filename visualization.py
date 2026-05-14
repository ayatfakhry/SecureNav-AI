"""
anomaly_detection.py
────────────────────
Unsupervised anomaly detection layer using Isolation Forest.

This module acts as a complementary second detection layer.
It is trained only on NORMAL-labelled data and flags anything
that deviates significantly from the clean-signal distribution.

Usage
─────
  detector = AnomalyDetector()
  detector.fit(X_normal)
  scores   = detector.score(X_test)   # raw anomaly scores
  flags    = detector.predict(X_test)  # 1 = normal, -1 = anomaly

Also includes:
  - PositionDriftAnalyser: dedicated statistical model for DRIFT class
  - Ensemble scorer: combines classifier probability with anomaly score
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing  import Optional, Tuple

from sklearn.ensemble       import IsolationForest
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline

from src.feature_extraction import FEATURE_COLUMNS


# ──────────────────────────────────────────────────────────────
# Anomaly Detector (Isolation Forest)
# ──────────────────────────────────────────────────────────────
class AnomalyDetector:
    """
    Isolation Forest-based GNSS anomaly detector.

    Parameters
    ----------
    contamination : float
        Expected fraction of anomalies in training data (set low for clean data).
    n_estimators  : int
        Number of isolation trees.
    seed          : int
    """

    def __init__(
        self,
        contamination : float = 0.02,
        n_estimators  : int   = 200,
        seed          : int   = 42,
    ):
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("isoforest", IsolationForest(
                n_estimators  = n_estimators,
                contamination = contamination,
                max_samples   = "auto",
                random_state  = seed,
                n_jobs        = -1,
            )),
        ])
        self._fitted = False

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """Train on normal (clean) samples."""
        self.pipeline.fit(X)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns
        -------
        Array of {1 = inlier/normal, -1 = anomaly}.
        """
        self._check_fitted()
        return self.pipeline.predict(X)

    # ------------------------------------------------------------------
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Returns raw anomaly scores (more negative → more anomalous).
        Normalised to [0, 1] where 1 = most anomalous.
        """
        self._check_fitted()
        raw    = self.pipeline.score_samples(X)
        # Invert and normalise to [0, 1]
        normed = (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
        return 1.0 - normed

    # ------------------------------------------------------------------
    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("AnomalyDetector must be fitted before calling predict/score.")

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        print(f"  [✓] AnomalyDetector saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyDetector":
        return joblib.load(path)


# ──────────────────────────────────────────────────────────────
# Position Drift Analyser
# ──────────────────────────────────────────────────────────────
class PositionDriftAnalyser:
    """
    Statistical model dedicated to detecting position DRIFT anomalies.

    Uses a CUSUM (Cumulative Sum) change-point detector on position
    residuals across a sliding window.

    Parameters
    ----------
    window   : int   – number of epochs in rolling window
    threshold: float – CUSUM threshold (metres)
    """

    def __init__(self, window: int = 10, threshold: float = 25.0):
        self.window    = window
        self.threshold = threshold
        self._baseline_std : Optional[float] = None

    # ------------------------------------------------------------------
    def fit(self, position_series: np.ndarray) -> "PositionDriftAnalyser":
        """
        Fit on normal position-jump sequence to calibrate baseline variance.

        Parameters
        ----------
        position_series : 1-D array of per-epoch position-jump magnitudes (m).
        """
        self._baseline_mean = float(np.mean(position_series))
        self._baseline_std  = float(np.std(position_series)) + 1e-6
        return self

    # ------------------------------------------------------------------
    def detect(self, position_series: np.ndarray) -> np.ndarray:
        """
        Run CUSUM detector on a position-jump series.

        Returns
        -------
        Boolean array: True where drift is detected.
        """
        if self._baseline_std is None:
            raise RuntimeError("PositionDriftAnalyser must be fitted first.")

        n      = len(position_series)
        flags  = np.zeros(n, dtype=bool)
        cusum_pos = 0.0
        cusum_neg = 0.0
        slack  = 0.5 * self._baseline_std

        for i, x in enumerate(position_series):
            z = (x - self._baseline_mean) / self._baseline_std
            cusum_pos = max(0.0, cusum_pos + z - slack)
            cusum_neg = max(0.0, cusum_neg - z - slack)
            if cusum_pos > self.threshold or cusum_neg > self.threshold:
                flags[i] = True
                # Reset after detection
                cusum_pos = 0.0
                cusum_neg = 0.0

        return flags

    # ------------------------------------------------------------------
    def rolling_drift_score(self, position_series: np.ndarray) -> np.ndarray:
        """
        Returns a continuous drift anomaly score in [0, 1] via rolling z-score.
        """
        if self._baseline_std is None:
            raise RuntimeError("PositionDriftAnalyser must be fitted first.")

        z_scores = np.abs((position_series - self._baseline_mean) / self._baseline_std)
        # Smooth with rolling window
        scores = np.zeros_like(z_scores)
        for i in range(len(z_scores)):
            start       = max(0, i - self.window + 1)
            scores[i]   = np.mean(z_scores[start: i + 1])

        # Map to [0, 1] via sigmoid
        scores = 1.0 / (1.0 + np.exp(-0.5 * (scores - 3.0)))
        return scores


# ──────────────────────────────────────────────────────────────
# Ensemble scorer
# ──────────────────────────────────────────────────────────────
def ensemble_anomaly_score(
    clf_probabilities : np.ndarray,   # shape (n, n_classes), class 0 = NORMAL
    anomaly_scores    : np.ndarray,   # shape (n,) from AnomalyDetector
    normal_class_idx  : int = 0,
    alpha             : float = 0.6,  # weight for classifier
) -> np.ndarray:
    """
    Blend classifier-based normality probability with Isolation Forest
    anomaly score to produce a unified threat score in [0, 1].

    score = alpha * (1 - P(normal)) + (1 - alpha) * anomaly_score
    """
    clf_threat = 1.0 - clf_probabilities[:, normal_class_idx]
    return alpha * clf_threat + (1.0 - alpha) * anomaly_scores


# ──────────────────────────────────────────────────────────────
# Convenience: build from labelled DataFrame
# ──────────────────────────────────────────────────────────────
def fit_anomaly_detector_from_df(
    df              : pd.DataFrame,
    contamination   : float = 0.02,
    seed            : int = 42,
) -> AnomalyDetector:
    """Fit an AnomalyDetector using only normal samples from df."""
    normal_mask = df["label"] == "NORMAL"
    X_normal    = df.loc[normal_mask, FEATURE_COLUMNS].fillna(0.0).values
    detector    = AnomalyDetector(contamination=contamination, seed=seed)
    detector.fit(X_normal)
    return detector

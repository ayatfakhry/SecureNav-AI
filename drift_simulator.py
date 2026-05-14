"""
alert_system.py
───────────────
Real-time alert generation engine for SecureNav AI.

Produces structured Alert objects with:
  • Severity level  (INFO / WARNING / CRITICAL)
  • Threat type     (NORMAL / SPOOFING / JAMMING / DRIFT)
  • Confidence      (0–1, derived from classifier probability)
  • Anomaly score   (0–1, from Isolation Forest)
  • Recommended action
  • Rich metadata   (epoch timestamp, position, key features)

Also implements:
  • Rate-limiter   – suppresses repeated alerts within a cooldown window
  • Alert logger   – append-only JSON log for post-mission analysis
  • Summary reporter – aggregates alert counts / rates per threat class
"""

import json
import time
import datetime
from dataclasses import dataclass, field, asdict
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
SEVERITY_INFO     = "INFO"
SEVERITY_WARNING  = "WARNING"
SEVERITY_CRITICAL = "CRITICAL"

THREAT_THRESHOLDS = {
    # (min_confidence, min_anomaly_score) → severity
    "SPOOFING": [(0.90, 0.60, SEVERITY_CRITICAL),
                 (0.70, 0.40, SEVERITY_WARNING),
                 (0.40, 0.00, SEVERITY_INFO)],
    "JAMMING" : [(0.90, 0.55, SEVERITY_CRITICAL),
                 (0.65, 0.35, SEVERITY_WARNING),
                 (0.35, 0.00, SEVERITY_INFO)],
    "DRIFT"   : [(0.85, 0.50, SEVERITY_CRITICAL),
                 (0.60, 0.30, SEVERITY_WARNING),
                 (0.30, 0.00, SEVERITY_INFO)],
    "NORMAL"  : [(0.00, 0.00, SEVERITY_INFO)],
}

RECOMMENDED_ACTIONS = {
    ("SPOOFING",  SEVERITY_CRITICAL): "IMMEDIATE: Switch to dead-reckoning. Reject GNSS fix. Alert operator.",
    ("SPOOFING",  SEVERITY_WARNING) : "WARNING: Cross-check with IMU/barometric alt. Log epoch for forensics.",
    ("SPOOFING",  SEVERITY_INFO)    : "MONITOR: Elevated spoofing indicators. Increase sampling rate.",
    ("JAMMING",   SEVERITY_CRITICAL): "IMMEDIATE: Enable anti-jam antenna. Switch to backup navigation.",
    ("JAMMING",   SEVERITY_WARNING) : "WARNING: RF interference detected. Verify jammer direction with ADF.",
    ("JAMMING",   SEVERITY_INFO)    : "MONITOR: SNR degradation observed. Check RF environment.",
    ("DRIFT",     SEVERITY_CRITICAL): "IMMEDIATE: Apply ionospheric correction model. Check SBAS signal.",
    ("DRIFT",     SEVERITY_WARNING) : "WARNING: Position drift exceeds threshold. Verify with reference station.",
    ("DRIFT",     SEVERITY_INFO)    : "MONITOR: Minor drift detected. Review multipath environment.",
    ("NORMAL",    SEVERITY_INFO)    : "NOMINAL: Signal healthy. No action required.",
}


# ──────────────────────────────────────────────────────────────
# Alert data class
# ──────────────────────────────────────────────────────────────
@dataclass
class Alert:
    alert_id        : int
    timestamp_unix  : float
    timestamp_str   : str
    epoch_time      : float          # GNSS epoch timestamp
    threat_class    : str
    severity        : str
    confidence      : float          # classifier probability for predicted class
    anomaly_score   : float          # Isolation Forest score [0, 1]
    ensemble_score  : float          # blended threat score [0, 1]
    lat             : float
    lon             : float
    alt             : float
    num_satellites  : int
    snr_mean        : float
    pdop            : float
    pos_jump_m      : float
    recommended_action : str
    metadata        : Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)

    def __str__(self) -> str:
        return (
            f"[{self.severity:8s}] #{self.alert_id:05d} | "
            f"{self.timestamp_str} | "
            f"Threat: {self.threat_class:8s} | "
            f"Conf: {self.confidence:.3f} | "
            f"AnomalyScore: {self.anomaly_score:.3f} | "
            f"Pos: ({self.lat:.5f}, {self.lon:.5f}) | "
            f"{self.recommended_action}"
        )


# ──────────────────────────────────────────────────────────────
# Alert generator
# ──────────────────────────────────────────────────────────────
class AlertGenerator:
    """
    Converts classifier output + anomaly score into structured alerts.

    Parameters
    ----------
    cooldown_s   : int    – minimum seconds between repeated same-class alerts
    min_severity : str    – suppress alerts below this level ('INFO'/'WARNING'/'CRITICAL')
    log_path     : Path   – file path for JSON alert log (None = no file logging)
    """

    SEVERITY_ORDER = {SEVERITY_INFO: 0, SEVERITY_WARNING: 1, SEVERITY_CRITICAL: 2}

    def __init__(
        self,
        cooldown_s   : int  = 3,
        min_severity : str  = SEVERITY_INFO,
        log_path     : Optional[Path] = None,
    ):
        self.cooldown_s   = cooldown_s
        self.min_severity = min_severity
        self.log_path     = Path(log_path) if log_path else None
        self._counter     = 0
        self._last_alert  : Dict[str, float] = {}   # threat → last alert unix time
        self._history     : List[Alert]      = []

        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_path.exists():
                self.log_path.write_text("[]")

    # ------------------------------------------------------------------
    def generate(
        self,
        threat_class    : str,
        confidence      : float,
        anomaly_score   : float,
        ensemble_score  : float,
        row             : pd.Series,
        epoch_time      : float = 0.0,
    ) -> Optional[Alert]:
        """
        Generate an alert for a single prediction.

        Parameters
        ----------
        threat_class   : predicted class string
        confidence     : P(predicted class) from classifier
        anomaly_score  : Isolation Forest score [0,1]
        ensemble_score : blended score [0,1]
        row            : feature-row Series (contains lat, lon, snr_mean etc.)
        epoch_time     : GNSS epoch timestamp

        Returns Alert or None if suppressed by cooldown / min_severity.
        """
        severity = self._determine_severity(threat_class, confidence, anomaly_score)

        # Filter by minimum severity
        if self.SEVERITY_ORDER[severity] < self.SEVERITY_ORDER[self.min_severity]:
            return None

        # Cooldown: suppress if same threat seen very recently
        now = time.time()
        last = self._last_alert.get(threat_class, 0.0)
        if threat_class != "NORMAL" and (now - last) < self.cooldown_s:
            return None

        self._last_alert[threat_class] = now
        self._counter += 1

        action = RECOMMENDED_ACTIONS.get(
            (threat_class, severity),
            RECOMMENDED_ACTIONS.get(("NORMAL", SEVERITY_INFO), "No action.")
        )

        alert = Alert(
            alert_id          = self._counter,
            timestamp_unix    = now,
            timestamp_str     = datetime.datetime.utcfromtimestamp(now).strftime("%Y-%m-%dT%H:%M:%SZ"),
            epoch_time        = float(epoch_time),
            threat_class      = threat_class,
            severity          = severity,
            confidence        = round(float(confidence), 4),
            anomaly_score     = round(float(anomaly_score), 4),
            ensemble_score    = round(float(ensemble_score), 4),
            lat               = float(row.get("lat", 0.0)),
            lon               = float(row.get("lon", 0.0)),
            alt               = float(row.get("alt", 0.0)),
            num_satellites    = int(row.get("num_satellites", 0)),
            snr_mean          = round(float(row.get("snr_mean", 0.0)), 2),
            pdop              = round(float(row.get("pdop", 0.0)), 2),
            pos_jump_m        = round(float(row.get("pos_jump", 0.0)), 2),
            recommended_action= action,
            metadata          = {
                "jamming_score" : round(float(row.get("jamming_score",  0.0)), 4),
                "spoofing_score": round(float(row.get("spoofing_score", 0.0)), 4),
                "snr_min"       : round(float(row.get("snr_min", 0.0)),        2),
                "clock_bias"    : round(float(row.get("clock_bias", 0.0)),     2),
            },
        )

        self._history.append(alert)
        if self.log_path:
            self._append_to_log(alert)

        return alert

    # ------------------------------------------------------------------
    def _determine_severity(
        self, threat: str, confidence: float, anomaly_score: float
    ) -> str:
        rules = THREAT_THRESHOLDS.get(threat, THREAT_THRESHOLDS["NORMAL"])
        for min_conf, min_anom, sev in rules:
            if confidence >= min_conf and anomaly_score >= min_anom:
                return sev
        return SEVERITY_INFO

    # ------------------------------------------------------------------
    def _append_to_log(self, alert: Alert) -> None:
        try:
            existing = json.loads(self.log_path.read_text())
            existing.append(alert.to_dict())
            self.log_path.write_text(json.dumps(existing, indent=2))
        except Exception:
            pass   # non-fatal

    # ------------------------------------------------------------------
    def get_history(self) -> List[Alert]:
        return list(self._history)

    def clear_history(self) -> None:
        self._history.clear()


# ──────────────────────────────────────────────────────────────
# Batch processor
# ──────────────────────────────────────────────────────────────
def process_batch(
    predictions     : np.ndarray,       # (n,) int labels
    probabilities   : np.ndarray,       # (n, n_classes)
    anomaly_scores  : np.ndarray,       # (n,)
    ensemble_scores : np.ndarray,       # (n,)
    feature_df      : pd.DataFrame,     # (n, features+lat+lon etc.)
    class_names     : List[str],
    generator       : AlertGenerator,
    verbose         : bool = True,
) -> List[Alert]:
    """
    Run the alert generator over a full prediction batch.

    Returns list of non-suppressed Alert objects.
    """
    alerts = []
    for i in range(len(predictions)):
        cls_idx = int(predictions[i])
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else "UNKNOWN"
        conf = float(probabilities[i, cls_idx]) if cls_idx < probabilities.shape[1] else 0.5
        row  = feature_df.iloc[i]

        alert = generator.generate(
            threat_class   = cls_name,
            confidence     = conf,
            anomaly_score  = float(anomaly_scores[i]),
            ensemble_score = float(ensemble_scores[i]),
            row            = row,
            epoch_time     = float(row.get("timestamp", i)),
        )
        if alert:
            alerts.append(alert)
            if verbose and alert.severity in (SEVERITY_WARNING, SEVERITY_CRITICAL):
                print(f"  🚨 {alert}")

    return alerts


# ──────────────────────────────────────────────────────────────
# Summary report
# ──────────────────────────────────────────────────────────────
def alert_summary(alerts: List[Alert]) -> pd.DataFrame:
    """Aggregate alert statistics by threat class and severity."""
    if not alerts:
        return pd.DataFrame()

    records = [a.to_dict() for a in alerts]
    df = pd.DataFrame(records)

    summary = (
        df.groupby(["threat_class", "severity"])
          .agg(
              count          = ("alert_id",       "count"),
              mean_conf      = ("confidence",      "mean"),
              mean_anomaly   = ("anomaly_score",   "mean"),
              mean_ensemble  = ("ensemble_score",  "mean"),
          )
          .reset_index()
          .sort_values(["threat_class", "severity"])
    )
    return summary


def print_alert_summary(alerts: List[Alert]) -> None:
    """Print a formatted summary table to stdout."""
    summary = alert_summary(alerts)
    if summary.empty:
        print("  No alerts generated.")
        return

    sep = "─" * 75
    print(f"\n{sep}")
    print(f"  {'ALERT SUMMARY':^71}")
    print(sep)
    print(f"  {'Threat':<12} {'Severity':<12} {'Count':>6}  {'AvgConf':>8}  {'AvgAnomaly':>10}  {'AvgEnsemble':>11}")
    print(sep)
    for _, row in summary.iterrows():
        icon = {"CRITICAL": "🔴", "WARNING": "🟡", "INFO": "🟢"}.get(row["severity"], "⚪")
        print(
            f"  {row['threat_class']:<12} {icon} {row['severity']:<10} "
            f"{int(row['count']):>6}  {row['mean_conf']:>8.3f}  "
            f"{row['mean_anomaly']:>10.3f}  {row['mean_ensemble']:>11.3f}"
        )
    print(sep)
    print(f"  Total alerts: {len(alerts)}\n")

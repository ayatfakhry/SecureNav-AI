"""
feature_extraction.py
─────────────────────
Derives machine-learning-ready features from raw GNSS epoch data.

Features are computed both per-epoch (static) and across a sliding
window (temporal/differential), giving the classifier temporal context
without requiring a sequential model.

Feature groups
──────────────
  A. SNR-based          : mean, std, min, max, drop-rate, low-SNR count
  B. Pseudorange-based  : mean, std, inter-satellite consistency
  C. Position-based     : jump magnitude, velocity, acceleration
  D. Geometry-based     : PDOP, num_satellites, elevation stats
  E. Clock-based        : bias, bias rate-of-change
  F. Carrier-phase      : residual std, phase consistency
  G. Doppler-based      : std across sats, anomaly score
  H. Composite scores   : jamming_score, spoofing_score (heuristic)
"""

import numpy as np
import pandas as pd
from typing import List, Optional

from src.gnss_simulator import GNSSEpoch, epochs_to_dataframe


# ──────────────────────────────────────────────────────────────
# Per-epoch feature extraction
# ──────────────────────────────────────────────────────────────
def extract_epoch_features(epoch: GNSSEpoch, prev_epoch: Optional[GNSSEpoch] = None) -> dict:
    """
    Extract the full feature vector for a single epoch.

    Parameters
    ----------
    epoch      : Current GNSSEpoch
    prev_epoch : Previous epoch for differential features (None → zeros)
    """
    sats   = epoch.satellites
    n_sats = len(sats)

    snrs   = np.array([s.snr_db_hz            for s in sats]) if sats else np.array([0.0])
    prs    = np.array([s.pseudorange_m         for s in sats]) if sats else np.array([0.0])
    els    = np.array([s.elevation_deg         for s in sats]) if sats else np.array([0.0])
    phases = np.array([s.carrier_phase_cycles  for s in sats]) if sats else np.array([0.0])
    dops   = np.array([s.doppler_hz            for s in sats]) if sats else np.array([0.0])

    # ── A. SNR features ──────────────────────────────────────
    snr_mean   = float(np.mean(snrs))
    snr_std    = float(np.std(snrs))
    snr_min    = float(np.min(snrs))
    snr_max    = float(np.max(snrs))
    snr_range  = snr_max - snr_min
    low_snr_count = int(np.sum(snrs < 25.0))   # sats near acquisition threshold

    # ── B. Pseudorange features ──────────────────────────────
    pr_mean    = float(np.mean(prs))
    pr_std     = float(np.std(prs))
    # Inter-satellite pseudorange consistency: low variance expected for clean signal
    pr_cv      = pr_std / (pr_mean + 1e-9)     # coefficient of variation

    # ── C. Carrier-phase features ────────────────────────────
    phase_std  = float(np.std(phases))
    # Carrier-phase residual: deviation of phase from expected (pr / wavelength)
    L1_WL = 0.19029  # metres
    expected_phases = prs / L1_WL
    phase_residuals = phases - expected_phases
    phase_res_std   = float(np.std(phase_residuals))
    phase_res_mean  = float(np.mean(np.abs(phase_residuals)))

    # ── D. Doppler features ──────────────────────────────────
    doppler_std  = float(np.std(dops))
    doppler_mean = float(np.mean(np.abs(dops)))

    # ── E. Geometry features ─────────────────────────────────
    el_mean  = float(np.mean(els))
    el_min   = float(np.min(els))
    el_std   = float(np.std(els))
    pdop     = float(epoch.pdop)
    num_sats = n_sats

    # ── F. Clock features ────────────────────────────────────
    clock_bias = float(epoch.clock_bias_m)

    # ── G. Velocity features ─────────────────────────────────
    vel    = epoch.velocity_ned
    vel_h  = float(np.sqrt(vel[0]**2 + vel[1]**2))   # horizontal speed m/s
    vel_v  = float(abs(vel[2]))                        # vertical speed m/s
    vel_mag = float(np.linalg.norm(vel))

    # ── H. Differential features (require prev_epoch) ────────
    if prev_epoch is not None:
        dt = max(epoch.timestamp - prev_epoch.timestamp, 1e-6)

        # Position jump (metres, approx via lat/lon diff)
        dlat   = (epoch.lat - prev_epoch.lat) * 111_319.5
        dlon   = (epoch.lon - prev_epoch.lon) * 111_319.5 * np.cos(np.deg2rad(epoch.lat))
        dalt   = epoch.alt - prev_epoch.alt
        pos_jump = float(np.sqrt(dlat**2 + dlon**2 + dalt**2))

        # Clock bias rate-of-change
        clock_rate = float((epoch.clock_bias_m - prev_epoch.clock_bias_m) / dt)

        # SNR drop rate
        prev_snrs = np.array([s.snr_db_hz for s in prev_epoch.satellites]) if prev_epoch.satellites else snrs
        snr_drop_rate = float((np.mean(prev_snrs) - snr_mean) / dt)

        # Satellite count change
        sat_count_delta = n_sats - len(prev_epoch.satellites)

        # PDOP change rate
        pdop_rate = float((epoch.pdop - prev_epoch.pdop) / dt)
    else:
        pos_jump        = 0.0
        clock_rate      = 0.0
        snr_drop_rate   = 0.0
        sat_count_delta = 0
        pdop_rate       = 0.0

    # ── I. Heuristic composite scores ────────────────────────
    # Jamming score: high when SNR drops, satellites lost, PDOP rises
    jamming_score = (
        max(0.0, (35.0 - snr_mean) / 35.0) * 0.4
        + max(0.0, (8 - num_sats)  / 8.0)  * 0.35
        + min(1.0, pdop / 15.0)            * 0.25
    )

    # Spoofing score: high when clock jumps, phase residuals large, SNR suspiciously high
    spoofing_score = (
        min(1.0, abs(clock_bias) / 500.0)    * 0.35
        + min(1.0, phase_res_std / 1e4)       * 0.25
        + min(1.0, pos_jump      / 5000.0)    * 0.20
        + max(0.0, (snr_mean - 40.0) / 15.0) * 0.20
    )

    return {
        # SNR
        "snr_mean"          : snr_mean,
        "snr_std"           : snr_std,
        "snr_min"           : snr_min,
        "snr_max"           : snr_max,
        "snr_range"         : snr_range,
        "low_snr_count"     : low_snr_count,
        # Pseudorange
        "pr_mean"           : pr_mean,
        "pr_std"            : pr_std,
        "pr_cv"             : pr_cv,
        # Carrier phase
        "phase_std"         : phase_std,
        "phase_res_std"     : phase_res_std,
        "phase_res_mean"    : phase_res_mean,
        # Doppler
        "doppler_std"       : doppler_std,
        "doppler_mean"      : doppler_mean,
        # Geometry
        "el_mean"           : el_mean,
        "el_min"            : el_min,
        "el_std"            : el_std,
        "pdop"              : pdop,
        "num_satellites"    : num_sats,
        # Clock
        "clock_bias"        : clock_bias,
        # Velocity
        "vel_horizontal"    : vel_h,
        "vel_vertical"      : vel_v,
        "vel_magnitude"     : vel_mag,
        # Differential
        "pos_jump"          : pos_jump,
        "clock_rate"        : clock_rate,
        "snr_drop_rate"     : snr_drop_rate,
        "sat_count_delta"   : sat_count_delta,
        "pdop_rate"         : pdop_rate,
        # Composite
        "jamming_score"     : jamming_score,
        "spoofing_score"    : spoofing_score,
        # Label
        "label"             : epoch.label,
    }


# ──────────────────────────────────────────────────────────────
# Dataset-level extraction
# ──────────────────────────────────────────────────────────────
def extract_features(epochs: List[GNSSEpoch]) -> pd.DataFrame:
    """
    Extract features for an ordered list of epochs.
    Differential features use the preceding epoch as context.

    Returns
    -------
    pd.DataFrame with one row per epoch and all feature columns.
    """
    rows = []
    prev = None
    for ep in epochs:
        rows.append(extract_epoch_features(ep, prev))
        prev = ep
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# Feature names (for external use)
# ──────────────────────────────────────────────────────────────
FEATURE_COLUMNS = [
    "snr_mean", "snr_std", "snr_min", "snr_max", "snr_range", "low_snr_count",
    "pr_mean", "pr_std", "pr_cv",
    "phase_std", "phase_res_std", "phase_res_mean",
    "doppler_std", "doppler_mean",
    "el_mean", "el_min", "el_std",
    "pdop", "num_satellites",
    "clock_bias",
    "vel_horizontal", "vel_vertical", "vel_magnitude",
    "pos_jump", "clock_rate", "snr_drop_rate", "sat_count_delta", "pdop_rate",
    "jamming_score", "spoofing_score",
]

TARGET_COLUMN = "label"
CLASS_NAMES   = ["NORMAL", "SPOOFING", "JAMMING", "DRIFT"]

"""
jamming_simulator.py
────────────────────
Simulates RF jamming attacks on a GNSS receiver.

Three jamming profiles are modelled:
  1. WIDEBAND   – Broadband noise raises the receiver noise floor uniformly
  2. NARROWBAND – Continuous-wave (CW) jammer on L1 causes SNR collapse for
                  satellites near the jammer frequency
  3. SWEEP      – Frequency-swept jammer periodically wipes out satellites

Physical effects encoded:
  - SNR degradation (primary indicator)
  - Satellite loss (num_satellites drops as SNR < acquisition threshold)
  - PDOP increase (fewer, poorly-distributed visible satellites)
  - Pseudorange noise floor elevation
  - Receiver clock instability under heavy jamming
"""

import copy
import numpy as np
from typing import List, Optional

from src.gnss_simulator import GNSSEpoch, SNR_MIN_THRESHOLD


# ──────────────────────────────────────────────
# Jamming parameters
# ──────────────────────────────────────────────
class JammingConfig:
    # Wideband
    WIDEBAND_SNR_DROP_MEAN = 18.0   # dB-Hz mean suppression
    WIDEBAND_SNR_DROP_STD  =  4.0
    WIDEBAND_PR_NOISE_STD  = 15.0   # m

    # Narrowband CW – only satellites within ±bandwidth are hit
    NARROWBAND_BANDWIDTH_MHZ = 2.0
    NARROWBAND_HIT_SNR_DROP  = 30.0  # dB-Hz (severe)
    NARROWBAND_MISS_SNR_DROP =  3.0  # dB-Hz (minor out-of-band effect)

    # Sweep
    SWEEP_RATE_HZ_PER_S = 5e6       # 5 MHz/s sweep rate
    SWEEP_AFFECTED_FRACTION = 0.5   # fraction of sats hit per epoch

    # Common
    JAM_CLOCK_NOISE_STD = 10.0      # m – receiver clock stress
    SNR_FLOOR           = 5.0       # absolute minimum SNR dB-Hz
    ACQUISITION_THRESHOLD = SNR_MIN_THRESHOLD  # below this → satellite lost


# ──────────────────────────────────────────────
# Internal attack functions
# ──────────────────────────────────────────────
def _apply_wideband(epoch: GNSSEpoch, rng: np.random.Generator, cfg: JammingConfig) -> GNSSEpoch:
    """Broadband noise: all satellites lose SNR uniformly."""
    ep = copy.deepcopy(epoch)
    drop = float(rng.normal(cfg.WIDEBAND_SNR_DROP_MEAN, cfg.WIDEBAND_SNR_DROP_STD))

    surviving = []
    for sat in ep.satellites:
        sat.snr_db_hz = max(cfg.SNR_FLOOR, sat.snr_db_hz - drop + float(rng.normal(0.0, 2.0)))
        sat.pseudorange_m += float(rng.normal(0.0, cfg.WIDEBAND_PR_NOISE_STD))
        if sat.snr_db_hz >= cfg.ACQUISITION_THRESHOLD:
            surviving.append(sat)

    ep.satellites = surviving
    ep.pdop = _recompute_pdop(ep)
    ep.clock_bias_m += float(rng.normal(0.0, cfg.JAM_CLOCK_NOISE_STD))
    ep.label = "JAMMING"
    return ep


def _apply_narrowband(epoch: GNSSEpoch, rng: np.random.Generator, cfg: JammingConfig,
                      jammer_el_deg: float = 45.0) -> GNSSEpoch:
    """CW jammer: satellites near jammer direction are severely suppressed."""
    ep = copy.deepcopy(epoch)
    bw = cfg.NARROWBAND_BANDWIDTH_MHZ

    surviving = []
    for sat in ep.satellites:
        # Satellites near jammer elevation are hit harder
        el_diff = abs(sat.elevation_deg - jammer_el_deg)
        if el_diff < bw * 3:   # within ~6 deg of jammer beam
            drop = cfg.NARROWBAND_HIT_SNR_DROP * (1.0 - el_diff / (bw * 3))
            drop += float(rng.normal(0.0, 3.0))
        else:
            drop = cfg.NARROWBAND_MISS_SNR_DROP + float(rng.normal(0.0, 1.0))

        sat.snr_db_hz = max(cfg.SNR_FLOOR, sat.snr_db_hz - drop)
        sat.pseudorange_m += float(rng.normal(0.0, 5.0 + drop * 0.5))
        if sat.snr_db_hz >= cfg.ACQUISITION_THRESHOLD:
            surviving.append(sat)

    ep.satellites = surviving
    ep.pdop = _recompute_pdop(ep)
    ep.clock_bias_m += float(rng.normal(0.0, cfg.JAM_CLOCK_NOISE_STD * 0.5))
    ep.label = "JAMMING"
    return ep


def _apply_sweep(epoch: GNSSEpoch, rng: np.random.Generator, cfg: JammingConfig,
                 sweep_phase: float = 0.0) -> GNSSEpoch:
    """Swept jammer: a fraction of satellites are periodically blanked."""
    ep = copy.deepcopy(epoch)

    n_affected = max(1, int(len(ep.satellites) * cfg.SWEEP_AFFECTED_FRACTION
                            * (0.5 + 0.5 * np.sin(sweep_phase))))

    indices = rng.choice(len(ep.satellites), size=min(n_affected, len(ep.satellites)), replace=False)
    surviving = []
    for i, sat in enumerate(ep.satellites):
        if i in indices:
            sat.snr_db_hz = cfg.SNR_FLOOR + float(rng.uniform(0.0, 5.0))
            sat.pseudorange_m += float(rng.normal(0.0, 20.0))
            if sat.snr_db_hz >= cfg.ACQUISITION_THRESHOLD:
                surviving.append(sat)
        else:
            sat.snr_db_hz = max(cfg.SNR_FLOOR,
                                sat.snr_db_hz - float(rng.normal(5.0, 2.0)))
            surviving.append(sat)

    ep.satellites = surviving
    ep.pdop = _recompute_pdop(ep)
    ep.clock_bias_m += float(rng.normal(0.0, cfg.JAM_CLOCK_NOISE_STD * 0.3))
    ep.label = "JAMMING"
    return ep


def _recompute_pdop(ep: GNSSEpoch) -> float:
    """Recompute PDOP after satellite loss."""
    if len(ep.satellites) < 4:
        return 20.0 + float(np.random.default_rng().uniform(0, 10))
    elevs = np.array([s.elevation_deg for s in ep.satellites])
    el    = np.deg2rad(elevs)
    H     = np.column_stack([np.cos(el), np.zeros(len(el)), np.sin(el), np.ones(len(el))])
    try:
        cov  = np.linalg.inv(H.T @ H)
        pdop = float(np.sqrt(np.trace(cov[:3, :3])))
        return np.clip(pdop, 1.0, 30.0)
    except np.linalg.LinAlgError:
        return 15.0


# ──────────────────────────────────────────────
# High-level interface
# ──────────────────────────────────────────────
class JammingSimulator:
    """
    Applies jamming attacks to a stream of GNSSEpoch objects.

    Parameters
    ----------
    mode : str
        'wideband' | 'narrowband' | 'sweep' | 'mixed'
    seed : int
    """

    MODES = ("wideband", "narrowband", "sweep", "mixed")

    def __init__(self, mode: str = "mixed", seed: int = 13, cfg: Optional[JammingConfig] = None):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode        = mode
        self.rng         = np.random.default_rng(seed)
        self.cfg         = cfg or JammingConfig()
        self._sweep_phase = 0.0
        # Random jammer elevation (narrowband)
        self._jammer_el  = float(self.rng.uniform(20.0, 70.0))

    def attack(self, epoch: GNSSEpoch) -> GNSSEpoch:
        """Apply the configured jamming mode to a single epoch."""
        self._sweep_phase += 0.15  # advance sweep phase

        if self.mode == "wideband":
            return _apply_wideband(epoch, self.rng, self.cfg)
        elif self.mode == "narrowband":
            return _apply_narrowband(epoch, self.rng, self.cfg, self._jammer_el)
        elif self.mode == "sweep":
            return _apply_sweep(epoch, self.rng, self.cfg, self._sweep_phase)
        else:  # mixed
            choice = self.rng.choice(["wideband", "narrowband", "sweep"])
            if choice == "wideband":
                return _apply_wideband(epoch, self.rng, self.cfg)
            elif choice == "narrowband":
                return _apply_narrowband(epoch, self.rng, self.cfg, self._jammer_el)
            else:
                return _apply_sweep(epoch, self.rng, self.cfg, self._sweep_phase)

    def generate_dataset(self, base_epochs: List[GNSSEpoch]) -> List[GNSSEpoch]:
        """Apply jamming attacks to a list of clean epochs."""
        return [self.attack(ep) for ep in base_epochs]

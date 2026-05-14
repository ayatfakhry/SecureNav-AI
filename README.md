"""
drift_simulator.py
──────────────────
Simulates GNSS position drift anomalies caused by:
  1. Multipath interference   – signal reflections off buildings/terrain
  2. Ionospheric delay buildup – TEC (Total Electron Content) storms
  3. Tropospheric delay        – humidity / pressure fronts
  4. Receiver clock drift      – oscillator aging / temperature effects
  5. Urban canyon effect       – severe multipath in dense environments

Drift differs from spoofing: the receiver is NOT under attack.
Errors grow slowly and organically from environmental physics.
The label is "DRIFT".
"""

import copy
import numpy as np
from typing import List, Optional

from src.gnss_simulator import GNSSEpoch, EARTH_RADIUS_M


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
class DriftConfig:
    # Multipath
    MULTIPATH_REFL_STD_M     = 3.0    # pseudorange error std per reflection (m)
    MULTIPATH_MAX_REFLECTIONS = 3

    # Ionospheric
    IONO_INITIAL_DELAY_M     = 2.0
    IONO_GROWTH_RATE_M_EPOCH = 0.15   # m per epoch (storm buildup)
    IONO_MAX_DELAY_M         = 40.0

    # Tropospheric
    TROPO_INITIAL_DELAY_M    = 1.0
    TROPO_GROWTH_RATE        = 0.05
    TROPO_MAX_DELAY_M        = 15.0

    # Clock drift
    CLOCK_DRIFT_RATE_M_EPOCH = 0.08   # m/epoch (slow oscillator aging)
    CLOCK_DRIFT_NOISE_STD    = 0.02

    # Urban canyon
    URBAN_SNR_PENALTY_DB     = 8.0    # dB-Hz loss
    URBAN_SAT_LOSS_PROB      = 0.25   # probability a satellite is blocked

    # Position noise amplification under drift
    POS_NOISE_SCALE          = 4.0    # multiplier on normal position noise


# ──────────────────────────────────────────────────────────────
# Stateful drift injector
# ──────────────────────────────────────────────────────────────
class DriftSimulator:
    """
    Stateful simulator that accumulates physical drift errors over time.

    Parameters
    ----------
    mode : str
        'multipath' | 'ionospheric' | 'tropospheric' | 'clock' | 'urban' | 'mixed'
    seed : int
    cfg  : DriftConfig (optional override)
    """

    MODES = ("multipath", "ionospheric", "tropospheric", "clock", "urban", "mixed")

    def __init__(
        self,
        mode : str = "mixed",
        seed : int = 99,
        cfg  : Optional[DriftConfig] = None,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.rng  = np.random.default_rng(seed)
        self.cfg  = cfg or DriftConfig()

        # Accumulated state
        self._iono_delay_m  = self.cfg.IONO_INITIAL_DELAY_M
        self._tropo_delay_m = self.cfg.TROPO_INITIAL_DELAY_M
        self._clock_accum_m = 0.0
        self._epoch_count   = 0

    # ------------------------------------------------------------------
    def inject(self, epoch: GNSSEpoch) -> GNSSEpoch:
        """Apply the configured drift to a single epoch and return it."""
        self._epoch_count += 1
        ep = copy.deepcopy(epoch)

        if self.mode == "multipath":
            ep = self._apply_multipath(ep)
        elif self.mode == "ionospheric":
            ep = self._apply_ionospheric(ep)
        elif self.mode == "tropospheric":
            ep = self._apply_tropospheric(ep)
        elif self.mode == "clock":
            ep = self._apply_clock_drift(ep)
        elif self.mode == "urban":
            ep = self._apply_urban_canyon(ep)
        else:  # mixed — layer multiple effects
            ep = self._apply_multipath(ep)
            if self._epoch_count % 3 == 0:
                ep = self._apply_ionospheric(ep)
            if self._epoch_count % 5 == 0:
                ep = self._apply_tropospheric(ep)
            ep = self._apply_clock_drift(ep)

        # Add amplified position noise
        pos_noise = float(self.rng.normal(0.0, 1e-5 * self.cfg.POS_NOISE_SCALE))
        ep.lat += pos_noise
        ep.lon += pos_noise * 0.7
        ep.alt += float(self.rng.normal(0.0, 1.5 * self.cfg.POS_NOISE_SCALE))

        ep.label = "DRIFT"
        return ep

    # ------------------------------------------------------------------
    def _apply_multipath(self, ep: GNSSEpoch) -> GNSSEpoch:
        """Add pseudorange errors from signal reflections."""
        n_reflections = int(self.rng.integers(1, self.cfg.MULTIPATH_MAX_REFLECTIONS + 1))
        for sat in ep.satellites:
            mp_error = sum(
                float(self.rng.normal(0.0, self.cfg.MULTIPATH_REFL_STD_M))
                for _ in range(n_reflections)
            )
            sat.pseudorange_m       += mp_error
            sat.carrier_phase_cycles += mp_error / 0.19029  # ~L1 wavelength
            sat.snr_db_hz            = max(15.0, sat.snr_db_hz - abs(mp_error) * 0.3)

        # Multipath degrades PDOP slightly
        ep.pdop = min(20.0, ep.pdop * float(self.rng.uniform(1.0, 1.3)))
        return ep

    # ------------------------------------------------------------------
    def _apply_ionospheric(self, ep: GNSSEpoch) -> GNSSEpoch:
        """Build up ionospheric delay storm."""
        self._iono_delay_m = min(
            self.cfg.IONO_MAX_DELAY_M,
            self._iono_delay_m + self.cfg.IONO_GROWTH_RATE_M_EPOCH
                                + float(self.rng.normal(0.0, 0.05))
        )
        delay = self._iono_delay_m

        for sat in ep.satellites:
            # Iono delay is elevation-dependent: larger at low elevation
            el_factor = 1.0 / max(0.1, np.sin(np.deg2rad(sat.elevation_deg)))
            el_factor = np.clip(el_factor, 1.0, 4.0)
            sat.pseudorange_m += delay * el_factor + float(self.rng.normal(0.0, 0.5))
            # Phase advance (opposite sign to pseudorange for iono)
            sat.carrier_phase_cycles -= (delay * el_factor) / 0.19029

        # Position computation biased by iono
        iono_pos_bias = delay * 1e-7  # converts to ~degrees
        ep.lat += float(self.rng.normal(iono_pos_bias, iono_pos_bias * 0.3))
        ep.lon += float(self.rng.normal(iono_pos_bias, iono_pos_bias * 0.3))
        return ep

    # ------------------------------------------------------------------
    def _apply_tropospheric(self, ep: GNSSEpoch) -> GNSSEpoch:
        """Add tropospheric wet/dry delay."""
        self._tropo_delay_m = min(
            self.cfg.TROPO_MAX_DELAY_M,
            self._tropo_delay_m + self.cfg.TROPO_GROWTH_RATE
                                 + float(self.rng.normal(0.0, 0.02))
        )
        delay = self._tropo_delay_m

        for sat in ep.satellites:
            el_factor = 1.0 / max(0.1, np.sin(np.deg2rad(sat.elevation_deg)))
            el_factor = np.clip(el_factor, 1.0, 3.0)
            sat.pseudorange_m += delay * el_factor * float(self.rng.uniform(0.9, 1.1))

        ep.alt -= delay * 0.6  # altitude biased downward
        return ep

    # ------------------------------------------------------------------
    def _apply_clock_drift(self, ep: GNSSEpoch) -> GNSSEpoch:
        """Accumulate receiver clock drift."""
        self._clock_accum_m += (
            self.cfg.CLOCK_DRIFT_RATE_M_EPOCH
            + float(self.rng.normal(0.0, self.cfg.CLOCK_DRIFT_NOISE_STD))
        )
        ep.clock_bias_m += self._clock_accum_m

        # Clock drift affects all pseudoranges
        for sat in ep.satellites:
            sat.pseudorange_m += self._clock_accum_m * float(self.rng.uniform(0.98, 1.02))

        return ep

    # ------------------------------------------------------------------
    def _apply_urban_canyon(self, ep: GNSSEpoch) -> GNSSEpoch:
        """Model urban canyon: lost satellites + SNR penalty."""
        surviving = []
        for sat in ep.satellites:
            if float(self.rng.random()) < self.cfg.URBAN_SAT_LOSS_PROB:
                continue  # satellite blocked by building
            sat.snr_db_hz = max(15.0,
                sat.snr_db_hz - self.cfg.URBAN_SNR_PENALTY_DB
                              - float(self.rng.uniform(0.0, 4.0))
            )
            sat.pseudorange_m += float(self.rng.normal(0.0, 5.0))
            surviving.append(sat)

        ep.satellites = surviving if len(surviving) >= 2 else ep.satellites[:3]
        ep.pdop = min(25.0, ep.pdop * float(self.rng.uniform(1.5, 3.0)))
        return ep

    # ------------------------------------------------------------------
    def generate_dataset(self, base_epochs: List[GNSSEpoch]) -> List[GNSSEpoch]:
        """Inject drift into a list of clean epochs."""
        return [self.inject(ep) for ep in base_epochs]

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset accumulated state (call between independent runs)."""
        self._iono_delay_m  = self.cfg.IONO_INITIAL_DELAY_M
        self._tropo_delay_m = self.cfg.TROPO_INITIAL_DELAY_M
        self._clock_accum_m = 0.0
        self._epoch_count   = 0

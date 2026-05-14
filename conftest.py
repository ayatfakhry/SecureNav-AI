"""
gnss_simulator.py
─────────────────
Simulates realistic GNSS receiver measurements for clean / normal operation.

Each epoch produces a GNSSMeasurement containing:
  - Receiver position (lat, lon, alt) with realistic noise
  - Per-satellite SNR, pseudorange, carrier-phase, elevation/azimuth
  - Derived scalars: PDOP, clock bias, velocity, timestamp
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
EARTH_RADIUS_M = 6_371_000.0
SPEED_OF_LIGHT = 2.998e8          # m/s
L1_FREQ_HZ     = 1_575_420_000.0  # GPS L1 carrier
L1_WAVELENGTH  = SPEED_OF_LIGHT / L1_FREQ_HZ  # ~0.1903 m

# Typical urban / open-sky SNR range (dB-Hz)
SNR_OPEN_SKY_MEAN = 42.0
SNR_OPEN_SKY_STD  =  3.0
SNR_MIN_THRESHOLD = 20.0  # Below this → likely jammed

# Pseudorange noise (1-sigma, metres)
PR_NOISE_STD = 2.5


# ──────────────────────────────────────────────
# Data Classes
# ──────────────────────────────────────────────
@dataclass
class SatelliteMeasurement:
    prn: int
    elevation_deg: float
    azimuth_deg: float
    snr_db_hz: float
    pseudorange_m: float
    carrier_phase_cycles: float
    doppler_hz: float


@dataclass
class GNSSEpoch:
    timestamp: float                              # POSIX seconds
    lat: float                                    # degrees
    lon: float                                    # degrees
    alt: float                                    # metres
    velocity_ned: np.ndarray                      # [N, E, D] m/s
    clock_bias_m: float                           # metres (receiver clock offset)
    pdop: float
    satellites: List[SatelliteMeasurement] = field(default_factory=list)
    label: str = "NORMAL"


# ──────────────────────────────────────────────
# Helper utilities
# ──────────────────────────────────────────────
def _elevation_weighted_snr(elevation_deg: float, rng: np.random.Generator) -> float:
    """Higher elevation → better SNR."""
    el_rad = np.deg2rad(np.clip(elevation_deg, 5.0, 90.0))
    base   = SNR_OPEN_SKY_MEAN - 10.0 * np.log10(1.0 / np.sin(el_rad))
    base   = np.clip(base, SNR_MIN_THRESHOLD + 1.0, 55.0)
    return float(rng.normal(base, SNR_OPEN_SKY_STD))


def _compute_pdop(elevations_deg: np.ndarray) -> float:
    """
    Approximate PDOP from satellite elevation angles using
    the simplified geometry matrix.
    """
    if len(elevations_deg) < 4:
        return 99.0  # Degenerate geometry
    el = np.deg2rad(elevations_deg)
    # Build unit LOS vectors (simplified planar model)
    H = np.column_stack([
        np.cos(el),
        np.zeros(len(el)),
        np.sin(el),
        np.ones(len(el))   # clock column
    ])
    try:
        cov = np.linalg.inv(H.T @ H)
        pdop = float(np.sqrt(np.trace(cov[:3, :3])))
        return np.clip(pdop, 1.0, 20.0)
    except np.linalg.LinAlgError:
        return 5.0


# ──────────────────────────────────────────────
# Main Simulator
# ──────────────────────────────────────────────
class GNSSSimulator:
    """
    Simulates a GNSS receiver tracking a moving platform.

    Parameters
    ----------
    lat0, lon0 : float
        Initial position in decimal degrees.
    alt0 : float
        Initial altitude in metres.
    num_satellites : tuple
        (min, max) visible satellites per epoch (randomised each epoch).
    dt : float
        Epoch interval in seconds.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        lat0: float = 37.7749,
        lon0: float = -122.4194,
        alt0: float = 10.0,
        num_satellites: Tuple[int, int] = (7, 12),
        dt: float = 1.0,
        seed: int = 42,
    ):
        self.lat     = lat0
        self.lon     = lon0
        self.alt     = alt0
        self.num_sat = num_satellites
        self.dt      = dt
        self.rng     = np.random.default_rng(seed)
        self._t      = 0.0

        # Receiver motion model (slow random-walk)
        self._vel_ned = np.array([0.5, 0.3, 0.0])  # m/s initial
        self._clock_bias_m = float(self.rng.normal(0.0, 5.0))
        self._clock_drift  = float(self.rng.normal(0.0, 0.01))  # m/s

        # Satellite constellation (PRNs and rough elevations, azimuth)
        self._constellation = self._init_constellation()

    # ------------------------------------------------------------------
    def _init_constellation(self) -> List[dict]:
        """Generate a fixed-sky satellite constellation."""
        sats = []
        for prn in range(1, 33):
            el  = float(self.rng.uniform(10.0, 85.0))
            az  = float(self.rng.uniform(0.0, 360.0))
            range_m = EARTH_RADIUS_M * float(self.rng.uniform(20_000_000 / EARTH_RADIUS_M,
                                                               25_000_000 / EARTH_RADIUS_M))
            sats.append({"prn": prn, "el": el, "az": az, "range_m": range_m})
        return sats

    # ------------------------------------------------------------------
    def _update_position(self):
        """Random-walk dynamics for the receiver."""
        # Small velocity perturbation
        self._vel_ned += self.rng.normal(0.0, 0.05, size=3)
        self._vel_ned  = np.clip(self._vel_ned, -5.0, 5.0)

        # Convert NED velocity to lat/lon/alt increments
        delta_n = self._vel_ned[0] * self.dt
        delta_e = self._vel_ned[1] * self.dt
        delta_d = self._vel_ned[2] * self.dt

        self.lat += np.rad2deg(delta_n / EARTH_RADIUS_M)
        self.lon += np.rad2deg(delta_e / (EARTH_RADIUS_M * np.cos(np.deg2rad(self.lat))))
        self.alt -= delta_d   # D is downward

        # Clock model
        self._clock_bias_m += self._clock_drift * self.dt
        self._clock_drift   += float(self.rng.normal(0.0, 0.001))

    # ------------------------------------------------------------------
    def _select_visible_sats(self) -> List[dict]:
        """Pick a random visible subset of the constellation."""
        n = int(self.rng.integers(self.num_sat[0], self.num_sat[1] + 1))
        return list(self.rng.choice(self._constellation, size=n, replace=False))

    # ------------------------------------------------------------------
    def generate_epoch(self, label: str = "NORMAL") -> GNSSEpoch:
        """Generate a single clean GNSS epoch."""
        self._update_position()

        visible = self._select_visible_sats()
        elevations = np.array([s["el"] for s in visible])

        sat_measurements = []
        for s in visible:
            snr     = _elevation_weighted_snr(s["el"], self.rng)
            true_range = s["range_m"]
            pr      = true_range + self._clock_bias_m + float(self.rng.normal(0.0, PR_NOISE_STD))
            phase   = pr / L1_WAVELENGTH + float(self.rng.normal(0.0, 0.02))
            doppler = float(self.rng.normal(0.0, 50.0))  # Hz

            sat_measurements.append(SatelliteMeasurement(
                prn=s["prn"],
                elevation_deg=s["el"],
                azimuth_deg=s["az"],
                snr_db_hz=snr,
                pseudorange_m=pr,
                carrier_phase_cycles=phase,
                doppler_hz=doppler,
            ))

        epoch = GNSSEpoch(
            timestamp    = self._t,
            lat          = self.lat + float(self.rng.normal(0.0, 1e-6)),
            lon          = self.lon + float(self.rng.normal(0.0, 1e-6)),
            alt          = self.alt + float(self.rng.normal(0.0, 0.5)),
            velocity_ned = self._vel_ned.copy(),
            clock_bias_m = self._clock_bias_m,
            pdop         = _compute_pdop(elevations),
            satellites   = sat_measurements,
            label        = label,
        )
        self._t += self.dt
        return epoch

    # ------------------------------------------------------------------
    def generate_dataset(self, n_epochs: int = 1000) -> List[GNSSEpoch]:
        """Generate n_epochs of normal GNSS data."""
        return [self.generate_epoch("NORMAL") for _ in range(n_epochs)]


# ──────────────────────────────────────────────
# Serialisation helper
# ──────────────────────────────────────────────
def epochs_to_dataframe(epochs: List[GNSSEpoch]) -> pd.DataFrame:
    """Flatten a list of GNSSEpoch objects into a Pandas DataFrame."""
    rows = []
    for ep in epochs:
        snrs = [s.snr_db_hz for s in ep.satellites]
        prs  = [s.pseudorange_m for s in ep.satellites]
        els  = [s.elevation_deg for s in ep.satellites]
        dops = [s.doppler_hz for s in ep.satellites]
        phases = [s.carrier_phase_cycles for s in ep.satellites]

        rows.append({
            "timestamp"          : ep.timestamp,
            "lat"                : ep.lat,
            "lon"                : ep.lon,
            "alt"                : ep.alt,
            "vel_n"              : ep.velocity_ned[0],
            "vel_e"              : ep.velocity_ned[1],
            "vel_d"              : ep.velocity_ned[2],
            "clock_bias_m"       : ep.clock_bias_m,
            "pdop"               : ep.pdop,
            "num_satellites"     : len(ep.satellites),
            "snr_mean"           : np.mean(snrs) if snrs else np.nan,
            "snr_std"            : np.std(snrs)  if snrs else np.nan,
            "snr_min"            : np.min(snrs)  if snrs else np.nan,
            "snr_max"            : np.max(snrs)  if snrs else np.nan,
            "pr_mean"            : np.mean(prs)  if prs  else np.nan,
            "pr_std"             : np.std(prs)   if prs  else np.nan,
            "el_mean"            : np.mean(els)  if els  else np.nan,
            "el_min"             : np.min(els)   if els  else np.nan,
            "doppler_std"        : np.std(dops)  if dops else np.nan,
            "phase_std"          : np.std(phases) if phases else np.nan,
            "label"              : ep.label,
        })
    return pd.DataFrame(rows)

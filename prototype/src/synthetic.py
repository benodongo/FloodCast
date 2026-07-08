"""Synthetic SWMM-like reference ("reality") generator.

Until real inundation sensors or SWMM output are available, we generate a
physically-based synthetic flood response driven by the *real* observed
rainfall. The synthetic catchment is an ensemble of sub-catchment
storage-overflow reservoirs ("chokepoints"), each with its own runoff
coefficient, infiltration loss, conveyance capacity, storage capacity and
travel-time lag. Surface flooding at a chokepoint is the overflow above its
storage capacity.

This richer, heterogeneous, noisy process is what the lumped GRC surrogate
cannot fully capture -- which is precisely what the ML residual learner is
meant to recover. IMPORTANT: because the target is synthetic, results
demonstrate the *method*, not validated real-world skill.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return x
    out = np.zeros_like(x)
    out[k:] = x[:-k]
    return out


def generate_truth(forcing: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Return per-hour synthetic flood depth (mm) and a binary flood label.

    Parameters
    ----------
    forcing : DataFrame with a ``rain_mm`` column indexed by hourly time.
    """
    rng = np.random.default_rng(seed)
    rain = forcing["rain_mm"].to_numpy(dtype=float)
    n = rain.shape[0]
    cfg = config.SYNTH

    total_flood = np.zeros(n)
    depth_by_sub = np.zeros((cfg["n_subcatchments"], n))

    for j in range(cfg["n_subcatchments"]):
        cr = rng.uniform(*cfg["runoff_coeff_range"])
        infil = rng.uniform(*cfg["infil_rate_range"])
        pipe = rng.uniform(*cfg["pipe_capacity_range"])
        scap = rng.uniform(*cfg["storage_capacity_range"])
        lag = rng.integers(cfg["routing_lag_hours_range"][0],
                           cfg["routing_lag_hours_range"][1] + 1)

        r = _lag(rain, int(lag))
        # Effective runoff: impervious fraction minus an infiltration loss,
        # with mild nonlinearity so intense bursts run off proportionally more.
        intensity_gain = 1.0 + 0.02 * r                     # nonlinear boost
        eff = np.clip(cr * r * intensity_gain - infil, 0.0, None)

        S = 0.0
        depth = np.zeros(n)
        for t in range(n):
            S += eff[t]
            q = min(pipe, S)          # conveyance out, capacity-limited
            S -= q
            if S > scap:              # surface flooding = overflow above storage
                depth[t] = S - scap
                S = scap
        depth_by_sub[j] = depth
        total_flood += depth

    # Aggregate chokepoint flooding: emphasise the worst-affected chokepoints
    # (localised urban flooding), using the mean of the two deepest per hour.
    k_worst = min(2, cfg["n_subcatchments"])
    sorted_desc = np.sort(depth_by_sub, axis=0)[::-1]
    flood_mm = sorted_desc[:k_worst].mean(axis=0)

    # Heteroskedastic observation noise: larger when flooding is larger.
    noise = rng.normal(0.0, cfg["noise_sigma"] * (0.3 + flood_mm))
    flood_obs = np.clip(flood_mm + noise, 0.0, None)

    out = pd.DataFrame(
        {
            "flood_mm": flood_obs,
            "flood_mm_clean": flood_mm,
            "flood_event": (flood_obs >= cfg["flood_threshold_mm"]).astype(int),
        },
        index=forcing.index,
    )
    # Per-chokepoint depths (used for the spatial risk map).
    for j in range(cfg["n_subcatchments"]):
        out[f"choke_{j}"] = depth_by_sub[j]
    return out


if __name__ == "__main__":  # pragma: no cover
    from .data import build_forcing

    f = build_forcing()
    y = generate_truth(f)
    print(y.describe())
    print("flood-event rate", round(y["flood_event"].mean(), 4))
    print("max depth mm", round(y["flood_mm"].max(), 2))

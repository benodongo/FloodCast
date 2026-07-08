"""Feature engineering for the residual learner.

Builds rainfall-history and hydrologic-state features that are predictive of
short-lead urban flooding: rolling accumulations, antecedent precipitation
index (API), burst intensity, dry-spell length, plus the GRC internal state.
Targets are the residual (truth - GRC flood) at each lead time 1..6 h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def build_features(forcing: pd.DataFrame, grc: pd.DataFrame) -> pd.DataFrame:
    rain = forcing["rain_mm"]
    df = pd.DataFrame(index=forcing.index)

    df["rain"] = rain
    # Rolling accumulations over several windows.
    for w in config.FEATURE_LAGS:
        df[f"accum_{w}h"] = rain.rolling(w, min_periods=1).sum()
    # Peak burst intensity in recent windows.
    for w in [3, 6, 12]:
        df[f"peak_{w}h"] = rain.rolling(w, min_periods=1).max()

    # Antecedent Precipitation Index (exponential decay memory of wetness).
    k = 0.9
    api = np.zeros(len(rain))
    r = rain.to_numpy()
    acc = 0.0
    for i in range(len(r)):
        acc = k * acc + r[i]
        api[i] = acc
    df["api"] = api

    # Dry-spell length (hours since last rain) - capped for stability.
    dry = np.zeros(len(r))
    c = 0
    for i in range(len(r)):
        c = 0 if r[i] > 0 else c + 1
        dry[i] = min(c, 168)
    df["dry_spell_h"] = dry

    # Regional wetness context from CHIRPS.
    df["chirps_ctx"] = forcing["chirps_ctx_mm"].to_numpy()

    # GRC internal state (physics prior for the ML learner).
    df["grc_flood"] = grc["grc_flood_mm"].to_numpy()
    df["grc_storage"] = grc["grc_storage_mm"].to_numpy()
    df["grc_discharge"] = grc["grc_discharge_mm"].to_numpy()

    # Time-of-day / seasonality (storm climatology).
    hour = forcing.index.hour.to_numpy()
    doy = forcing.index.dayofyear.to_numpy()
    df["sin_hour"] = np.sin(2 * np.pi * hour / 24)
    df["cos_hour"] = np.cos(2 * np.pi * hour / 24)
    df["sin_doy"] = np.sin(2 * np.pi * doy / 365)
    df["cos_doy"] = np.cos(2 * np.pi * doy / 365)

    return df.fillna(0.0)


def build_targets(truth: pd.DataFrame, grc: pd.DataFrame) -> dict[int, pd.Series]:
    """Residual targets (truth - GRC) at each forecast lead time."""
    resid = truth["flood_mm"] - grc["grc_flood_mm"]
    targets = {}
    for h in config.LEAD_HOURS:
        targets[h] = resid.shift(-h)   # residual h hours ahead
    return targets


FEATURE_COLUMNS = None  # populated at runtime by build_features order

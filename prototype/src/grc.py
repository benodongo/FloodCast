"""Generalized Runoff Capacity (GRC) surrogate model.

A compact, physically-plausible lumped reservoir with:

    dS/dt : S_{t} = S_{t-1} + c_r * R_t' - f(S) - Q_out
    Q_out = min(min(C_g, C_p), S)          (conveyance, capacity limited)
    flood = max(0, S - S_cap)              (surface overflow)

Improvements over the original single-line equation in the write-up:
  * explicit infiltration / loss term f(S)
  * a travel-time lag on the rainfall input (timing)

It is deliberately *lumped* (one store), so it cannot represent the spatial
heterogeneity of the synthetic multi-chokepoint reality -- leaving a structured
residual for the ML learner.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


@dataclass
class GRCParams:
    runoff_coeff: float = config.GRC["runoff_coeff"]
    infil_rate: float = config.GRC["infil_rate"]
    pipe_capacity: float = config.GRC["pipe_capacity"]
    gutter_capacity: float = config.GRC["gutter_capacity"]
    storage_capacity: float = config.GRC["storage_capacity"]
    routing_lag_hours: int = config.GRC["routing_lag_hours"]


def _lag(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        return x
    out = np.zeros_like(x)
    out[k:] = x[:-k]
    return out


def run_grc(rain_mm: np.ndarray, params: GRCParams | None = None) -> dict:
    """Run the GRC model over a rainfall series.

    Returns a dict with flood depth, storage, discharge and the mass-balance
    residual (which should be ~0 by construction and is used as a physics
    penalty signal for the ML stage).
    """
    p = params or GRCParams()
    r = _lag(np.asarray(rain_mm, dtype=float), int(p.routing_lag_hours))
    cap_out = min(p.gutter_capacity, p.pipe_capacity)

    n = r.shape[0]
    S = 0.0
    flood = np.zeros(n)
    storage = np.zeros(n)
    discharge = np.zeros(n)
    mb_res = np.zeros(n)

    for t in range(n):
        inflow = max(p.runoff_coeff * r[t] - p.infil_rate, 0.0)
        S_pre = S + inflow
        q = min(cap_out, S_pre)
        S_post = S_pre - q
        f = 0.0
        if S_post > p.storage_capacity:
            f = S_post - p.storage_capacity
            S_post = p.storage_capacity
        # Mass balance: inflow == dS + discharge + flood spill.
        mb_res[t] = inflow - ((S_post - S) + q + f)
        S = S_post
        flood[t] = f
        storage[t] = S
        discharge[t] = q

    return {
        "grc_flood_mm": flood,
        "grc_storage_mm": storage,
        "grc_discharge_mm": discharge,
        "grc_mb_residual": mb_res,
        "grc_pipe_capacity": p.pipe_capacity,
    }


def grc_frame(forcing: pd.DataFrame, params: GRCParams | None = None) -> pd.DataFrame:
    res = run_grc(forcing["rain_mm"].to_numpy(), params)
    return pd.DataFrame(res, index=forcing.index)


if __name__ == "__main__":  # pragma: no cover
    from .data import build_forcing

    f = build_forcing()
    g = grc_frame(f)
    print(g.describe())
    print("max mass-balance residual:", float(np.abs(g["grc_mb_residual"]).max()))

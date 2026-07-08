"""Central configuration for the South C flood-forecast prototype.

All tunable parameters live here so the pipeline is reproducible and easy to
adjust. Values are deliberately physically plausible rather than calibrated to
a specific real catchment (no observed inundation data exists yet).
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths.  config.py lives in  ...\ML\prototype\src\config.py
#   parents[0] = ...\ML\prototype\src
#   parents[1] = ...\ML\prototype
#   parents[2] = ...\ML            <- raw data files live here
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
DATA_DIR = _HERE.parents[2]
PROTO_DIR = _HERE.parents[1]
OUT_DIR = PROTO_DIR / "outputs"
FIG_DIR = OUT_DIR / "figures"

HOURLY_CSV = DATA_DIR / "TA00026_Hourly_Clean.csv"
CHIRPS_XLSX = DATA_DIR / "South_C_Daily_CHIRPS_2016_2026.xlsx"
MAF_XLSX = DATA_DIR / "TA00026.MAF.xlsx"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Synthetic "reality" (SWMM-like reference) parameters
# ---------------------------------------------------------------------------
SYNTH = {
    "n_subcatchments": 6,
    "runoff_coeff_range": (0.55, 0.9),
    "infil_rate_range": (0.3, 1.2),
    "pipe_capacity_range": (2.0, 5.0),
    "storage_capacity_range": (3.0, 10.0),
    "routing_lag_hours_range": (0, 2),
    "noise_sigma": 0.5,
    "flood_threshold_mm": 2.0,
}

# ---------------------------------------------------------------------------
# GRC surrogate (lumped, deliberately simpler than the "reality")
# ---------------------------------------------------------------------------
GRC = {
    "runoff_coeff": 0.7,
    "infil_rate": 1.0,
    "pipe_capacity": 5.0,
    "gutter_capacity": 6.0,
    "storage_capacity": 15.0,
    "routing_lag_hours": 1,
}

GRC_SWEEP = {
    "runoff_coeff": (0.6, 0.85),
    "pipe_capacity": (4.0, 6.5),
    "storage_capacity": (12.0, 18.0),
}

# ---------------------------------------------------------------------------
# Forecasting setup
# ---------------------------------------------------------------------------
LEAD_HOURS = [1, 2, 3, 4, 5, 6]
FEATURE_LAGS = [1, 2, 3, 6, 12, 24]

N_ENSEMBLE = 8
TEST_FRACTION = 0.2

ENSEMBLE = {
    "n_rain_scenarios": 25,
    "n_param_draws": 25,
    "rain_perturb_cv": 0.35,
}

# ---------------------------------------------------------------------------
# Decision analysis (cost-loss / REV)
# ---------------------------------------------------------------------------
DECISION = {
    "cost_loss_ratios": [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
}

"""Data loading and cleaning.

Handles the three raw rainfall sources and returns a clean, gap-free hourly
rainfall series for South C (station TA00026). Also repairs the corrupted
day-of-year style dates in the CHIRPS spreadsheet.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_hourly_gauge() -> pd.DataFrame:
    """Load the cleaned hourly gauge series (primary model forcing)."""
    df = pd.read_csv(config.HOURLY_CSV, parse_dates=["time"])
    df = df.rename(columns={"precip_mm": "rain_mm"}).sort_values("time")
    df = df.set_index("time")
    # Enforce a continuous hourly index (fill any gaps with 0 rainfall).
    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    df = df.reindex(full)
    df["rain_mm"] = df["rain_mm"].fillna(0.0).clip(lower=0.0)
    df.index.name = "time"
    return df


def _repair_chirps_dates(raw: pd.DataFrame) -> pd.DataFrame:
    """Fix rows like ``2026-05-149`` (year-month-dayofyear leakage).

    Valid rows parse straight away. Broken rows encode a day-of-year in the
    final token, so we reconstruct the true calendar date from Jan 1 of the
    stated year.
    """
    dates = []
    for val in raw["date"].astype(str):
        ts = pd.to_datetime(val, errors="coerce")
        if pd.notna(ts):
            dates.append(ts)
            continue
        # Broken pattern: YYYY-MM-DOY  -> use year + day-of-year offset.
        parts = val.split("-")
        try:
            year = int(parts[0])
            doy = int(parts[-1])
            ts = pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy - 1)
        except Exception:
            ts = pd.NaT
        dates.append(ts)
    out = raw.copy()
    out["date"] = dates
    out = out.dropna(subset=["date"]).sort_values("date")
    return out


def load_chirps_daily() -> pd.DataFrame:
    """Load CHIRPS satellite daily rainfall with date repair."""
    raw = pd.read_excel(config.CHIRPS_XLSX)
    raw = _repair_chirps_dates(raw)
    raw = raw.rename(columns={"precipitation_mm": "chirps_mm"})
    raw = raw.set_index("date")[["chirps_mm"]]
    raw["chirps_mm"] = raw["chirps_mm"].fillna(0.0).clip(lower=0.0)
    return raw


def build_forcing() -> pd.DataFrame:
    """Assemble the model forcing table: hourly gauge rain + CHIRPS context.

    CHIRPS is daily, so its value is broadcast to each hour of the day and used
    only as a slow-varying regional-wetness context feature, never as the
    primary sub-hourly signal.
    """
    hourly = load_hourly_gauge()
    chirps = load_chirps_daily()

    daily_key = hourly.index.normalize()
    chirps_aligned = chirps.reindex(chirps.index.normalize().unique())
    ctx = pd.Series(
        chirps["chirps_mm"].reindex(daily_key).to_numpy(),
        index=hourly.index,
        name="chirps_ctx_mm",
    ).fillna(0.0)

    forcing = hourly.copy()
    forcing["chirps_ctx_mm"] = ctx.to_numpy()
    return forcing


if __name__ == "__main__":  # pragma: no cover - quick manual check
    f = build_forcing()
    print(f.shape)
    print(f.head())
    print("rain total mm", round(f["rain_mm"].sum(), 1))
    print("wet hours", int((f["rain_mm"] > 0).sum()))

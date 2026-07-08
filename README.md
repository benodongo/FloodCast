# FloodCast — South C Urban Flood Nowcasting

A hybrid **physics + machine-learning** flood-forecasting system for **South C, Nairobi**,
with a modern web dashboard for live forecasts, ensemble uncertainty, decision
support, a spatial risk map and an auto-generated operator alert bulletin.

It couples a **Generalized Runoff Capacity (GRC)** hydrologic core with an
**uncertainty-aware deep-ensemble ML residual learner**, then propagates
uncertainty through a **Monte-Carlo ensemble** and turns probabilities into
actions via **cost–loss / relative-economic-value (REV)** decision analysis.

> ⚠️ **Honesty note.** No observed inundation data exists for this catchment yet,
> so a physically-based **synthetic SWMM-like "reality"** is generated from the
> **real rainfall** to provide flood-depth labels. Results demonstrate that the
> *method is sound and calibrated* — they are **not** validated real-world flood
> skill. Swap `prototype/src/synthetic.py` for real SWMM output or sensor
> observations to obtain operational skill estimates.

## Data

Driven by real rainfall in this repo:

| File | Description |
|------|-------------|
| `TA00026_Hourly_Clean.csv` | Cleaned hourly rain gauge (station TA00026), 2015–2025 |
| `TA00026.MAF.xlsx` | Raw 5-minute gauge readings with quality flags |
| `South_C_Daily_CHIRPS_2016_2026.xlsx` | Daily CHIRPS satellite rainfall (context) |

## Quick start

```powershell
cd prototype
pip install -r requirements.txt

# 1) Run the full offline pipeline (metrics + figures)
python run_prototype.py

# 2) Launch the interactive web dashboard
python -m app.server
# then open http://127.0.0.1:5000
```

The server trains all models once at startup (~30 s) and then serves fast JSON
queries.

## Deploy on Render.com

The repo includes a [`render.yaml`](render.yaml) blueprint, so Render can deploy
straight from git:

1. Push this repo to GitHub (already at `benodongo/FloodCast`).
2. On Render → **New → Blueprint**, point it at the repo. Render reads
   `render.yaml` and provisions a Python web service.
3. It runs `pip install -r requirements.txt` (in `prototype/`) and starts
   `gunicorn app.server:app` bound to `$PORT`.

Models are trained once on import (gunicorn `--preload`), so the app serves a
ready dashboard. Note: training is memory-intensive; the free plan (512 MB) may
be tight — bump the plan if the worker is killed during boot.

## What's inside

**Modelling pipeline** (`prototype/src/`)

| Module | Role |
|--------|------|
| `data.py` | Load/clean hourly gauge, repair corrupted CHIRPS dates |
| `synthetic.py` | Physically-based multi-chokepoint flood "truth" from real rain |
| `grc.py` | GRC storage–overflow core (infiltration, routing lag, mass balance) |
| `features.py` | Rainfall-history + hydrologic-state features; residual targets (lead 1–6 h) |
| `residual_model.py` | Deep-ensemble ML residual learner with aleatory + epistemic variance |
| `ensemble.py` | Monte-Carlo ensemble (~5000 draws) + law-of-total-variance decomposition |
| `evaluation.py` | Skill (RMSE/CRPS), calibration (coverage/reliability), cost–loss / REV |
| `chokepoints.py` | Named South C drainage chokepoints with coordinates |

**Web application** (`prototype/app/`)

- `service.py` — `FloodForecastService`: overview, timeseries, events, metrics,
  Monte-Carlo forecast, risk profile, contingency/REV, driver importance,
  per-chokepoint map risk and formatted alert bulletin.
- `server.py` — Flask API (`/api/overview`, `/forecast`, `/map`, `/bulletin`, …).
- `templates/` + `static/` — single-page dark dashboard (Chart.js + Leaflet).

Dashboard views: **Dashboard · Live Forecast · Ensemble & Uncertainty ·
Decision Support · Risk Map & Bulletin · Model Performance · Decision Value**.

## Example results (synthetic labels)

- GRC-only RMSE ≈ **2.19 mm** → hybrid (GRC + ML) RMSE ≈ **0.73 mm** at lead 1 h
- ROC-AUC **0.87** (lead 1 h) decaying to ~0.64 (lead 6 h) — realistic skill decay
- 90% prediction-interval coverage ≈ **0.90** — well calibrated
- Relative economic value up to **0.58** vs climatology

## Methodology → code map

- `ΔS = c_r R_t − f(S) − Q_out` → `src/grc.py` (with infiltration + routing)
- Physics-aware residual learning + non-negativity → `src/residual_model.py`
- `σ²_total = σ²_aleatory + σ²_epistemic` → `src/ensemble.py::monte_carlo_forecast`
- `p* = C/L`, REV → `src/evaluation.py::relative_economic_value`

See [`prototype/README.md`](prototype/README.md) for pipeline-level detail.

## Disclaimer

Demonstration / research prototype on **synthetic flood labels**. Not an official
warning system.

# South C Hybrid Flood-Forecast Prototype

A complete, runnable prototype implementing the methodology:
**GRC hydrologic core → uncertainty-aware ML residual learner → ensemble
forecasting with variance decomposition → cost–loss / REV decision analysis.**

It is driven by the **real rainfall data** in this workspace (station `TA00026`
hourly gauge + `South C` CHIRPS satellite). Because no observed inundation data
exists yet, a physically-based **synthetic SWMM-like "reality"** is generated
from the real rainfall to provide flood-depth labels.

> ⚠️ **Important honesty note:** the flood target is *synthetic*. These results
> demonstrate that the *method works and is calibrated*, **not** validated
> real-world flood skill. Replace `src/synthetic.py` with real SWMM output or
> sensor observations to obtain operational skill estimates.

## How to run

```powershell
cd prototype
pip install -r requirements.txt
python run_prototype.py
```

## Pipeline (`run_prototype.py`)

| Step | Module | What it does |
|------|--------|--------------|
| 1 | `src/data.py` | Load + clean hourly gauge, repair corrupted CHIRPS dates |
| 2 | `src/synthetic.py` | Generate synthetic multi-chokepoint flood "truth" |
| 3 | `src/grc.py` | Generalized Runoff Capacity core (storage–overflow, infiltration, routing lag, mass balance) |
| 4 | `src/features.py` | Rainfall-history + hydrologic-state features; residual targets (lead 1–6 h) |
| 5 | `src/residual_model.py` | Deep-ensemble ML residual learner with aleatory + epistemic variance |
| 6 | `src/ensemble.py` | Monte-Carlo ensemble (~5000 draws): rainfall perturbations × GRC param sweep × ML members; law-of-total-variance decomposition |
| 7 | `src/evaluation.py` | Skill (RMSE/CRPS), calibration (coverage, reliability), cost–loss / REV |

## Outputs (`outputs/`)

- `metrics.json` — all skill, calibration and REV numbers per lead time
- `predictions_lead1h.csv` — per-hour forecast mean, σ, flood probability
- `figures/` — forecast event, reliability diagram, REV curve, variance
  decomposition, Monte-Carlo ensemble histogram

## Example results (synthetic)

- GRC-only RMSE ≈ **2.19 mm** → hybrid (GRC + ML) RMSE ≈ **0.73 mm** at lead 1 h
- ROC-AUC **0.87** (lead 1 h) decaying to ~0.64 (lead 6 h) — realistic skill decay
- 90% interval coverage ≈ **0.90** — well calibrated
- Relative economic value up to **0.58** vs climatology

## Where the methodology maps to code

- `ΔS = c_r R_t − f(S) − Q_out` → `src/grc.py` (with added infiltration + routing)
- Composite physics-aware loss → deep-ensemble + `project_physical` constraint in `src/residual_model.py`
- `σ²_total = σ²_aleatory + σ²_epistemic` → `src/ensemble.py::monte_carlo_forecast`
- `p* = C/L`, REV → `src/evaluation.py::relative_economic_value`

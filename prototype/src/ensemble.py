"""Ensemble forecasting and predictive-variance decomposition.

Two complementary paths:

1. ``analytic_forecast`` -- fast, closed-form per-hour prediction over the whole
   evaluation period. Uses the trained residual ensemble (which already yields
   epistemic + aleatory variance) added to the GRC flood.

2. ``monte_carlo_forecast`` -- full ensemble for a single forecast instance,
   combining the three uncertainty sources described in the methodology:
     * stochastic rainfall perturbations (aleatory forcing uncertainty),
     * GRC parameter sweep (epistemic / parameter uncertainty),
     * ML deep-ensemble members (epistemic model uncertainty),
   producing ~n_rain * n_param * n_members predictive draws (~5000).

The total predictive variance is decomposed via the law of total variance:

    mu_ens      = mean_m(mu_m)
    sigma^2_ens = mean_m(sigma_m^2)            # aleatory
                + var_m(mu_m)                  # epistemic
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .grc import GRCParams, run_grc
from .residual_model import ResidualEnsemble, project_physical


def analytic_forecast(
    grc_flood: np.ndarray,
    X: np.ndarray,
    model: ResidualEnsemble,
) -> pd.DataFrame:
    """Per-hour probabilistic flood forecast for one lead time."""
    pred = model.predict(X)
    mu = project_physical(grc_flood, pred["resid_mu"])
    sigma = np.sqrt(pred["var_total"])
    return pd.DataFrame(
        {
            "flood_pred_mu": mu,
            "sigma_total": sigma,
            "var_aleatory": pred["var_aleatory"],
            "var_epistemic": pred["var_epistemic"],
        }
    )


def _perturb_rainfall(rain: np.ndarray, n: int, cv: float, rng) -> np.ndarray:
    """Multiplicative lognormal perturbations of a rainfall sequence."""
    sigma = np.sqrt(np.log(1 + cv ** 2))
    mu = -0.5 * sigma ** 2                       # keep mean ~1
    factors = rng.lognormal(mu, sigma, size=(n, 1))
    return rain[None, :] * factors


def _sample_params(rng) -> GRCParams:
    s = config.GRC_SWEEP
    return GRCParams(
        runoff_coeff=rng.uniform(*s["runoff_coeff"]),
        pipe_capacity=rng.uniform(*s["pipe_capacity"]),
        storage_capacity=rng.uniform(*s["storage_capacity"]),
    )


def monte_carlo_forecast(
    rain_window: np.ndarray,
    feat_row: np.ndarray,
    model: ResidualEnsemble,
    seed: int = config.SEED,
) -> dict:
    """Full ensemble for a single forecast instance.

    ``rain_window`` is the recent rainfall sequence feeding the GRC; the flood
    prediction is taken at its final step. ``feat_row`` is the feature vector
    for the ML residual (shape [n_features]).
    """
    rng = np.random.default_rng(seed)
    ecfg = config.ENSEMBLE
    rain_scen = _perturb_rainfall(
        rain_window, ecfg["n_rain_scenarios"], ecfg["rain_perturb_cv"], rng
    )

    member_mu = []      # per (rain, param, ml-member) predictive mean
    member_var = []     # per predictive aleatory variance
    X = feat_row.reshape(1, -1)
    ml_pred = model.predict(X)                      # member-wise residuals
    ml_member_resid = ml_pred["member_preds"][:, 0]  # [n_members]
    ml_aleatory = float(ml_pred["var_aleatory"][0])

    for s in range(ecfg["n_rain_scenarios"]):
        for _ in range(ecfg["n_param_draws"]):
            p = _sample_params(rng)
            grc_out = run_grc(rain_scen[s], p)
            grc_flood_end = grc_out["grc_flood_mm"][-1]
            for resid in ml_member_resid:
                mu = max(0.0, grc_flood_end + resid)
                member_mu.append(mu)
                member_var.append(ml_aleatory)

    member_mu = np.asarray(member_mu)
    member_var = np.asarray(member_var)

    mu_ens = member_mu.mean()
    var_aleatory = member_var.mean()               # mean of member variances
    var_epistemic = member_mu.var()                # variance of member means
    var_total = var_aleatory + var_epistemic

    return {
        "n_draws": member_mu.size,
        "mu_ens": mu_ens,
        "var_aleatory": var_aleatory,
        "var_epistemic": var_epistemic,
        "var_total": var_total,
        "sigma_total": np.sqrt(var_total),
        "draws": member_mu,
    }

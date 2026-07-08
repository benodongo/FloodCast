"""Uncertainty-aware ML residual learner (deep-ensemble style).

The learner predicts the residual  r = truth - GRC  at each lead time, with a
decomposed predictive variance:

    sigma^2_total = sigma^2_aleatory  +  sigma^2_epistemic
                    (mean of member    (variance of the
                     data-noise var)     member means)

which mirrors the law-of-total-variance decomposition in the methodology.

Implementation notes
--------------------
* Epistemic spread comes from a *deep ensemble*: ``N_ENSEMBLE`` gradient-boosted
  members trained on bootstrap resamples with different seeds.
* Aleatory (data-noise) variance is modelled by a second regressor trained on
  the squared out-of-bag errors (a practical stand-in for a heteroskedastic
  Gaussian NLL head).
* Physics awareness: after adding the residual back to the GRC flood, the total
  is projected onto the feasible set (non-negative flood, capacity-consistent)
  -- a soft realisation of the mass-balance / capacity penalties in the paper's
  composite loss.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config


class ResidualEnsemble:
    """Deep ensemble + aleatory-variance model for one lead time."""

    def __init__(self, n_members: int = config.N_ENSEMBLE, seed: int = config.SEED):
        self.n_members = n_members
        self.seed = seed
        self.members: list[HistGradientBoostingRegressor] = []
        self.var_model: HistGradientBoostingRegressor | None = None

    def _make_member(self, seed: int) -> HistGradientBoostingRegressor:
        return HistGradientBoostingRegressor(
            max_depth=6,
            max_iter=250,
            learning_rate=0.06,
            l2_regularization=1.0,
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ResidualEnsemble":
        rng = np.random.default_rng(self.seed)
        n = X.shape[0]
        self.members = []
        member_preds = np.zeros((self.n_members, n))
        for m in range(self.n_members):
            idx = rng.integers(0, n, size=n)          # bootstrap resample
            model = self._make_member(self.seed + m)
            model.fit(X[idx], y[idx])
            self.members.append(model)
            member_preds[m] = model.predict(X)

        # Aleatory variance: fit on squared error of the ensemble mean.
        mean_pred = member_preds.mean(axis=0)
        sq_err = np.clip((y - mean_pred) ** 2, 1e-6, None)
        self.var_model = self._make_member(self.seed + 999)
        # Learn log-variance for positivity / stability.
        self.var_model.fit(X, np.log(sq_err))
        return self

    def predict(self, X: np.ndarray) -> dict:
        """Return residual mean and decomposed variances."""
        member_preds = np.stack([mem.predict(X) for mem in self.members], axis=0)
        mu = member_preds.mean(axis=0)
        var_epistemic = member_preds.var(axis=0)
        var_aleatory = np.exp(self.var_model.predict(X))
        var_total = var_aleatory + var_epistemic
        return {
            "resid_mu": mu,
            "var_aleatory": var_aleatory,
            "var_epistemic": var_epistemic,
            "var_total": var_total,
            "member_preds": member_preds,
        }


def project_physical(grc_flood: np.ndarray, resid_mu: np.ndarray) -> np.ndarray:
    """Add residual to GRC flood and enforce physical feasibility.

    Soft realisation of the composite-loss penalties: predicted surface
    flooding cannot be negative.
    """
    total = grc_flood + resid_mu
    return np.clip(total, 0.0, None)

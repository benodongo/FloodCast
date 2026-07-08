"""Evaluation: forecast skill, calibration and decision value.

Covers deterministic skill (MAE/RMSE), probabilistic skill (CRPS, coverage,
Brier score, ROC-AUC), calibration (reliability), and the decision-oriented
cost-loss / relative economic value (REV) analysis from the methodology.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm
from sklearn.metrics import brier_score_loss, roc_auc_score


# --------------------------------------------------------------------------- #
# Deterministic + probabilistic skill
# --------------------------------------------------------------------------- #
def deterministic_skill(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
    }


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS for a Gaussian predictive distribution."""
    sigma = np.clip(sigma, 1e-6, None)
    z = (y - mu) / sigma
    crps = sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    return float(np.mean(crps))


def coverage(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, level: float = 0.9) -> float:
    """Empirical coverage of a central prediction interval."""
    z = norm.ppf(0.5 + level / 2)
    lo, hi = mu - z * sigma, mu + z * sigma
    return float(np.mean((y >= lo) & (y <= hi)))


def exceedance_prob(mu: np.ndarray, sigma: np.ndarray, threshold: float) -> np.ndarray:
    """P(flood depth > threshold) under a Gaussian predictive distribution."""
    sigma = np.clip(sigma, 1e-6, None)
    return 1.0 - norm.cdf((threshold - mu) / sigma)


# --------------------------------------------------------------------------- #
# Classification skill + calibration
# --------------------------------------------------------------------------- #
def classification_skill(events: np.ndarray, probs: np.ndarray) -> dict:
    out = {"base_rate": float(np.mean(events)),
           "brier": float(brier_score_loss(events, probs))}
    if 0 < events.sum() < len(events):
        out["roc_auc"] = float(roc_auc_score(events, probs))
    else:
        out["roc_auc"] = float("nan")
    return out


def reliability_curve(events: np.ndarray, probs: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(probs, bins) - 1, 0, n_bins - 1)
    mean_pred, obs_freq, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() > 0:
            mean_pred.append(probs[mask].mean())
            obs_freq.append(events[mask].mean())
            counts.append(int(mask.sum()))
    return np.array(mean_pred), np.array(obs_freq), np.array(counts)


# --------------------------------------------------------------------------- #
# Decision analysis: cost-loss / relative economic value
# --------------------------------------------------------------------------- #
def _expected_expense(events: np.ndarray, act: np.ndarray, C: float, L: float) -> float:
    """Mean expense: cost C if action taken, else loss L on missed events."""
    expense = np.where(act, C, np.where(events == 1, L, 0.0))
    return float(np.mean(expense))


def relative_economic_value(events: np.ndarray, probs: np.ndarray,
                            cost_loss_ratios) -> list[dict]:
    """REV of the probabilistic forecast vs climatology across C/L ratios.

    Normalised so L = 1 and C = C/L. Action taken when p > p* = C/L.
    REV = (E_clim - E_forecast) / (E_clim - E_perfect).
    """
    base_rate = float(np.mean(events))
    results = []
    for r in cost_loss_ratios:
        C, L = r, 1.0
        p_star = C / L

        act_fc = probs > p_star
        E_fc = _expected_expense(events, act_fc, C, L)

        # Climatology: act on all or none depending on base rate vs threshold.
        act_all = _expected_expense(events, np.ones_like(events, bool), C, L)
        act_none = _expected_expense(events, np.zeros_like(events, bool), C, L)
        E_clim = min(act_all, act_none)

        # Perfect foresight: act only when the event occurs.
        E_perfect = _expected_expense(events, events.astype(bool), C, L)

        denom = E_clim - E_perfect
        rev = (E_clim - E_fc) / denom if abs(denom) > 1e-12 else float("nan")
        results.append({
            "cost_loss_ratio": r,
            "p_star": p_star,
            "E_forecast": E_fc,
            "E_clim": E_clim,
            "E_perfect": E_perfect,
            "REV": rev,
        })
    return results, base_rate

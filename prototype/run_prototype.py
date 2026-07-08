"""End-to-end orchestrator for the South C hybrid flood-forecast prototype.

Pipeline
--------
1. Load + clean rainfall forcing (gauge + CHIRPS).
2. Generate synthetic SWMM-like flood "truth".
3. Run the GRC hydrologic core.
4. Engineer features and residual targets (lead 1..6 h).
5. Train the uncertainty-aware ML residual ensemble per lead time.
6. Produce calibrated probabilistic forecasts with variance decomposition.
7. Evaluate skill, calibration and cost-loss / REV decision value.
8. Run a full Monte-Carlo ensemble (~5000 draws) on a demonstration event.
9. Save metrics (JSON), predictions (CSV) and figures (PNG).

Run from the ``prototype`` directory:  python run_prototype.py
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.data import build_forcing
from src.synthetic import generate_truth
from src.grc import grc_frame
from src.features import build_features, build_targets
from src.residual_model import ResidualEnsemble
from src.ensemble import analytic_forecast, monte_carlo_forecast
from src import evaluation as ev


def _chrono_split(n: int, test_fraction: float):
    n_test = int(n * test_fraction)
    n_train = n - n_test
    return np.arange(n_train), np.arange(n_train, n)


def main() -> None:
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.SEED)

    print("[1/9] Loading rainfall forcing ...")
    forcing = build_forcing()
    print(f"      hours={len(forcing):,}  "
          f"range={forcing.index.min()} -> {forcing.index.max()}")

    print("[2/9] Generating synthetic flood truth ...")
    truth = generate_truth(forcing)
    print(f"      flood-event base rate = {truth['flood_event'].mean():.4f}")

    print("[3/9] Running GRC hydrologic core ...")
    grc = grc_frame(forcing)
    grc_only_skill = ev.deterministic_skill(
        truth["flood_mm"].to_numpy(), grc["grc_flood_mm"].to_numpy())
    print(f"      GRC-only RMSE = {grc_only_skill['RMSE']:.3f} mm")

    print("[4/9] Building features and residual targets ...")
    feats = build_features(forcing, grc)
    targets = build_targets(truth, grc)
    feat_cols = list(feats.columns)
    X_all = feats.to_numpy()

    thr = config.SYNTH["flood_threshold_mm"]
    train_idx, test_idx = _chrono_split(len(feats), config.TEST_FRACTION)
    print(f"      train={len(train_idx):,}  test={len(test_idx):,}")

    metrics = {
        "data": {
            "hours": int(len(forcing)),
            "start": str(forcing.index.min()),
            "end": str(forcing.index.max()),
            "flood_base_rate": float(truth["flood_event"].mean()),
            "grc_only_rmse_mm": grc_only_skill["RMSE"],
        },
        "leads": {},
    }

    per_lead_frames = {}
    print("[5/9] Training residual ensemble per lead time ...")
    for h in config.LEAD_HOURS:
        y = targets[h].to_numpy()
        valid = ~np.isnan(y)
        tr = train_idx[valid[train_idx]]
        te = test_idx[valid[test_idx]]

        model = ResidualEnsemble().fit(X_all[tr], y[tr])
        fc = analytic_forecast(grc["grc_flood_mm"].to_numpy()[te], X_all[te], model)

        # Truth aligned to this lead.
        y_true_flood = truth["flood_mm"].shift(-h).to_numpy()[te]
        events = truth["flood_event"].shift(-h).to_numpy()[te].astype(int)

        # Deterministic + probabilistic skill.
        det = ev.deterministic_skill(y_true_flood, fc["flood_pred_mu"].to_numpy())
        crps = ev.crps_gaussian(y_true_flood, fc["flood_pred_mu"].to_numpy(),
                                fc["sigma_total"].to_numpy())
        cov90 = ev.coverage(y_true_flood, fc["flood_pred_mu"].to_numpy(),
                            fc["sigma_total"].to_numpy(), 0.9)

        # Event probabilities + classification skill.
        probs = ev.exceedance_prob(fc["flood_pred_mu"].to_numpy(),
                                   fc["sigma_total"].to_numpy(), thr)
        cls = ev.classification_skill(events, probs)
        rev, base_rate = ev.relative_economic_value(
            events, probs, config.DECISION["cost_loss_ratios"])

        metrics["leads"][h] = {
            **det, "CRPS": crps, "coverage_90": cov90, **cls,
            "REV": rev,
        }
        per_lead_frames[h] = {
            "te": te, "fc": fc, "y_true": y_true_flood,
            "events": events, "probs": probs,
        }
        print(f"      lead {h}h: RMSE={det['RMSE']:.3f}  CRPS={crps:.3f}  "
              f"AUC={cls['roc_auc']:.3f}  cov90={cov90:.2f}")

    print("[6/9] Monte-Carlo ensemble on demonstration event ...")
    # Pick the highest-rainfall hour in the test set as the demo instance.
    lead_demo = 1
    demo = per_lead_frames[lead_demo]
    te = demo["te"]
    rain_test = forcing["rain_mm"].to_numpy()
    # Demonstration instance: the largest actual (synthetic) flood in the test set.
    demo_global_idx = te[int(np.argmax(demo["y_true"]))]
    window = rain_test[max(0, demo_global_idx - 47): demo_global_idx + 1]
    feat_row = X_all[demo_global_idx]
    model_demo = ResidualEnsemble().fit(
        X_all[train_idx[~np.isnan(targets[lead_demo].to_numpy())[train_idx]]],
        targets[lead_demo].to_numpy()[
            train_idx[~np.isnan(targets[lead_demo].to_numpy())[train_idx]]],
    )
    mc = monte_carlo_forecast(window, feat_row, model_demo)
    metrics["monte_carlo_demo"] = {
        "time": str(forcing.index[demo_global_idx]),
        "n_draws": int(mc["n_draws"]),
        "mu_ens": float(mc["mu_ens"]),
        "sigma_total": float(mc["sigma_total"]),
        "var_aleatory": float(mc["var_aleatory"]),
        "var_epistemic": float(mc["var_epistemic"]),
    }
    print(f"      draws={mc['n_draws']}  mu={mc['mu_ens']:.2f} mm  "
          f"sigma={mc['sigma_total']:.2f}  "
          f"(aleatory={mc['var_aleatory']:.2f}, epistemic={mc['var_epistemic']:.2f})")

    print("[7/9] Saving metrics + predictions ...")
    with open(config.OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Predictions CSV for lead 1h.
    d1 = per_lead_frames[1]
    pred_df = pd.DataFrame({
        "time": forcing.index[d1["te"]],
        "rain_mm": forcing["rain_mm"].to_numpy()[d1["te"]],
        "flood_true_mm": d1["y_true"],
        "flood_pred_mu_mm": d1["fc"]["flood_pred_mu"].to_numpy(),
        "sigma_total_mm": d1["fc"]["sigma_total"].to_numpy(),
        "flood_prob": d1["probs"],
        "flood_event_true": d1["events"],
    })
    pred_df.to_csv(config.OUT_DIR / "predictions_lead1h.csv", index=False)

    print("[8/9] Rendering figures ...")
    _make_figures(forcing, truth, grc, per_lead_frames, metrics, mc, thr)

    print("[9/9] Done. Outputs in:", config.OUT_DIR)
    print("\n=== Summary (lead 1h) ===")
    m1 = metrics["leads"][1]
    print(f" RMSE={m1['RMSE']:.3f} mm | CRPS={m1['CRPS']:.3f} | "
          f"AUC={m1['roc_auc']:.3f} | Brier={m1['brier']:.4f} | cov90={m1['coverage_90']:.2f}")
    best_rev = max(r["REV"] for r in m1["REV"] if np.isfinite(r["REV"]))
    print(f" Best REV vs climatology = {best_rev:.3f}")


def _make_figures(forcing, truth, grc, per_lead_frames, metrics, mc, thr):
    # 1) Time-series: rainfall + truth vs forecast with uncertainty band (event window).
    d1 = per_lead_frames[1]
    te = d1["te"]
    center = int(np.argmax(d1["y_true"]))
    lo = max(0, center - 72); hi = min(len(te), center + 72)
    sl = slice(lo, hi)
    t = forcing.index[te][sl]
    mu = d1["fc"]["flood_pred_mu"].to_numpy()[sl]
    sig = d1["fc"]["sigma_total"].to_numpy()[sl]

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                           gridspec_kw={"height_ratios": [1, 2]})
    ax[0].bar(t, forcing["rain_mm"].to_numpy()[te][sl], width=0.03,
              color="#3b7dd8", label="Rainfall")
    ax[0].set_ylabel("Rain (mm/h)")
    ax[0].legend(loc="upper right")
    ax[0].set_title("South C hybrid flood forecast (demonstration event, lead 1 h)")

    ax[1].plot(t, d1["y_true"][sl], "k-", lw=1.5, label="Truth (synthetic)")
    ax[1].plot(t, mu, color="#d1495b", lw=1.5, label="Forecast mean")
    ax[1].fill_between(t, np.clip(mu - 1.64 * sig, 0, None), mu + 1.64 * sig,
                       color="#d1495b", alpha=0.25, label="90% interval")
    ax[1].axhline(thr, color="gray", ls="--", lw=1, label="Flood threshold")
    ax[1].set_ylabel("Flood depth (mm)")
    ax[1].legend(loc="upper right")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "forecast_event.png", dpi=130)
    plt.close(fig)

    # 2) Reliability diagram (lead 1h).
    mp, of, cnt = ev.reliability_curve(d1["events"], d1["probs"], n_bins=10)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect")
    ax.plot(mp, of, "o-", color="#2e7d32", label="Forecast")
    ax.set_xlabel("Forecast probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram (flood event, lead 1 h)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "reliability.png", dpi=130)
    plt.close(fig)

    # 3) REV vs cost-loss ratio (all leads).
    fig, ax = plt.subplots(figsize=(7, 5))
    for h in config.LEAD_HOURS:
        rev = metrics["leads"][h]["REV"]
        xs = [r["cost_loss_ratio"] for r in rev]
        ys = [r["REV"] for r in rev]
        ax.plot(xs, ys, "o-", label=f"lead {h}h")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("Cost / loss ratio (C/L)")
    ax.set_ylabel("Relative economic value")
    ax.set_title("Decision value vs climatology")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "relative_economic_value.png", dpi=130)
    plt.close(fig)

    # 4) Uncertainty decomposition across leads.
    fig, ax = plt.subplots(figsize=(7, 5))
    al = [np.mean(per_lead_frames[h]["fc"]["var_aleatory"]) for h in config.LEAD_HOURS]
    ep = [np.mean(per_lead_frames[h]["fc"]["var_epistemic"]) for h in config.LEAD_HOURS]
    ax.bar(config.LEAD_HOURS, al, label="Aleatory", color="#1f77b4")
    ax.bar(config.LEAD_HOURS, ep, bottom=al, label="Epistemic", color="#ff7f0e")
    ax.set_xlabel("Lead time (h)")
    ax.set_ylabel("Mean predictive variance (mm$^2$)")
    ax.set_title("Predictive variance decomposition by lead time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "variance_decomposition.png", dpi=130)
    plt.close(fig)

    # 5) Monte-Carlo ensemble histogram for the demo event.
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(mc["draws"], bins=50, color="#6a51a3", alpha=0.85)
    ax.axvline(mc["mu_ens"], color="k", lw=2, label=f"Ensemble mean = {mc['mu_ens']:.2f} mm")
    ax.axvline(thr, color="red", ls="--", label="Flood threshold")
    ax.set_xlabel("Predicted flood depth (mm)")
    ax.set_ylabel("Ensemble draws")
    ax.set_title(f"Monte-Carlo predictive ensemble ({mc['n_draws']} draws)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIG_DIR / "monte_carlo_ensemble.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()

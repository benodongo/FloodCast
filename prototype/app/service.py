"""In-memory model service for the dashboard.

Trains the hybrid pipeline once at startup, precomputes full-series
probabilistic forecasts for every lead time, and exposes fast query methods for
the web API. Reuses the ``src`` prototype modules.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Make the prototype package importable when run from anywhere.
PROTO_DIR = Path(__file__).resolve().parents[1]
if str(PROTO_DIR) not in sys.path:
    sys.path.insert(0, str(PROTO_DIR))

from src import config  # noqa: E402
from src.data import build_forcing  # noqa: E402
from src.synthetic import generate_truth  # noqa: E402
from src.grc import grc_frame  # noqa: E402
from src.features import build_features, build_targets  # noqa: E402
from src.residual_model import ResidualEnsemble, project_physical  # noqa: E402
from src.ensemble import monte_carlo_forecast  # noqa: E402
from src import evaluation as ev  # noqa: E402
from src.chokepoints import CHOKEPOINTS  # noqa: E402

# Pretrained-model cache. Built once with ``python -m app.service`` and committed
# so the live server loads instantly instead of retraining on every boot.
ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_PATH = ARTIFACT_DIR / "service_state.joblib"

# Attributes that fully describe a trained service (everything the query methods
# read). Pickled together so a cached build restores identical behaviour.
_STATE_ATTRS = [
    "threshold", "forcing", "truth", "feats", "targets",
    "index", "n", "n_train", "train_idx", "test_idx",
    "models", "pred", "metrics", "X", "feat_names",
    "grc_only_rmse", "choke_depth", "choke_weight",
]


def _lib_versions() -> dict:
    """Signature of the libraries the pickle depends on for binary compat."""
    import sklearn
    return {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
    }


class FloodForecastService:
    """Loads data, trains models and answers dashboard queries."""

    def __init__(self) -> None:
        self.ready = False
        self.threshold = config.SYNTH["flood_threshold_mm"]

    # ------------------------------------------------------------------ #
    # Startup / training
    # ------------------------------------------------------------------ #
    def build(self, force: bool = False) -> None:
        if self.ready:
            return
        if not force and self._load_cache():
            self.ready = True
            print(f"Loaded pretrained models from {ARTIFACT_PATH.name}.", flush=True)
            return
        self.forcing = build_forcing()
        self.truth = generate_truth(self.forcing)
        self.grc = grc_frame(self.forcing)
        self.feats = build_features(self.forcing, self.grc)
        self.targets = build_targets(self.truth, self.grc)

        self.index = self.forcing.index
        self.n = len(self.forcing)
        n_test = int(self.n * config.TEST_FRACTION)
        self.n_train = self.n - n_test
        self.train_idx = np.arange(self.n_train)
        self.test_idx = np.arange(self.n_train, self.n)

        X = self.feats.to_numpy()
        grc_flood = self.grc["grc_flood_mm"].to_numpy()

        self.models: dict[int, ResidualEnsemble] = {}
        self.pred = {}          # lead -> dict of full-series arrays
        self.metrics = {"leads": {}}

        for h in config.LEAD_HOURS:
            y = self.targets[h].to_numpy()
            valid = ~np.isnan(y)
            tr = self.train_idx[valid[self.train_idx]]
            model = ResidualEnsemble().fit(X[tr], y[tr])
            self.models[h] = model

            out = model.predict(X)
            mu = project_physical(grc_flood, out["resid_mu"])
            sigma = np.sqrt(out["var_total"])
            prob = ev.exceedance_prob(mu, sigma, self.threshold)
            self.pred[h] = {
                "mu": mu, "sigma": sigma, "prob": prob,
                "var_aleatory": out["var_aleatory"],
                "var_epistemic": out["var_epistemic"],
            }

            # Test-set metrics.
            te = self.test_idx[valid[self.test_idx]]
            y_true = self.truth["flood_mm"].shift(-h).to_numpy()[te]
            events = self.truth["flood_event"].shift(-h).to_numpy()[te].astype(int)
            det = ev.deterministic_skill(y_true, mu[te])
            crps = ev.crps_gaussian(y_true, mu[te], sigma[te])
            cov = ev.coverage(y_true, mu[te], sigma[te], 0.9)
            cls = ev.classification_skill(events, prob[te])
            rev, base_rate = ev.relative_economic_value(
                events, prob[te], config.DECISION["cost_loss_ratios"])
            self.metrics["leads"][h] = {
                **det, "CRPS": crps, "coverage_90": cov, **cls, "REV": rev}

        self.grc_only_rmse = float(np.sqrt(np.mean(
            (self.truth["flood_mm"].to_numpy() - grc_flood) ** 2)))
        self.X = X
        self.feat_names = list(self.feats.columns)

        # Per-chokepoint depths + climatological susceptibility weights.
        n_sub = config.SYNTH["n_subcatchments"]
        cols = [f"choke_{j}" for j in range(n_sub)]
        self.choke_depth = self.truth[cols].to_numpy()
        mean_depth = self.choke_depth.mean(axis=0)
        self.choke_weight = mean_depth / (mean_depth.max() + 1e-9)
        self.ready = True
        try:
            self._save_cache()
        except Exception as exc:  # pragma: no cover - caching is best-effort
            print(f"Could not write model cache: {exc}", flush=True)

    # ------------------------------------------------------------------ #
    # Pretrained-model cache
    # ------------------------------------------------------------------ #
    def _save_cache(self) -> None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "versions": _lib_versions(),
            "state": {a: getattr(self, a) for a in _STATE_ATTRS},
        }
        joblib.dump(payload, ARTIFACT_PATH, compress=3)
        print(f"Saved model cache to {ARTIFACT_PATH}", flush=True)

    def _load_cache(self) -> bool:
        if not ARTIFACT_PATH.exists():
            return False
        try:
            payload = joblib.load(ARTIFACT_PATH)
        except Exception as exc:
            print(f"Ignoring model cache (load failed): {exc}", flush=True)
            return False
        if payload.get("versions") != _lib_versions():
            print("Ignoring model cache (library version mismatch); retraining.",
                  flush=True)
            return False
        for attr, value in payload["state"].items():
            setattr(self, attr, value)
        return True

    def load_cached(self) -> bool:
        """Synchronously restore a committed artifact without training.

        Returns True if the pretrained state was loaded and the service is ready.
        """
        if self.ready:
            return True
        if self._load_cache():
            self.ready = True
            return True
        return False

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def overview(self) -> dict:
        rain = self.forcing["rain_mm"]
        m1 = self.metrics["leads"][1]
        best_rev = max((r["REV"] for r in m1["REV"] if np.isfinite(r["REV"])),
                       default=float("nan"))
        return {
            "hours": int(self.n),
            "start": str(self.index.min()),
            "end": str(self.index.max()),
            "total_rain_mm": round(float(rain.sum()), 1),
            "wet_hours": int((rain > 0).sum()),
            "max_hourly_mm": round(float(rain.max()), 1),
            "flood_base_rate": round(float(self.truth["flood_event"].mean()), 4),
            "n_flood_events": int(self.truth["flood_event"].sum()),
            "grc_only_rmse": round(self.grc_only_rmse, 3),
            "hybrid_rmse_lead1": round(m1["RMSE"], 3),
            "auc_lead1": round(m1["roc_auc"], 3),
            "coverage_lead1": round(m1["coverage_90"], 3),
            "best_rev": round(best_rev, 3),
            "train_end": str(self.index[self.n_train - 1]),
        }

    def timeseries(self, start: str | None, end: str | None, lead: int) -> dict:
        lead = int(lead)
        idx = self.index
        mask = np.ones(self.n, dtype=bool)
        if start:
            mask &= idx >= pd.Timestamp(start)
        if end:
            mask &= idx <= pd.Timestamp(end)
        sel = np.where(mask)[0]
        # Cap points for responsiveness.
        if sel.size > 2000:
            step = int(np.ceil(sel.size / 2000))
            sel = sel[::step]

        p = self.pred[lead]
        truth = self.truth["flood_mm"].to_numpy()
        return {
            "lead": lead,
            "threshold": self.threshold,
            "time": [str(t) for t in idx[sel]],
            "rain": np.round(self.forcing["rain_mm"].to_numpy()[sel], 2).tolist(),
            "truth": np.round(truth[sel], 2).tolist(),
            "mu": np.round(p["mu"][sel], 2).tolist(),
            "lo": np.round(np.clip(p["mu"][sel] - 1.64 * p["sigma"][sel], 0, None), 2).tolist(),
            "hi": np.round(p["mu"][sel] + 1.64 * p["sigma"][sel], 2).tolist(),
            "prob": np.round(p["prob"][sel], 3).tolist(),
        }

    def events(self, top: int = 12) -> list[dict]:
        truth = self.truth["flood_mm"].to_numpy()
        order = np.argsort(truth)[::-1]
        picked, seen = [], []
        for i in order:
            if any(abs(int(i) - s) < 48 for s in seen):
                continue
            seen.append(int(i))
            picked.append({
                "time": str(self.index[i]),
                "flood_mm": round(float(truth[i]), 2),
                "rain_mm": round(float(self.forcing["rain_mm"].to_numpy()[i]), 2),
            })
            if len(picked) >= top:
                break
        return picked

    def metrics_table(self) -> dict:
        rows = []
        for h in config.LEAD_HOURS:
            m = self.metrics["leads"][h]
            rows.append({
                "lead": h,
                "RMSE": round(m["RMSE"], 3),
                "CRPS": round(m["CRPS"], 3),
                "AUC": round(m["roc_auc"], 3),
                "Brier": round(m["brier"], 4),
                "coverage_90": round(m["coverage_90"], 3),
            })
        # REV curves per lead.
        rev = {h: [{"cl": r["cost_loss_ratio"], "rev": round(r["REV"], 3)}
                   for r in self.metrics["leads"][h]["REV"]]
               for h in config.LEAD_HOURS}
        # Variance decomposition (mean over test set) per lead.
        var = []
        for h in config.LEAD_HOURS:
            te = self.test_idx
            var.append({
                "lead": h,
                "aleatory": round(float(np.mean(self.pred[h]["var_aleatory"][te])), 3),
                "epistemic": round(float(np.mean(self.pred[h]["var_epistemic"][te])), 3),
            })
        return {"rows": rows, "rev": rev, "variance": var,
                "cost_loss_ratios": config.DECISION["cost_loss_ratios"]}

    def monte_carlo(self, timestamp: str, lead: int = 1) -> dict:
        lead = int(lead)
        ts = pd.Timestamp(timestamp)
        pos = int(self.index.get_indexer([ts], method="nearest")[0])
        rain = self.forcing["rain_mm"].to_numpy()
        window = rain[max(0, pos - 47): pos + 1]
        feat_row = self.feats.to_numpy()[pos]
        mc = monte_carlo_forecast(window, feat_row, self.models[lead])
        draws = mc["draws"]
        # Histogram for the frontend.
        counts, edges = np.histogram(draws, bins=40)
        centers = 0.5 * (edges[:-1] + edges[1:])
        exceed = float(np.mean(draws >= self.threshold))
        return {
            "time": str(self.index[pos]),
            "lead": lead,
            "n_draws": int(mc["n_draws"]),
            "mu_ens": round(float(mc["mu_ens"]), 2),
            "sigma_total": round(float(mc["sigma_total"]), 2),
            "var_aleatory": round(float(mc["var_aleatory"]), 2),
            "var_epistemic": round(float(mc["var_epistemic"]), 2),
            "exceed_prob": round(exceed, 3),
            "threshold": self.threshold,
            "truth_mm": round(float(self.truth["flood_mm"].to_numpy()[pos]), 2),
            "hist_x": np.round(centers, 2).tolist(),
            "hist_y": counts.tolist(),
        }

    # ------------------------------------------------------------------ #
    # Analytics / decision support
    # ------------------------------------------------------------------ #
    @staticmethod
    def _alert(prob: float) -> dict:
        """Map a flood probability to an operational alert tier + actions."""
        if prob >= 0.60:
            return {"level": "Emergency", "rank": 3, "color": "#ff5d73",
                    "action": "Activate emergency response: close low-lying roads, "
                              "deploy pumps and barriers at chokepoints, alert residents."}
        if prob >= 0.30:
            return {"level": "Warning", "rank": 2, "color": "#ffb020",
                    "action": "Pre-position crews and barriers, clear drains at "
                              "chokepoints, issue a public advisory."}
        if prob >= 0.10:
            return {"level": "Watch", "rank": 1, "color": "#4f8cff",
                    "action": "Heightened monitoring; ready crews and equipment."}
        return {"level": "All clear", "rank": 0, "color": "#2fd98a",
                "action": "Routine monitoring only."}

    def risk_profile(self, timestamp: str) -> dict:
        """Probability + alert tier at each 0-6 h lead for a given moment."""
        ts = pd.Timestamp(timestamp)
        pos = int(self.index.get_indexer([ts], method="nearest")[0])
        leads = []
        max_prob = 0.0
        for h in config.LEAD_HOURS:
            p = float(self.pred[h]["prob"][pos])
            mu = float(self.pred[h]["mu"][pos])
            sig = float(self.pred[h]["sigma"][pos])
            max_prob = max(max_prob, p)
            leads.append({"lead": h, "prob": round(p, 3),
                          "mu": round(mu, 2), "sigma": round(sig, 2),
                          "alert": self._alert(p)})
        return {
            "time": str(self.index[pos]),
            "rain_now": round(float(self.forcing["rain_mm"].to_numpy()[pos]), 2),
            "headline": self._alert(max_prob),
            "max_prob": round(max_prob, 3),
            "leads": leads,
        }

    def contingency(self, lead: int, cost_loss: float) -> dict:
        """Verification + economics at decision threshold p* = C/L (test set)."""
        lead = int(lead)
        p_star = float(cost_loss)         # C/L, action when prob > p*
        te = self.test_idx
        y = self.targets[lead].to_numpy()
        valid = ~np.isnan(y)
        te = te[valid[te]]
        events = self.truth["flood_event"].shift(-lead).to_numpy()[te].astype(int)
        prob = self.pred[lead]["prob"][te]
        act = prob > p_star

        a = int(np.sum(act & (events == 1)))      # hits
        b = int(np.sum(act & (events == 0)))      # false alarms
        c = int(np.sum(~act & (events == 1)))     # misses
        d = int(np.sum(~act & (events == 0)))     # correct negatives
        pod = a / (a + c) if (a + c) else float("nan")     # detection rate
        far = b / (a + b) if (a + b) else float("nan")     # false-alarm ratio
        csi = a / (a + b + c) if (a + b + c) else float("nan")

        # Economics: L = 1, C = C/L. Expense = C if act else (L if event).
        C, L = p_star, 1.0
        exp_fc = float(np.mean(np.where(act, C, np.where(events == 1, L, 0.0))))
        exp_all = float(np.mean(np.where(True, C, 0.0)))
        exp_none = float(np.mean(np.where(events == 1, L, 0.0)))
        exp_clim = min(exp_all, exp_none)
        exp_perfect = float(np.mean(np.where(events == 1, C, 0.0)))
        denom = exp_clim - exp_perfect
        rev = (exp_clim - exp_fc) / denom if abs(denom) > 1e-12 else float("nan")
        saving = (exp_clim - exp_fc) / exp_clim * 100 if exp_clim > 1e-12 else 0.0

        return {
            "lead": lead, "p_star": round(p_star, 3),
            "hits": a, "false_alarms": b, "misses": c, "correct_neg": d,
            "POD": round(pod, 3), "FAR": round(far, 3), "CSI": round(csi, 3),
            "alerts_issued": a + b, "events_observed": a + c,
            "E_forecast": round(exp_fc, 4), "E_clim": round(exp_clim, 4),
            "E_perfect": round(exp_perfect, 4), "REV": round(rev, 3),
            "cost_saving_pct": round(saving, 1),
        }

    def drivers(self, lead: int = 1, sample: int = 1500) -> dict:
        """Permutation feature importance -> forecast explainability."""
        lead = int(lead)
        model = self.models[lead]
        rng = np.random.default_rng(config.SEED)
        te = self.test_idx
        y = self.targets[lead].to_numpy()
        valid = ~np.isnan(y)
        te = te[valid[te]]
        if te.size > sample:
            te = rng.choice(te, sample, replace=False)
        X = self.X[te].copy()
        y_true = y[te]

        base = model.predict(X)["resid_mu"]
        base_mse = float(np.mean((y_true - base) ** 2))
        imps = []
        for j, name in enumerate(self.feat_names):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            pred = model.predict(Xp)["resid_mu"]
            mse = float(np.mean((y_true - pred) ** 2))
            imps.append((name, max(0.0, mse - base_mse)))
        total = sum(v for _, v in imps) or 1.0
        imps.sort(key=lambda t: t[1], reverse=True)
        return {
            "lead": lead,
            "features": [n for n, _ in imps[:10]],
            "importance": [round(100 * v / total, 1) for _, v in imps[:10]],
        }

    # ------------------------------------------------------------------ #
    # Spatial risk map + alert bulletin
    # ------------------------------------------------------------------ #
    def map_risk(self, timestamp: str, lead: int = 1) -> dict:
        """Per-chokepoint flood risk for the map view."""
        lead = int(lead)
        ts = pd.Timestamp(timestamp)
        pos = int(self.index.get_indexer([ts], method="nearest")[0])
        agg_prob = float(self.pred[lead]["prob"][pos])
        agg_mu = float(self.pred[lead]["mu"][pos])

        points = []
        for cp in CHOKEPOINTS:
            j = cp["id"]
            w = float(self.choke_weight[j])
            prob = float(min(1.0, agg_prob * (0.5 + 0.5 * w) / 0.75))
            prob = max(0.0, min(1.0, prob))
            depth = round(agg_mu * w, 2)
            a = self._alert(prob)
            points.append({
                **cp, "prob": round(prob, 3), "depth_mm": depth,
                "level": a["level"], "color": a["color"], "rank": a["rank"],
            })
        points.sort(key=lambda p: p["prob"], reverse=True)
        return {
            "time": str(self.index[pos]), "lead": lead,
            "center": {"lat": -1.3155, "lon": 36.8325},
            "threshold": self.threshold, "points": points,
        }

    def bulletin(self, timestamp: str, lead: int = 1, cost_loss: float = 0.1) -> dict:
        """Auto-generated, printable flood alert bulletin."""
        rp = self.risk_profile(timestamp)
        mp = self.map_risk(timestamp, lead)
        ct = self.contingency(lead, cost_loss)
        issued = pd.Timestamp(rp["time"])
        valid_to = issued + pd.Timedelta(hours=6)
        head = rp["headline"]

        top = [p for p in mp["points"] if p["rank"] >= 1][:4]
        top_lines = [f"  - {p['name']}: {int(p['prob']*100)}% ({p['level']})"
                     for p in top] or ["  - No chokepoints above Watch level."]
        profile = " | ".join(
            f"+{L['lead']}h {int(L['prob']*100)}%" for L in rp["leads"])

        text = (
            "SOUTH C URBAN FLOOD ALERT BULLETIN\n"
            "FloodCast · Hybrid GRC + ML nowcasting system\n"
            "=================================================\n"
            f"ISSUED   : {issued:%Y-%m-%d %H:%M} (local)\n"
            f"VALID    : next 6 hours (to {valid_to:%H:%M})\n"
            f"AREA     : South C, Nairobi\n"
            f"STATUS   : {head['level'].upper()}  "
            f"(peak risk {int(rp['max_prob']*100)}%)\n"
            "-------------------------------------------------\n"
            f"CURRENT RAINFALL : {rp['rain_now']} mm/h\n"
            f"RISK BY LEAD TIME: {profile}\n"
            "-------------------------------------------------\n"
            "MOST EXPOSED CHOKEPOINTS:\n"
            + "\n".join(top_lines) + "\n"
            "-------------------------------------------------\n"
            "RECOMMENDED ACTION:\n"
            f"  {head['action']}\n"
            "-------------------------------------------------\n"
            "DECISION BASIS (hold-out verification):\n"
            f"  Detection rate {int(ct['POD']*100)}% · "
            f"false-alarm {int(ct['FAR']*100)}% · "
            f"cost saved vs climatology {ct['cost_saving_pct']}%\n"
            "-------------------------------------------------\n"
            "NOTE: Demonstration system on synthetic flood labels.\n"
            "Not an official warning.\n"
        )
        return {"time": rp["time"], "level": head["level"],
                "color": head["color"], "text": text}



SERVICE = FloodForecastService()


if __name__ == "__main__":
    # Build the pretrained artifact for committing: `python -m app.service`.
    print("Building and caching model state ...", flush=True)
    SERVICE.build(force=True)
    print("Done.", flush=True)

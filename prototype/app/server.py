"""Flask web server for the South C flood-forecast dashboard."""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, render_template, request

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR.parent) not in sys.path:
    sys.path.insert(0, str(APP_DIR.parent))

from app.service import SERVICE  # noqa: E402

app = Flask(__name__, template_folder=str(APP_DIR / "templates"),
            static_folder=str(APP_DIR / "static"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def api_health():
    return jsonify({
        "ready": SERVICE.ready,
        "phase": _build_status["phase"],
        "error": _build_status["error"],
    })


@app.route("/api/overview")
def api_overview():
    return jsonify(SERVICE.overview())


@app.route("/api/timeseries")
def api_timeseries():
    start = request.args.get("start")
    end = request.args.get("end")
    lead = int(request.args.get("lead", 1))
    return jsonify(SERVICE.timeseries(start, end, lead))


@app.route("/api/events")
def api_events():
    return jsonify(SERVICE.events(int(request.args.get("top", 12))))


@app.route("/api/metrics")
def api_metrics():
    return jsonify(SERVICE.metrics_table())


@app.route("/api/forecast")
def api_forecast():
    ts = request.args.get("time")
    lead = int(request.args.get("lead", 1))
    return jsonify(SERVICE.monte_carlo(ts, lead))


@app.route("/api/risk_profile")
def api_risk_profile():
    return jsonify(SERVICE.risk_profile(request.args.get("time")))


@app.route("/api/contingency")
def api_contingency():
    lead = int(request.args.get("lead", 1))
    cl = float(request.args.get("cl", 0.1))
    return jsonify(SERVICE.contingency(lead, cl))


@app.route("/api/drivers")
def api_drivers():
    lead = int(request.args.get("lead", 1))
    return jsonify(SERVICE.drivers(lead))


@app.route("/api/map")
def api_map():
    ts = request.args.get("time")
    lead = int(request.args.get("lead", 1))
    return jsonify(SERVICE.map_risk(ts, lead))


@app.route("/api/bulletin")
def api_bulletin():
    ts = request.args.get("time")
    lead = int(request.args.get("lead", 1))
    cl = float(request.args.get("cl", 0.1))
    return jsonify(SERVICE.bulletin(ts, lead, cl))

_build_lock = threading.Lock()
_build_started = False
_build_status = {"phase": "starting", "error": None}


@app.before_request
def _gate_until_ready():
    # Let the page, static assets and health check through while models train;
    # data endpoints answer 503 so the front-end can show a warming-up state.
    p = request.path
    if p.startswith("/api/") and p != "/api/health" and not SERVICE.ready:
        return jsonify({"status": "warming_up"}), 503


def _background_build():
    try:
        print("Preparing models (background startup) ...", flush=True)
        _build_status["phase"] = "building"
        SERVICE.build()
        _build_status["phase"] = "ready"
        print("Models ready.", flush=True)
    except Exception as exc:  # noqa: BLE001 - surface any startup failure
        import traceback
        _build_status["phase"] = "error"
        _build_status["error"] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        print(f"BUILD FAILED: {exc}", flush=True)


def _start_build():
    global _build_started
    with _build_lock:
        if _build_started:
            return
        _build_started = True
    threading.Thread(target=_background_build, daemon=True).start()


def _startup():
    # Fast path: load the committed pretrained artifact synchronously so the
    # worker is READY before it accepts any request. The gunicorn master has
    # already bound the port, so this short (~seconds) load does not delay
    # Render's port detection, and it avoids any thread-visibility race.
    try:
        if SERVICE.load_cached():
            _build_status["phase"] = "ready"
            print("Loaded pretrained models (startup).", flush=True)
            return
    except Exception as exc:  # noqa: BLE001 - fall back to training
        print(f"Cache load failed ({exc}); falling back to training.", flush=True)
    # Slow path (no/incompatible artifact): train in a background thread so the
    # port stays responsive while the front-end shows a warming-up state.
    _start_build()


# Prepare models on startup (synchronous load when the artifact is present,
# background training otherwise).
_startup()


def main():
    print("Starting server on http://0.0.0.0:5000 (models train in background)")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()

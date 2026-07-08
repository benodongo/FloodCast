"""Flask web server for the South C flood-forecast dashboard."""
from __future__ import annotations

import os
import sys
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


def _ensure_ready():
    if not SERVICE.ready:
        print("Training models (one-time startup, please wait) ...", flush=True)
        SERVICE.build()
        print("Models ready.", flush=True)


# Train on import so WSGI servers (e.g. gunicorn on Render) serve a ready app.
_ensure_ready()


def main():
    print("Ready. Open http://127.0.0.1:5000")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


if __name__ == "__main__":
    main()

/* South C FloodCast dashboard */
const $ = (s) => document.querySelector(s);
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v)) ? "–" : Number(v).toFixed(d);
const COL = { accent: "#4f8cff", accent2: "#7c5cff", good: "#2fd98a", warn: "#ffb020", danger: "#ff5d73", muted: "#8a94b4" };
Chart.defaults.font.family = "Inter, sans-serif";

/* ---------------- Theme helpers ---------------- */
function isDark() { return document.documentElement.classList.contains("dark"); }
function truthColor() { return isDark() ? "#e8ecf7" : "#334155"; }
function applyChartTheme() {
  Chart.defaults.color = isDark() ? "#8a94b4" : "#64748b";
  Chart.defaults.borderColor = isDark() ? "rgba(140,160,220,0.10)" : "rgba(100,116,139,0.16)";
}
applyChartTheme();

let charts = {};
let EVENTS = [];

async function api(path) { const r = await fetch(path); return r.json(); }

/* ---------------- KPIs ---------------- */
function kpi(label, value, unit = "", delta = "") {
  return `<div class="kpi"><div class="label">${label}</div>
    <div class="value">${value}${unit ? ` <small>${unit}</small>` : ""}</div>
    ${delta ? `<div class="delta good">${delta}</div>` : ""}</div>`;
}
async function loadOverview() {
  const o = await api("/api/overview");
  const reduction = (100 * (1 - o.hybrid_rmse_lead1 / o.grc_only_rmse)).toFixed(0);
  $("#kpis").innerHTML = [
    kpi("Record length", (o.hours / 8760).toFixed(1), "yrs", `${o.hours.toLocaleString()} hours`),
    kpi("Total rainfall", o.total_rain_mm.toLocaleString(), "mm", `${o.wet_hours.toLocaleString()} wet hours`),
    kpi("Peak intensity", o.max_hourly_mm, "mm/h"),
    kpi("Flood events", o.n_flood_events, "", `base rate ${(o.flood_base_rate * 100).toFixed(2)}%`),
    kpi("Hybrid RMSE", o.hybrid_rmse_lead1, "mm", `▼ ${reduction}% vs GRC-only`),
    kpi("Skill (AUC)", o.auc_lead1, "", "lead 1 h"),
    kpi("90% coverage", o.coverage_lead1, "", "well calibrated"),
    kpi("Best REV", o.best_rev, "", "vs climatology"),
  ].join("");
  $("#status").classList.add("ready");
  $("#status").innerHTML = `<span class="dot"></span> Models ready · train ≤ ${o.train_end.slice(0, 10)}`;
}

/* ---------------- Events ---------------- */
async function loadEvents() {
  EVENTS = await api("/api/events?top=12");
  $("#event-select").innerHTML = EVENTS.map((e, i) =>
    `<option value="${i}">${e.time.slice(0, 16)} · ${e.flood_mm} mm</option>`).join("");
}

/* ---------------- Time series ---------------- */
function windowFor(ts) {
  const t = new Date(ts.replace(" ", "T"));
  const start = new Date(t.getTime() - 3 * 864e5);
  const end = new Date(t.getTime() + 3 * 864e5);
  const iso = (d) => d.toISOString().slice(0, 19).replace("T", " ");
  return { start: iso(start), end: iso(end) };
}

async function loadTimeseries() {
  const ei = +$("#event-select").value || 0;
  const lead = +$("#lead-select").value;
  const ev = EVENTS[ei];
  const { start, end } = windowFor(ev.time);
  const d = await api(`/api/timeseries?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}&lead=${lead}`);
  const labels = d.time.map((t) => t.slice(5, 16));

  const thr = d.threshold;
  const thrLine = d.mu.map(() => thr);

  charts.ts?.destroy();
  charts.ts = new Chart($("#tsChart"), {
    data: {
      labels,
      datasets: [
        { type: "bar", label: "Rainfall (mm/h)", data: d.rain, yAxisID: "yR",
          backgroundColor: "rgba(79,140,255,0.45)", order: 5, barPercentage: 1, categoryPercentage: 1 },
        { type: "line", label: "90% lower", data: d.lo, yAxisID: "y",
          borderWidth: 0, pointRadius: 0, fill: false, order: 4 },
        { type: "line", label: "90% interval", data: d.hi, yAxisID: "y",
          borderWidth: 0, pointRadius: 0, backgroundColor: "rgba(255,93,115,0.18)", fill: "-1", order: 4 },
        { type: "line", label: "Forecast mean", data: d.mu, yAxisID: "y",
          borderColor: COL.danger, borderWidth: 2, pointRadius: 0, tension: 0.25, order: 2 },
        { type: "line", label: "Truth (synthetic)", data: d.truth, yAxisID: "y",
          borderColor: truthColor(), borderWidth: 1.6, pointRadius: 0, tension: 0.25, order: 1 },
        { type: "line", label: "Flood threshold", data: thrLine, yAxisID: "y",
          borderColor: COL.warn, borderDash: [6, 5], borderWidth: 1.2, pointRadius: 0, order: 3 },
      ],
    },
    options: {
      responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false },
      plugins: { legend: { labels: { boxWidth: 12, filter: (i) => i.text !== "90% lower" } } },
      scales: {
        y: { title: { display: true, text: "Flood depth (mm)" }, beginAtZero: true },
        yR: { position: "right", reverse: true, title: { display: true, text: "Rain (mm/h)" },
              grid: { drawOnChartArea: false }, beginAtZero: true },
        x: { ticks: { maxTicksLimit: 12 } },
      },
    },
  });

  charts.prob?.destroy();
  charts.prob = new Chart($("#probChart"), {
    type: "line",
    data: { labels, datasets: [{ label: "Flood probability", data: d.prob,
      borderColor: COL.accent2, backgroundColor: "rgba(124,92,255,0.25)", fill: true,
      borderWidth: 1.5, pointRadius: 0, tension: 0.25 }] },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { y: { min: 0, max: 1, title: { display: true, text: "P(flood)" } }, x: { ticks: { maxTicksLimit: 12 } } } },
  });
}

/* ---------------- Monte-Carlo ensemble ---------------- */
async function runEnsemble() {
  const ei = +$("#event-select").value || 0;
  const lead = +$("#lead-select").value;
  const ev = EVENTS[ei];
  $("#mc-stats").textContent = "Running ensemble…";
  const d = await api(`/api/forecast?time=${encodeURIComponent(ev.time)}&lead=${lead}`);

  $("#mc-stats").innerHTML = `
    <div class="s"><b>${d.n_draws.toLocaleString()}</b><span>predictive draws</span></div>
    <div class="s"><b>${fmt(d.mu_ens)} mm</b><span>ensemble mean</span></div>
    <div class="s"><b>±${fmt(d.sigma_total)}</b><span>total σ</span></div>
    <div class="s"><b>${fmt(d.truth_mm)} mm</b><span>synthetic truth</span></div>
    <div class="s"><b><span class="badge ${d.exceed_prob > 0.5 ? "hi" : "lo"}">${(d.exceed_prob * 100).toFixed(0)}%</span></b><span>P(depth > ${d.threshold} mm)</span></div>`;

  charts.mc?.destroy();
  charts.mc = new Chart($("#mcChart"), {
    type: "bar",
    data: { labels: d.hist_x, datasets: [{ label: "Ensemble draws", data: d.hist_y,
      backgroundColor: "rgba(124,92,255,0.55)", barPercentage: 1, categoryPercentage: 1 }] },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false },
        tooltip: { callbacks: { title: (i) => `${i[0].label} mm` } } },
      scales: { x: { title: { display: true, text: "Predicted flood depth (mm)" }, ticks: { maxTicksLimit: 12 } },
                y: { title: { display: true, text: "Draws" } } } },
  });
}

/* ---------------- Analytics / Decision support ---------------- */
async function loadAnalytics() {
  const ei = +$("#event-select").value || 0;
  const lead = +$("#lead-select").value;
  const cl = +$("#asset-select").value;
  const ev = EVENTS[ei];

  const rp = await api(`/api/risk_profile?time=${encodeURIComponent(ev.time)}`);
  const h = rp.headline;
  $("#alert-banner").style.background = `linear-gradient(90deg, ${h.color}22, transparent)`;
  $("#alert-banner").style.borderColor = h.color;
  $("#alert-banner").innerHTML = `
    <div class="tier" style="background:${h.color}">${h.level}</div>
    <div class="msg"><b>Peak risk ${(rp.max_prob * 100).toFixed(0)}%</b> over next 6 h ·
      rain now ${rp.rain_now} mm/h<br><small>${h.action}</small></div>`;

  $("#risk-profile").innerHTML = rp.leads.map((L) => `
    <div class="risk-cell">
      <div class="lead">+${L.lead} h</div>
      <div class="pct" style="color:${L.alert.color}">${(L.prob * 100).toFixed(0)}%</div>
      <div class="tier" style="color:${L.alert.color}">${L.alert.level}</div>
      <div class="bar"><i style="width:${Math.min(100, L.prob * 100)}%;background:${L.alert.color}"></i></div>
    </div>`).join("");

  const ct = await api(`/api/contingency?lead=${lead}&cl=${cl}`);
  const decClass = ct.cost_saving_pct > 0 ? "lo" : "hi";
  $("#decision-cards").innerHTML = `
    <div class="s"><b>${(ct.p_star * 100).toFixed(0)}%</b><span>act-if probability &gt; p*</span></div>
    <div class="s"><b>${fmt(ct.POD, 2)}</b><span>detection rate (POD)</span></div>
    <div class="s"><b>${fmt(ct.FAR, 2)}</b><span>false-alarm ratio</span></div>
    <div class="s"><b>${fmt(ct.CSI, 2)}</b><span>critical success index</span></div>
    <div class="s"><b><span class="badge ${decClass}">${ct.cost_saving_pct}%</span></b><span>cost saved vs climatology</span></div>`;

  $("#contingency-table").innerHTML = `
    <thead><tr><th></th><th>Flood observed</th><th>No flood</th></tr></thead>
    <tbody>
      <tr><td>Alert issued</td><td>${ct.hits} <small>hits</small></td><td>${ct.false_alarms} <small>false</small></td></tr>
      <tr><td>No alert</td><td>${ct.misses} <small>missed</small></td><td>${ct.correct_neg.toLocaleString()} <small>correct</small></td></tr>
    </tbody>`;

  const dr = await api(`/api/drivers?lead=${lead}`);
  charts.drivers?.destroy();
  charts.drivers = new Chart($("#driversChart"), {
    type: "bar",
    data: { labels: dr.features, datasets: [{ label: "Relative importance (%)", data: dr.importance,
      backgroundColor: "rgba(47,217,138,0.6)" }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: { x: { title: { display: true, text: "Contribution to forecast (%)" } } } },
  });
}

/* ---------------- Metrics + REV + variance ---------------- */
async function loadMetrics() {
  const m = await api("/api/metrics");
  $("#metrics-table").innerHTML = `
    <thead><tr><th>Lead</th><th>RMSE</th><th>CRPS</th><th>AUC</th><th>Brier</th><th>Cov 90%</th></tr></thead>
    <tbody>${m.rows.map((r) => `<tr><td>${r.lead} h</td><td>${fmt(r.RMSE, 3)}</td><td>${fmt(r.CRPS, 3)}</td>
      <td>${fmt(r.AUC, 3)}</td><td>${fmt(r.Brier, 4)}</td><td>${fmt(r.coverage_90, 2)}</td></tr>`).join("")}</tbody>`;

  // REV chart
  const leads = Object.keys(m.rev);
  const palette = ["#4f8cff", "#7c5cff", "#2fd98a", "#ffb020", "#ff5d73", "#38bdf8"];
  charts.rev?.destroy();
  charts.rev = new Chart($("#revChart"), {
    type: "line",
    data: {
      labels: m.cost_loss_ratios,
      datasets: leads.map((h, i) => ({
        label: `lead ${h} h`, data: m.rev[h].map((p) => p.rev),
        borderColor: palette[i % palette.length], backgroundColor: palette[i % palette.length],
        borderWidth: 2, pointRadius: 3, tension: 0.25,
      })),
    },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { boxWidth: 12 } } },
      scales: { x: { title: { display: true, text: "Cost / loss ratio" } },
                y: { title: { display: true, text: "Relative economic value" }, suggestedMin: 0, suggestedMax: 1 } } },
  });

  // Variance decomposition stacked bar
  charts.var?.destroy();
  charts.var = new Chart($("#varChart"), {
    type: "bar",
    data: {
      labels: m.variance.map((v) => `${v.lead} h`),
      datasets: [
        { label: "Aleatory", data: m.variance.map((v) => v.aleatory), backgroundColor: COL.accent },
        { label: "Epistemic", data: m.variance.map((v) => v.epistemic), backgroundColor: COL.warn },
      ],
    },
    options: { responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { boxWidth: 12 } } },
      scales: { x: { stacked: true, title: { display: true, text: "Lead time" } },
                y: { stacked: true, title: { display: true, text: "Mean variance (mm²)" } } } },
  });
}

/* ---------------- Risk map + bulletin ---------------- */
let leafletMap = null, markerLayer = null, tileLayer = null, currentBulletin = "";

function setMapTiles() {
  if (!leafletMap) return;
  if (tileLayer) leafletMap.removeLayer(tileLayer);
  const url = isDark()
    ? "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
    : "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png";
  tileLayer = L.tileLayer(url, { maxZoom: 19 }).addTo(leafletMap);
  tileLayer.bringToBack();
}

async function loadMap() {
  const ei = +$("#event-select").value || 0;
  const lead = +$("#lead-select").value;
  const ev = EVENTS[ei];
  const d = await api(`/api/map?time=${encodeURIComponent(ev.time)}&lead=${lead}`);

  if (!leafletMap) {
    leafletMap = L.map("map", { zoomControl: true, attributionControl: false })
      .setView([d.center.lat, d.center.lon], 14);
    setMapTiles();
    markerLayer = L.layerGroup().addTo(leafletMap);
  }
  markerLayer.clearLayers();
  d.points.forEach((p) => {
    L.circleMarker([p.lat, p.lon], {
      radius: 8 + p.prob * 16, color: p.color, weight: 2,
      fillColor: p.color, fillOpacity: 0.55,
    }).bindPopup(
      `<b>${p.name}</b><br>${p.note}<br>` +
      `Risk: <b style="color:${p.color}">${(p.prob * 100).toFixed(0)}% · ${p.level}</b><br>` +
      `Est. depth: ${p.depth_mm} mm`
    ).addTo(markerLayer);
  });
  setTimeout(() => leafletMap.invalidateSize(), 100);

  const tiers = [["Emergency", "#ff5d73"], ["Warning", "#ffb020"], ["Watch", "#4f8cff"], ["All clear", "#2fd98a"]];
  $("#map-legend").innerHTML = tiers.map(([n, c]) =>
    `<div class="lg"><span class="dot2" style="background:${c}"></span>${n}</div>`).join("");
}

async function loadBulletin() {
  const ei = +$("#event-select").value || 0;
  const lead = +$("#lead-select").value;
  const cl = +$("#asset-select").value;
  const ev = EVENTS[ei];
  const d = await api(`/api/bulletin?time=${encodeURIComponent(ev.time)}&lead=${lead}&cl=${cl}`);
  currentBulletin = d.text;
  const el = $("#bulletin-text");
  el.textContent = d.text;
  el.style.borderColor = d.color;
}

/* ---------------- Nav ---------------- */
const VIEW_MAP = { dashboard: "#kpis", forecast: "#panel-forecast", ensemble: "#panel-ensemble",
  analytics: "#panel-analytics", map: "#panel-map-wrap", performance: "#panel-metrics", decision: "#panel-rev" };
const TITLES = { dashboard: ["Dashboard", "Rainfall-driven city flood-risk forecast · hybrid GRC + machine learning"],
  forecast: ["Live Forecast", "Probabilistic 0–6 h flood forecast with calibrated uncertainty"],
  ensemble: ["Ensemble & Uncertainty", "Monte-Carlo predictive ensemble and variance decomposition"],
  analytics: ["Decision Support", "Turning forecasts into alerts and cost-effective operational action"],
  map: ["Risk Map & Bulletin", "Per-chokepoint spatial risk and auto-generated operator bulletin"],
  performance: ["Model Performance", "Skill and calibration across lead times"],
  decision: ["Decision Value", "Cost–loss and relative economic value analysis"] };
document.querySelectorAll(".nav-item").forEach((el) => {
  el.addEventListener("click", () => {
    document.querySelectorAll(".nav-item").forEach((n) => n.classList.remove("active"));
    el.classList.add("active");
    const v = el.dataset.view;
    $("#view-title").textContent = TITLES[v][0];
    $("#view-sub").textContent = TITLES[v][1];
    document.querySelector(VIEW_MAP[v])?.scrollIntoView({ behavior: "smooth", block: "start" });
    if (v === "map" && leafletMap) setTimeout(() => leafletMap.invalidateSize(), 150);
    closeSidebar();
  });
});

/* ---------------- Mobile sidebar ---------------- */
function openSidebar() {
  const sb = $("#sidebar"), bd = $("#sidebar-backdrop"), tg = $("#menu-toggle");
  sb?.classList.remove("-translate-x-full");
  bd?.classList.remove("opacity-0", "pointer-events-none");
  tg?.setAttribute("aria-expanded", "true");
}
function closeSidebar() {
  const sb = $("#sidebar"), bd = $("#sidebar-backdrop"), tg = $("#menu-toggle");
  // Only collapse on small screens (drawer mode)
  if (window.matchMedia("(min-width: 1024px)").matches) return;
  sb?.classList.add("-translate-x-full");
  bd?.classList.add("opacity-0", "pointer-events-none");
  tg?.setAttribute("aria-expanded", "false");
}
function toggleSidebar() {
  const sb = $("#sidebar");
  if (sb?.classList.contains("-translate-x-full")) openSidebar(); else closeSidebar();
}
$("#menu-toggle")?.addEventListener("click", toggleSidebar);
$("#sidebar-backdrop")?.addEventListener("click", closeSidebar);
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeSidebar(); });
window.addEventListener("resize", () => {
  // Reset drawer state when switching to desktop layout
  if (window.matchMedia("(min-width: 1024px)").matches) {
    $("#sidebar-backdrop")?.classList.add("opacity-0", "pointer-events-none");
  }
});

/* ---------------- Theme toggle ---------------- */
function setThemeUI() {
  const d = isDark();
  const icon = $("#theme-icon"), label = $("#theme-label");
  if (icon) icon.textContent = d ? "☀️" : "🌙";
  if (label) label.textContent = d ? "Light" : "Dark";
}
async function toggleTheme() {
  const d = !isDark();
  document.documentElement.classList.toggle("dark", d);
  try { localStorage.setItem("fc-theme", d ? "dark" : "light"); } catch (e) {}
  setThemeUI();
  applyChartTheme();
  setMapTiles();
  await loadTimeseries();
  await loadMetrics();
  await loadAnalytics();
  if (charts.mc) await runEnsemble();
  await loadMap();
  await loadBulletin();
}

/* ---------------- Init ---------------- */
async function waitReady() {
  for (let i = 0; ; i++) {
    let phase = null, error = null;
    try {
      const h = await api("/api/health");
      if (h && h.ready) return;
      phase = h && h.phase;
      error = h && h.error;
    } catch (e) { /* server still starting */ }
    if (phase === "error") {
      $("#status").innerHTML = `<span class="dot"></span> Startup failed — ${error || "see server logs"}`;
    } else {
      const dots = ".".repeat((i % 3) + 1);
      $("#status").innerHTML = `<span class="dot"></span> Warming up — loading models${dots}`;
    }
    await new Promise((r) => setTimeout(r, 2500));
  }
}

function refreshEventViews() { loadTimeseries(); loadAnalytics(); loadMap(); loadBulletin(); }
$("#event-select").addEventListener("change", refreshEventViews);
$("#lead-select").addEventListener("change", refreshEventViews);
$("#asset-select").addEventListener("change", () => { loadAnalytics(); loadBulletin(); });
$("#run-ensemble").addEventListener("click", runEnsemble);
$("#theme-toggle").addEventListener("click", toggleTheme);
setThemeUI();

$("#dl-bulletin").addEventListener("click", () => {
  const blob = new Blob([currentBulletin], { type: "text/plain" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "south_c_flood_bulletin.txt";
  a.click();
});
$("#copy-bulletin").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(currentBulletin);
    $("#copy-bulletin").textContent = "Copied";
    setTimeout(() => ($("#copy-bulletin").textContent = "Copy"), 1500);
  } catch (e) { /* ignore */ }
});

(async function init() {
  await waitReady();
  await loadOverview();
  await loadEvents();
  await loadTimeseries();
  await loadAnalytics();
  await loadMap();
  await loadBulletin();
  await loadMetrics();
})();

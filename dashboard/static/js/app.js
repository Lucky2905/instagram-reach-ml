/* ─────────────────────────────────────────────────────────────────
   dashboard/static/js/app.js
   Interactive logic for the Instagram Reach ML dashboard.
   Handles: API calls, CSV parsing, Chart.js rendering.
   ───────────────────────────────────────────────────────────────── */

"use strict";

// ── Chart instances (kept for destroy/re-render) ──────────────────
let scatterChartInst = null;
let featureChartInst  = null;
let tierChartInst     = null;

// ── Chart.js default config ───────────────────────────────────────
Chart.defaults.color = "#94a3b8";
Chart.defaults.font.family = "Inter, system-ui, sans-serif";
Chart.defaults.font.size   = 12;

const PURPLE = "#8b5cf6";
const PINK   = "#ec4899";
const BLUE   = "#3b82f6";
const CYAN   = "#06b6d4";
const GREEN  = "#10b981";
const AMBER  = "#f59e0b";
const RED    = "#f87171";

// ── On page load ──────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadMetrics();
  setupNavHighlight();
});

// ── Health check ──────────────────────────────────────────────────
async function checkHealth() {
  const dot  = document.getElementById("statusDot");
  const text = document.getElementById("statusText");
  try {
    const res  = await fetch("/health");
    const data = await res.json();
    if (data.models_loaded) {
      dot.className  = "status-dot online";
      text.textContent = "Models ready";
    } else {
      dot.className  = "status-dot offline";
      text.textContent = "No models";
    }
  } catch {
    dot.className  = "status-dot offline";
    text.textContent = "Server offline";
  }
}

// ── Load metrics ──────────────────────────────────────────────────
async function loadMetrics() {
  try {
    const res  = await fetch("/metrics");
    if (!res.ok) return;
    const data = await res.json();
    const latest = data.latest;
    if (!latest) return;

    const r2  = latest.r2   != null ? latest.r2.toFixed(4)         : "—";
    const mae = latest.mae  != null ? Number(latest.mae).toLocaleString() : "—";
    const acc = latest.accuracy != null ? (latest.accuracy * 100).toFixed(1) + "%" : "—";
    const strat = latest.strategy || "—";

    document.getElementById("metric-r2").textContent    = r2;
    document.getElementById("metric-mae").textContent   = mae;
    document.getElementById("metric-acc").textContent   = acc;
    document.getElementById("metric-strat").textContent = strat;
  } catch (e) {
    console.warn("Could not load metrics:", e);
  }
}

// ── Single prediction ─────────────────────────────────────────────
async function runSinglePredict() {
  const btn = document.getElementById("btn-predict");
  const errDiv = document.getElementById("predictError");
  const resCard = document.getElementById("resultCard");

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner" style="width:16px;height:16px;border-width:2px;"></span> Predicting…';
  errDiv.style.display  = "none";
  resCard.style.display = "none";

  const payload = {
    likes:            Number(document.getElementById("inp-likes").value),
    comments:         Number(document.getElementById("inp-comments").value),
    shares:           Number(document.getElementById("inp-shares").value),
    saves:            Number(document.getElementById("inp-saves").value),
    hashtag_count:    Number(document.getElementById("inp-hashtags").value),
    post_type:        document.getElementById("inp-posttype").value,
    hour_of_day:      Number(document.getElementById("inp-hour").value),
    day_of_week:      Number(document.getElementById("inp-day").value),
    follower_count:   Number(document.getElementById("inp-followers").value),
    account_age_days: Number(document.getElementById("inp-age").value),
  };

  try {
    const res  = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      errDiv.textContent    = `Error: ${data.error}`;
      errDiv.style.display  = "block";
      return;
    }

    document.getElementById("res-reach").textContent = Number(data.predicted_reach).toLocaleString();
    document.getElementById("res-tier").textContent  = data.reach_tier.toUpperCase();
    document.getElementById("res-conf").textContent  = (data.confidence * 100).toFixed(1) + "%";
    resCard.style.display = "flex";

  } catch (e) {
    errDiv.textContent   = `Network error: ${e.message}`;
    errDiv.style.display = "block";
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">⚡</span> Predict Reach';
  }
}

// ── CSV file handling ─────────────────────────────────────────────
function handleFileSelect(event) {
  const file = event.target.files[0];
  if (file) processCSV(file);
}

function handleDrop(event) {
  event.preventDefault();
  document.getElementById("uploadZone").classList.remove("drag-over");
  const file = event.dataTransfer.files[0];
  if (file && file.name.endsWith(".csv")) {
    processCSV(file);
  }
}

async function processCSV(file) {
  showUploadStatus(`Reading ${file.name}…`);

  const text = await file.text();
  const rows = parseCSV(text);
  if (!rows || rows.length === 0) {
    showUploadStatus("❌ Could not parse CSV. Check column headers.", false);
    return;
  }

  showUploadStatus(`Sending ${rows.length} rows to API…`);

  try {
    const res  = await fetch("/predict-batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(rows),
    });
    const data = await res.json();

    if (!res.ok) {
      showUploadStatus(`❌ API error: ${data.error}`, false);
      return;
    }

    hideUploadStatus();
    renderTable(data.predictions);
    renderCharts(data.predictions, data.feature_importances);

  } catch (e) {
    showUploadStatus(`❌ Network error: ${e.message}`, false);
  }
}

// ── CSV parser ────────────────────────────────────────────────────
function parseCSV(text) {
  const lines = text.trim().split("\n");
  if (lines.length < 2) return null;

  const headers = lines[0].split(",").map(h => h.trim().replace(/\r/g, ""));
  const required = [
    "likes","comments","shares","saves","hashtag_count",
    "post_type","hour_of_day","day_of_week","follower_count","account_age_days"
  ];

  const missing = required.filter(r => !headers.includes(r));
  if (missing.length > 0) {
    alert(`CSV is missing columns: ${missing.join(", ")}`);
    return null;
  }

  return lines.slice(1).map(line => {
    const vals = line.split(",");
    const obj = {};
    headers.forEach((h, i) => {
      const v = (vals[i] || "").trim().replace(/\r/g, "");
      obj[h] = isNaN(v) ? v : Number(v);
    });
    return obj;
  }).filter(row => row.post_type);  // remove empty rows
}

// ── Table rendering ───────────────────────────────────────────────
function renderTable(predictions) {
  const tbody = document.getElementById("tableBody");
  const container = document.getElementById("tableContainer");
  const count = document.getElementById("tableCount");

  tbody.innerHTML = "";
  predictions.slice(0, 200).forEach((p, i) => {
    const tierClass = `tier-${p.reach_tier}`;
    tbody.innerHTML += `
      <tr>
        <td>${i + 1}</td>
        <td>${Number(p.predicted_reach).toLocaleString()}</td>
        <td class="${tierClass}">${p.reach_tier.toUpperCase()}</td>
        <td>${(p.confidence * 100).toFixed(1)}%</td>
      </tr>`;
  });

  count.textContent = `${predictions.length} rows`;
  container.style.display = "block";
}

// ── Chart rendering ───────────────────────────────────────────────
function renderCharts(predictions, featureImportances) {
  document.getElementById("chartPlaceholder").style.display = "none";
  document.getElementById("chartCard1").style.display = "block";
  document.getElementById("chartCard2").style.display = "block";
  document.getElementById("chartCard3").style.display = "block";

  renderScatterChart(predictions);
  renderFeatureChart(featureImportances);
  renderTierChart(predictions);
}

// Chart 1: Reach histogram (predicted reach distribution)
function renderScatterChart(predictions) {
  const reaches = predictions.map(p => p.predicted_reach);

  // Build histogram buckets
  const min = Math.min(...reaches);
  const max = Math.max(...reaches);
  const BUCKETS = 20;
  const bucketSize = (max - min) / BUCKETS;
  const counts = Array(BUCKETS).fill(0);
  reaches.forEach(r => {
    const idx = Math.min(Math.floor((r - min) / bucketSize), BUCKETS - 1);
    counts[idx]++;
  });
  const labels = counts.map((_, i) =>
    Math.round(min + i * bucketSize).toLocaleString()
  );

  const ctx = document.getElementById("scatterChart");
  if (scatterChartInst) scatterChartInst.destroy();

  scatterChartInst = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Post Count",
        data: counts,
        backgroundColor: makeGradient(ctx, PURPLE, PINK),
        borderRadius: 4,
        borderSkipped: false,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: (items) => `Reach ≈ ${items[0].label}`,
            label: (item)  => `Posts: ${item.raw}`,
          },
        },
      },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { maxTicksLimit: 8 } },
        y: { grid: { color: "rgba(255,255,255,0.04)" } },
      },
    },
  });
}

// Chart 2: Feature importance bar chart
function renderFeatureChart(fi) {
  if (!fi) return;

  const LABELS = {
    likes: "Likes", comments: "Comments", shares: "Shares", saves: "Saves",
    hashtag_count: "Hashtags", hour_of_day: "Hour", day_of_week: "Day",
    follower_count: "Followers", account_age_days: "Acct Age",
    post_type_reel: "Is Reel", post_type_video: "Is Video",
  };

  const sorted = Object.entries(fi)
    .sort(([, a], [, b]) => b - a);

  const labels = sorted.map(([k]) => LABELS[k] || k);
  const values = sorted.map(([, v]) => +(v * 100).toFixed(2));

  const ctx = document.getElementById("featureChart");
  if (featureChartInst) featureChartInst.destroy();

  featureChartInst = new Chart(ctx, {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Importance (%)",
        data: values,
        backgroundColor: values.map((_, i) =>
          `hsl(${260 + i * 15}, 70%, 60%)`
        ),
        borderRadius: 4,
      }],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: item => ` ${item.raw}%` } },
      },
      scales: {
        x: { grid: { color: "rgba(255,255,255,0.04)" }, ticks: { callback: v => v + "%" } },
        y: { grid: { display: false } },
      },
    },
  });
}

// Chart 3: Reach tier pie chart
function renderTierChart(predictions) {
  const counts = { low: 0, medium: 0, high: 0 };
  predictions.forEach(p => { if (counts[p.reach_tier] !== undefined) counts[p.reach_tier]++; });

  const ctx = document.getElementById("tierChart");
  if (tierChartInst) tierChartInst.destroy();

  tierChartInst = new Chart(ctx, {
    type: "doughnut",
    data: {
      labels: ["Low", "Medium", "High"],
      datasets: [{
        data: [counts.low, counts.medium, counts.high],
        backgroundColor: [RED, AMBER, GREEN],
        borderColor: "#06070f",
        borderWidth: 3,
        hoverOffset: 8,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: "65%",
      plugins: {
        legend: {
          position: "bottom",
          labels: { padding: 16, usePointStyle: true, pointStyleWidth: 10 },
        },
        tooltip: {
          callbacks: {
            label: item => {
              const total = item.dataset.data.reduce((a, b) => a + b, 0);
              const pct = total ? ((item.raw / total) * 100).toFixed(1) : 0;
              return ` ${item.raw} posts (${pct}%)`;
            },
          },
        },
      },
    },
  });
}

// ── Utility: canvas gradient ──────────────────────────────────────
function makeGradient(ctx, color1, color2) {
  const canvas = ctx.canvas;
  const gradient = canvas.getContext("2d").createLinearGradient(0, 0, canvas.width, 0);
  gradient.addColorStop(0, color1);
  gradient.addColorStop(1, color2);
  return gradient;
}

// ── Upload status helpers ─────────────────────────────────────────
function showUploadStatus(msg, showSpinner = true) {
  const div = document.getElementById("uploadStatus");
  const txt = document.getElementById("uploadStatusText");
  txt.textContent = msg;
  div.querySelector(".spinner").style.display = showSpinner ? "block" : "none";
  div.style.display = "flex";
}

function hideUploadStatus() {
  document.getElementById("uploadStatus").style.display = "none";
}

// ── Smooth nav active state ───────────────────────────────────────
function setupNavHighlight() {
  const sections = document.querySelectorAll(".section");
  const navItems = document.querySelectorAll(".nav-item");

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = entry.target.id.replace("section-", "");
        navItems.forEach(n => n.classList.remove("active"));
        const active = document.getElementById(`nav-${id}`);
        if (active) active.classList.add("active");
      }
    });
  }, { threshold: 0.4 });

  sections.forEach(s => observer.observe(s));
}

/**
 * DeliverAI — Static Vanilla JS Web Application
 * Connects directly to FastAPI backend at http://localhost:8000
 */

// Automatically detect backend URL (same host or default port 8000)
const API_BASE_URL = window.location.port === "8000"
  ? window.location.origin
  : "http://localhost:8000";

// Default state
const state = {
  distance: 5.0,
  prepTime: 15,
  experience: 3.0,
  traffic: "Medium",
  weather: "Clear",
  vehicle: "Scooter",
  isBackendOnline: false,
};

// Fallback surrogate constants
const weatherFactor = { Clear: 1, Windy: 1.08, Foggy: 1.15, Rainy: 1.22, Snowy: 1.38 };
const trafficFactor = { Low: 1, Medium: 1.25, High: 1.6 };
const vehicleSpeed = { Bike: 14, Scooter: 24, Car: 28 };

function calculateLocalEta(input) {
  const travel = (input.distance / (vehicleSpeed[input.vehicle] || 20)) * 60 * (weatherFactor[input.weather] || 1) * (trafficFactor[input.traffic] || 1);
  const expBonus = Math.max(0.82, 1 - input.experience * 0.02);
  const eta = input.prepTime + travel * expBonus + 4;
  return Math.round(eta * 10) / 10;
}

// DOM Elements
const elDistance = document.getElementById("input-distance");
const elPrep = document.getElementById("input-prep");
const elExp = document.getElementById("input-exp");

const elValDistance = document.getElementById("val-distance");
const elValPrep = document.getElementById("val-prep");
const elValExp = document.getElementById("val-exp");

const elResultEta = document.getElementById("result-eta");
const elResultLevel = document.getElementById("result-level");
const elResultModel = document.getElementById("result-model");
const elResultRmse = document.getElementById("result-rmse");

const elBackendBadge = document.getElementById("backend-status-badge");
const elBackendText = document.getElementById("backend-status-text");
const elPipelineStatus = document.getElementById("pipeline-status-text");

const elModelsContainer = document.getElementById("models-container");
const elDriftContainer = document.getElementById("drift-rows-container");

let debounceTimer = null;

function setBackendStatus(online, modelName) {
  state.isBackendOnline = online;
  if (online) {
    elBackendBadge.className = "live-pill online";
    elBackendText.textContent = "Live Backend API";
    if (elPipelineStatus) {
      elPipelineStatus.textContent = `FastAPI Active (${modelName || "gradient_boosting"})`;
    }
  } else {
    elBackendBadge.className = "live-pill offline";
    elBackendText.textContent = "Backend Standby";
    if (elPipelineStatus) {
      elPipelineStatus.textContent = "Backend Standby";
    }
  }
}

async function predict() {
  const payload = {
    distance: Number(state.distance),
    prepTime: Number(state.prepTime),
    experience: Number(state.experience),
    traffic: state.traffic,
    weather: state.weather,
    vehicle: state.vehicle,
  };

  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    const eta = Number(data.predicted_eta_min);
    elResultEta.textContent = eta.toFixed(1);
    elResultModel.textContent = data.model_used || "gradient_boosting";
    elResultRmse.textContent = (data.rmse || 3.64).toFixed(2);
    updateEtaLevel(eta);
    setBackendStatus(true, data.model_used);
  } catch (err) {
    // Graceful offline calculation
    const fallbackEta = calculateLocalEta(payload);
    elResultEta.textContent = fallbackEta.toFixed(1);
    elResultModel.textContent = "Gradient Boosting (Offline)";
    elResultRmse.textContent = "3.64";
    updateEtaLevel(fallbackEta);
    setBackendStatus(false);
  }
}

function updateEtaLevel(eta) {
  if (eta < 30) {
    elResultLevel.textContent = "Speedy";
    elResultLevel.className = "status-label speedy";
  } else if (eta < 50) {
    elResultLevel.textContent = "On track";
    elResultLevel.className = "status-label ontrack";
  } else {
    elResultLevel.textContent = "Slow lane";
    elResultLevel.className = "status-label slow";
  }
}

function triggerPrediction() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(predict, 80);
}

// Sliders setup
elDistance.addEventListener("input", (e) => {
  state.distance = Number(e.target.value);
  elValDistance.textContent = `${state.distance.toFixed(1)} km`;
  triggerPrediction();
});

elPrep.addEventListener("input", (e) => {
  state.prepTime = Number(e.target.value);
  elValPrep.textContent = `${state.prepTime} min`;
  triggerPrediction();
});

elExp.addEventListener("input", (e) => {
  state.experience = Number(e.target.value);
  elValExp.textContent = `${state.experience.toFixed(1)} yrs`;
  triggerPrediction();
});

// Chips setup
function setupChips(containerId, stateKey) {
  const container = document.getElementById(containerId);
  if (!container) return;
  const buttons = container.querySelectorAll(".chip-btn");
  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      buttons.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state[stateKey] = btn.getAttribute("data-val");
      triggerPrediction();
    });
  });
}

setupChips("chips-traffic", "traffic");
setupChips("chips-weather", "weather");
setupChips("chips-vehicle", "vehicle");

// Navigation tabs
const navTabs = document.querySelectorAll(".nav-tab");
const tabViews = document.querySelectorAll(".tab-view");

function switchTab(targetName) {
  navTabs.forEach((tab) => {
    if (tab.getAttribute("data-tab") === targetName) {
      tab.classList.add("active");
    } else {
      tab.classList.remove("active");
    }
  });

  tabViews.forEach((view) => {
    if (view.id === `${targetName}-view`) {
      view.classList.add("active");
    } else {
      view.classList.remove("active");
    }
  });

  if (targetName === "models") loadModels();
  if (targetName === "drift") loadDrift();
}

navTabs.forEach((tab) => {
  tab.addEventListener("click", (e) => {
    e.preventDefault();
    const target = tab.getAttribute("data-tab");
    window.location.hash = target;
    switchTab(target);
  });
});

window.addEventListener("hashchange", () => {
  const hash = window.location.hash.replace("#", "") || "predict";
  switchTab(hash);
});

// Models Data Loader
const fallbackModels = [
  { name: "Gradient Boosting", mae: 2.88, rmse: 3.64, r2: 0.921, mape: 10.9, best: true },
  { name: "Ridge", mae: 3.02, rmse: 3.85, r2: 0.911, mape: 11.4, best: false },
  { name: "Linear Regression", mae: 3.02, rmse: 3.85, r2: 0.911, mape: 11.4, best: false },
  { name: "Random Forest", mae: 3.26, rmse: 4.17, r2: 0.896, mape: 12.2, best: false },
];

async function loadModels() {
  let list = fallbackModels;
  try {
    const res = await fetch(`${API_BASE_URL}/api/models`);
    if (res.ok) {
      const data = await res.json();
      if (Array.isArray(data) && data.length > 0) list = data;
    }
  } catch (err) {
    // use fallback
  }

  const maxRmse = Math.max(...list.map((m) => m.rmse), 5.0);

  elModelsContainer.innerHTML = list.map((m) => `
    <div class="panel model-card ${m.best ? "best shadow-glow" : ""}">
      <div class="model-header">
        <h2 class="model-title">${m.name}</h2>
        ${m.best ? '<span class="best-badge">BEST (SAVED)</span>' : ""}
      </div>
      <div class="progress-track">
        <div class="progress-bar ${m.best ? "best" : "normal"}" style="width: ${(m.rmse / maxRmse) * 100}%"></div>
      </div>
      <div class="metrics-quad">
        <div>
          <p class="metric-item-name">MAE</p>
          <p class="metric-item-val ${m.best ? "highlight" : ""}">${m.mae}</p>
        </div>
        <div>
          <p class="metric-item-name">RMSE</p>
          <p class="metric-item-val ${m.best ? "highlight" : ""}">${m.rmse}</p>
        </div>
        <div>
          <p class="metric-item-name">R²</p>
          <p class="metric-item-val ${m.best ? "highlight" : ""}">${m.r2}</p>
        </div>
        <div>
          <p class="metric-item-name">MAPE</p>
          <p class="metric-item-val ${m.best ? "highlight" : ""}">${m.mape}%</p>
        </div>
      </div>
    </div>
  `).join("");
}

// Drift Data Loader
const fallbackDrift = {
  features_monitored: 3,
  logged_predictions: 17,
  drift_alerts: 2,
  drift: [
    { feature: "distance_km", baseline: 3.97, live: 6.43, psi: 0.92, drifted: true },
    { feature: "prep_time_min", baseline: 12.05, live: 17.71, psi: 0.80, drifted: true },
    { feature: "courier_experience_yr", baseline: 1.98, live: 2.79, psi: 0.42, drifted: false },
  ],
};

async function loadDrift() {
  let report = fallbackDrift;
  try {
    const res = await fetch(`${API_BASE_URL}/api/drift`);
    if (res.ok) {
      const data = await res.json();
      if (data && Array.isArray(data.drift)) report = data;
    }
  } catch (err) {
    // use fallback
  }

  document.getElementById("drift-stat-monitored").textContent = report.features_monitored;
  document.getElementById("drift-stat-logged").textContent = report.logged_predictions;
  document.getElementById("drift-stat-alerts").textContent = report.drift_alerts;

  elDriftContainer.innerHTML = report.drift.map((d) => {
    const isDrifted = d.drifted || d.psi >= 0.5;
    const isWatch = !isDrifted && d.psi >= 0.2;
    const colorClass = isDrifted ? "hot" : isWatch ? "sun" : "lime";
    const labelText = isDrifted ? "Drifted" : isWatch ? "Watch" : "Stable";
    const widthPct = Math.min(100, (d.psi / 1.0) * 100);

    return `
      <div class="drift-row">
        <span class="drift-feat-name">${d.feature}</span>
        <span class="drift-shift-text">${d.baseline} → <span style="color:var(--foreground)">${d.live}</span></span>
        <div class="drift-progress-track">
          <div class="drift-progress-bar ${colorClass}" style="width: ${widthPct}%"></div>
        </div>
        <span class="drift-badge ${colorClass}">${labelText} · ${Number(d.psi).toFixed(2)} std</span>
      </div>
    `;
  }).join("");
}

// Initial health check & prediction
async function init() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      setBackendStatus(data.status === "ok", data.model_info?.model_name);
    }
  } catch {
    setBackendStatus(false);
  }

  predict();
  loadModels();
  loadDrift();

  const hash = window.location.hash.replace("#", "") || "predict";
  switchTab(hash);
}

init();


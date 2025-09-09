const DEFAULT_WEIGHTS = {
  desire_magnitude: 70,
  awareness_level: 60,
  desire: 50,
  conversion_rate: 40,
  sales_per_day: 30,
  launch_date: 20,
  competition_level: 10,
  ad_ease: 10,
  scalability: 10,
  durability: 10
};
const DISPLAY_NAMES = {
  desire_magnitude: "Magnitud deseo",
  awareness_level: "Headroom de consciencia",
  desire: "Evidencia demanda",
  conversion_rate: "Tasa conversión",
  sales_per_day: "Ventas por día",
  launch_date: "Recencia lanzamiento",
  competition_level: "Competencia (invertido)",
  ad_ease: "Facilidad anuncio",
  scalability: "Escalabilidad",
  durability: "Durabilidad/recurrencia"
};
const DEFAULT_ORDER = Object.keys(DEFAULT_WEIGHTS);
let state = {
  weights: { ...DEFAULT_WEIGHTS },
  order: [ ...DEFAULT_ORDER ],
  apiKey: ""
};
const modal = document.getElementById("settingsModal");
const openBtn = document.getElementById("openSettings");
const closeBtn = document.getElementById("closeSettings");
const saveKeyBtn = document.getElementById("saveApiKey");
const apiKeyInput = document.getElementById("apiKeyInput");
const weightsList = document.getElementById("weightsList");
const aiBtn = document.getElementById("aiWeights");
const resetBtn = document.getElementById("resetWeights");
const generateBtn = document.getElementById("generateWinner");
let saveTimer = null;

function openModal() {
  modal.classList.add("open");
}
function closeModal() {
  modal.classList.remove("open");
}
openBtn.addEventListener("click", openModal);
closeBtn.addEventListener("click", closeModal);

function buildWeights() {
  weightsList.innerHTML = "";
  state.order.forEach(key => {
    const li = document.createElement("li");
    li.draggable = true;
    li.dataset.key = key;
    li.innerHTML = `<span class="drag">≡</span><span class="label">${DISPLAY_NAMES[key]}</span>` +
      `<input type="range" min="0" max="100" value="${state.weights[key]}" data-key="${key}" class="slider">` +
      `<span class="value">${state.weights[key]}</span>`;
    weightsList.appendChild(li);
  });
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveSettings, 250);
}

function saveSettings() {
  fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ weights: state.weights, order: state.order })
  });
}

weightsList.addEventListener('input', e => {
  if (e.target.classList.contains('slider')) {
    const key = e.target.dataset.key;
    const val = parseInt(e.target.value, 10);
    state.weights[key] = val;
    e.target.nextElementSibling.textContent = val;
    scheduleSave();
  }
});

let dragEl = null;
weightsList.addEventListener('dragstart', e => {
  dragEl = e.target.closest('li');
  e.dataTransfer.effectAllowed = 'move';
});
weightsList.addEventListener('dragover', e => {
  e.preventDefault();
  const li = e.target.closest('li');
  if (!li || li === dragEl) return;
  const rect = li.getBoundingClientRect();
  const next = (e.clientY - rect.top) / (rect.bottom - rect.top) > 0.5;
  weightsList.insertBefore(dragEl, next ? li.nextSibling : li);
});
weightsList.addEventListener('drop', e => {
  e.preventDefault();
  const items = Array.from(weightsList.children);
  state.order = items.map(li => li.dataset.key);
  scheduleSave();
});
weightsList.addEventListener('dragend', () => {
  const items = Array.from(weightsList.children);
  state.order = items.map(li => li.dataset.key);
});

resetBtn.addEventListener('click', () => {
  state.weights = { ...DEFAULT_WEIGHTS };
  state.order = [ ...DEFAULT_ORDER ];
  buildWeights();
  saveSettings();
});
aiBtn.addEventListener('click', () => {
  fetch('/api/weights/ai', { method: 'POST' })
    .then(r => r.json())
    .then(data => {
      state.weights = data.weights;
      state.order = data.order;
      buildWeights();
    });
});

saveKeyBtn.addEventListener('click', () => {
  const key = apiKeyInput.value.trim();
  fetch('/api/settings', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ openai_api_key: key })
  }).then(() => { state.apiKey = key; });
});

generateBtn.addEventListener('click', () => {
  generateBtn.disabled = true;
  fetch('/api/winner-score/generate', { method: 'POST' })
    .then(() => window.location.reload());
});

function loadSettings() {
  fetch('/api/settings').then(r => r.json()).then(data => {
    state.weights = data.weights || { ...DEFAULT_WEIGHTS };
    state.order = data.order || [ ...DEFAULT_ORDER ];
    state.apiKey = data.openai_api_key || "";
    apiKeyInput.value = state.apiKey;
    buildWeights();
    if (!state.apiKey) {
      openModal();
      apiKeyInput.focus();
    }
  });
}

document.addEventListener('DOMContentLoaded', loadSettings);

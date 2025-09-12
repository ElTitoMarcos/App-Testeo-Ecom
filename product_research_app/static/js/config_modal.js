const KEYS = ["price","rating","units_sold","revenue","desire","competition","review_count","image_count","shipping_days","profit_margin"];

function clampInt(v){
  const n = Math.round(Number(v) || 0);
  return Math.max(0, Math.min(100, n));
}

function setSliderAndNumber(k, v){
  const slider = document.getElementById(`w_${k}_range`);
  const number = document.getElementById(`w_${k}_number`);
  if(slider && number){
    slider.value = v;
    number.value = v;
  }
}

function collectWeights(){
  const out = {};
  KEYS.forEach(k => {
    const slider = document.getElementById(`w_${k}_range`);
    if(slider) out[k] = clampInt(slider.value);
  });
  return out;
}

async function openConfig(){
  const overlay = document.getElementById('configOverlay');
  const panel = overlay.querySelector('.modal-panel');
  panel.innerHTML = '';
  KEYS.forEach(k => {
    const wrap = document.createElement('div');
    wrap.className = 'weight-row';
    wrap.innerHTML = `<label>${k}<input type="range" id="w_${k}_range" min="0" max="100" step="1"><input type="number" id="w_${k}_number" min="0" max="100" step="1"></label>`;
    panel.appendChild(wrap);
    const slider = wrap.querySelector('input[type="range"]');
    const number = wrap.querySelector('input[type="number"]');
    slider.addEventListener('input', () => {
      const v = clampInt(slider.value);
      slider.value = v; number.value = v;
    });
    number.addEventListener('input', () => {
      const v = clampInt(number.value);
      number.value = v; slider.value = v;
    });
  });
  const saveBtn = document.createElement('button');
  saveBtn.id = 'saveSettingsBtn';
  saveBtn.textContent = 'Guardar';
  saveBtn.style.marginTop = '12px';
  panel.appendChild(saveBtn);
  saveBtn.addEventListener('click', saveSettings);
  try{
    const res = await fetch('/api/settings');
    const data = await res.json();
    const w = data.weights_raw_int || data.weights || {};
    for(const k in w){ setSliderAndNumber(k, clampInt(w[k])); }
  }catch(e){ console.error('load settings', e); }
  overlay.classList.remove('hidden');
}

async function saveSettings(){
  const weights_raw_int = collectWeights();
  try{
    await fetch('/api/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ weights_raw_int })
    });
  }catch(e){ console.error('save settings', e); }
  closeConfig();
}

function closeConfig(){
  const overlay = document.getElementById('configOverlay');
  overlay.classList.add('hidden');
}

const overlay = document.getElementById('configOverlay');
if(overlay){
  overlay.addEventListener('click', e => { if(e.target === overlay) closeConfig(); });
}

document.addEventListener('keydown', e => {
  const overlay = document.getElementById('configOverlay');
  if(!overlay.classList.contains('hidden') && e.key === 'Escape') closeConfig();
});

document.addEventListener('DOMContentLoaded', () => {
  const btn = document.getElementById('configBtn');
  if(btn) btn.onclick = openConfig;
});

import { fetchJson } from '/static/js/net.js';

const drawer = document.getElementById('analysisDrawer');
const content = document.getElementById('analysisContent');
let currentText = '';

const closeBtn = document.getElementById('closeAnalysis');
if (closeBtn) {
  closeBtn.addEventListener('click', () => drawer.classList.add('hidden'));
}

const exportBtn = document.getElementById('analysisExport');
if (exportBtn) {
  exportBtn.addEventListener('click', () => {
    if (!currentText) return;
    const blob = new Blob([currentText], { type: 'text/plain' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'analysis.txt';
    a.click();
  });
}

export async function openAnalysis(product) {
  try {
    const res = await fetchJson('/analysis', { method: 'POST', body: JSON.stringify(product) });
    currentText = res.analysis || '';
    content.textContent = currentText;
    drawer.classList.remove('hidden');
  } catch (err) {
    console.error(err);
  }
}

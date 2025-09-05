import { fetchJson } from '/static/js/net.js';

const drawer = document.getElementById('analysisDrawer');
const tabs = document.getElementById('analysisTabs');
const sections = {
  resumen: document.getElementById('tab-resumen'),
  fuentes: document.getElementById('tab-fuentes'),
  economia: document.getElementById('tab-economia'),
  riesgos: document.getElementById('tab-riesgos'),
  creatividad: document.getElementById('tab-creatividad')
};
let currentData = null;

function switchTab(name){
  if(!tabs) return;
  [...tabs.querySelectorAll('button')].forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab===name);
  });
  Object.keys(sections).forEach(key => {
    sections[key].classList.toggle('hidden', key!==name);
  });
}

if(tabs){
  tabs.addEventListener('click', e => {
    if(e.target.tagName === 'BUTTON'){
      switchTab(e.target.dataset.tab);
    }
  });
}

const closeBtn = document.getElementById('closeAnalysis');
if(closeBtn){
  closeBtn.addEventListener('click', () => drawer.classList.add('hidden'));
}

const exportBtn = document.getElementById('analysisExport');
if(exportBtn){
  exportBtn.addEventListener('click', () => {
    if(!currentData) return;
    const json = JSON.stringify(currentData, null, 2);
    const blobJson = new Blob([json], {type:'application/json'});
    const aJson = document.createElement('a');
    aJson.href = URL.createObjectURL(blobJson);
    aJson.download = 'analysis.json';
    aJson.click();
    const md = '```json\n' + json + '\n```';
    const blobMd = new Blob([md], {type:'text/markdown'});
    const aMd = document.createElement('a');
    aMd.href = URL.createObjectURL(blobMd);
    aMd.download = 'analysis.md';
    aMd.click();
  });
}

export async function openAnalysis(product){
  try{
    const res = await fetchJson('/api/analyze', {method:'POST', body: JSON.stringify(product)});
    currentData = res.result || {};
    sections.resumen.textContent = JSON.stringify({
      producto: currentData.producto,
      demanda_y_tendencia: currentData.demanda_y_tendencia,
      competencia: currentData.competencia,
      pros: currentData.pros,
      contras: currentData.contras,
      veredicto: currentData.veredicto
    }, null, 2);
    sections.fuentes.textContent = JSON.stringify({
      social_proof: currentData.social_proof,
      fuentes: currentData.fuentes
    }, null, 2);
    sections.economia.textContent = JSON.stringify(currentData.unit_economics, null, 2);
    sections.riesgos.textContent = JSON.stringify(currentData.logistica_y_riesgos, null, 2);
    sections.creatividad.textContent = JSON.stringify(currentData.insights_creativos, null, 2);
    switchTab('resumen');
    drawer.classList.remove('hidden');
  }catch(err){
    console.error(err);
  }
}

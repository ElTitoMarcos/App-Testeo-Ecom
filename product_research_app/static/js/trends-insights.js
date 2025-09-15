// Generador de insights heurísticos (sin GPT)
function localInsights(categories) {
  if (!categories || !categories.length) return ['Sin datos de categorías.'];

  const sortedRev = [...categories].sort((a, b) => Number(b.revenue || 0) - Number(a.revenue || 0));
  const totalRevenue = sortedRev.reduce((s, r) => s + Number(r.revenue || 0), 0) || 1;
  const top3 = sortedRev
    .slice(0, 3)
    .map((c) => `${c.path || c.name} (${(100 * Number(c.revenue || 0) / totalRevenue).toFixed(1)}%)`);

  const avgPrice = categories.reduce((s, c) => s + Number(c.price || 0), 0) / categories.length || 0;
  const highPrice = [...categories]
    .sort((a, b) => Number(b.price || 0) - Number(a.price || 0))
    .slice(0, 3)
    .map((c) => `${c.path || c.name} (€${Number(c.price || 0).toFixed(2).replace('.', ',')})`);

  return [
    `Top 3 por ingresos: ${top3.join(' · ')}`,
    `Precio medio ponderado aprox: €${avgPrice.toFixed(2).replace('.', ',')}`,
    `Categorías con precio medio más alto: ${highPrice.join(' · ')}`
  ];
}

function writeInsights(lines) {
  const box = document.getElementById('insightsContent');
  if (!box) return;
  box.innerHTML = '<ul>' + lines.map((l) => `<li>${l}</li>`).join('') + '</ul>';
}

function buildGptPrompt(categories) {
  const top = [...categories].sort((a, b) => Number(b.revenue || 0) - Number(a.revenue || 0)).slice(0, 10);
  const bullets = top.map(
    (c) =>
      `- ${c.path || c.name}: ingresos=${Number(c.revenue || 0)}, unidades=${Number(c.units || 0)}, precio_medio=${Number(
        c.price || 0
      )}, rating=${Number(c.rating || 0)}`
  );
  return `Analiza estas categorías (ecommerce). Resume oportunidades, riesgos y quick wins en 6 viñetas concisas, priorizando ROI:\n${bullets.join(
    '\n'
  )}`;
}

async function trySendToExistingGpt(prompt) {
  // 1) Intenta encontrar input/botón existentes en la UI
  const input = [...document.querySelectorAll('input,textarea')].find((el) => /gpt/i.test(el.placeholder || ''));
  const sendBtn = [...document.querySelectorAll('button')].find((b) => /enviar consulta a gpt/i.test(b.textContent || ''));

  if (input && sendBtn) {
    input.value = prompt;
    sendBtn.click();
    return true;
  }

  // 2) Si no hay UI accesible, copia al portapapeles
  try {
    await navigator.clipboard.writeText(prompt);
    toast?.info?.('Prompt copiado. Pégalo en tu módulo GPT.'); // si existe toast()
  } catch (_) {}
  return false;
}

function getCategoriesAgg() {
  return window.__latestTrendsData?.categoriesAgg || [];
}

document.addEventListener('click', async (ev) => {
  if (ev.target?.id === 'btnLocalInsights') {
    writeInsights(localInsights(getCategoriesAgg()));
  }
  if (ev.target?.id === 'btnGptInsights') {
    const prompt = buildGptPrompt(getCategoriesAgg());
    await trySendToExistingGpt(prompt);
  }
});

export {};

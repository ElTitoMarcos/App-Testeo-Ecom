import { fetchJson } from '/static/js/net.js';

const btn = document.getElementById('btn-ba-gpt');
if (btn) {
  btn.addEventListener('click', () => {
    if (selection.size !== 1) return;
    const id = Array.from(selection)[0];
    const product = (window.products || []).find(p => String(p.id) === id);
    if (!product) return;

    const box = document.createElement('div');
    box.style.padding = '20px';
    box.innerHTML = `<h3>Análisis BA para producto ${product.id}</h3><p>Usará GPT y puede consumir saldo. ¿Continuar?</p>`;

    let includeCb = null;
    if (product.image_url) {
      const label = document.createElement('label');
      includeCb = document.createElement('input');
      includeCb.type = 'checkbox';
      includeCb.checked = true;
      label.appendChild(includeCb);
      label.appendChild(document.createTextNode(' Incluir imagen'));
      box.appendChild(label);
    }

    const actions = document.createElement('div');
    actions.style.marginTop = '15px';
    actions.style.display = 'flex';
    actions.style.gap = '8px';
    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = 'Cancelar';
    const runBtn = document.createElement('button');
    runBtn.textContent = 'Analizar';
    actions.appendChild(cancelBtn);
    actions.appendChild(runBtn);
    box.appendChild(actions);

    const handle = window.modalManager.open(box, { returnFocus: btn });
    cancelBtn.onclick = () => handle.close();
    runBtn.onclick = async () => {
      runBtn.classList.add('loading');
      runBtn.textContent = 'Pensando…';
      runBtn.disabled = true;
      btn.disabled = true;
      const payload = {
        id: product.id,
        name: product.name,
        category: product.category,
        price: product.price,
        rating: product.rating,
        units_sold: product.units_sold,
        revenue: product.revenue,
        conversion_rate: product.conversion_rate,
        launch_date: product.launch_date,
        date_range: product.date_range,
        image_url: product.image_url,
        desire: product.desire,
        desire_magnitude: product.desire_magnitude,
        awareness_level: product.awareness_level,
        competition_level: product.competition_level
      };
      if (!includeCb || !includeCb.checked) payload.image_url = null;
      const modelSel = document.getElementById('modelSelect');
      const model = modelSel ? modelSel.value : undefined;
      try {
        const resp = await fetchJson('/api/ba/insights', {
          method: 'POST',
          body: JSON.stringify({ product: payload, model })
        });
        Object.assign(product, resp.grid_updates);
        renderTable();
        try {
          await fetchJson(`/products/${product.id}`, {
            method: 'PUT',
            body: JSON.stringify(resp.grid_updates)
          });
        } catch (e) {}
        handle.close();
        showResults(resp);
      } catch (e) {
        handle.close();
      } finally {
        btn.disabled = selection.size !== 1;
      }
    };
  });
}

function showResults(data) {
  const { grid_updates: gu, ba_insights: ba } = data;
  const box = document.createElement('div');
  box.style.maxWidth = '600px';
  box.style.padding = '20px';
  box.innerHTML = '<h3>Resultados BA</h3>';
  const chips = document.createElement('div');
  ['awareness_level', 'desire_magnitude', 'competition_level'].forEach(k => {
    const v = gu[k];
    if (v) {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = v;
      chips.appendChild(chip);
    }
  });
  box.appendChild(chips);
  if (gu.desire) {
    const p = document.createElement('p');
    p.innerHTML = '<strong>Desire:</strong> ' + gu.desire;
    box.appendChild(p);
  }
  const acc = document.createElement('div');
  addSection(acc, 'Ángulos', ba.angles);
  addSection(acc, 'Titulares', ba.headlines);
  addSection(acc, 'Hooks UGC', ba.hooks_ugc);
  if (Array.isArray(ba.objections_and_answers)) {
    const det = document.createElement('details');
    const sum = document.createElement('summary');
    sum.textContent = 'Objeciones y respuestas';
    det.appendChild(sum);
    const ul = document.createElement('ul');
    ba.objections_and_answers.forEach(o => {
      const li = document.createElement('li');
      li.innerHTML = `<strong>${o.objection}</strong>: ${o.answer}`;
      ul.appendChild(li);
    });
    det.appendChild(ul);
    acc.appendChild(det);
  }
  addSection(acc, 'CTAs', ba.cta_options);
  box.appendChild(acc);
  window.modalManager.open(box, { returnFocus: btn });
}

function addSection(container, title, items) {
  if (!items) return;
  const det = document.createElement('details');
  const sum = document.createElement('summary');
  sum.textContent = title;
  det.appendChild(sum);
  if (Array.isArray(items)) {
    const ul = document.createElement('ul');
    items.forEach(it => {
      const li = document.createElement('li');
      li.textContent = it;
      ul.appendChild(li);
    });
    det.appendChild(ul);
  } else {
    const p = document.createElement('p');
    p.textContent = items;
    det.appendChild(p);
  }
  container.appendChild(det);
}

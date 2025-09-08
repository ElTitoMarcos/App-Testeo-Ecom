// product_research_app/static/js/auto_columns_safe.js
(function(){
  const table = document.getElementById('productsTable');
  if(!table) return;

  // Elegimos la última fila del thead (por si hay dobles headers)
  const thead = table.tHead || table.querySelector('thead');
  const headerRow = thead ? thead.rows[thead.rows.length - 1] : null;
  const headers = headerRow ? Array.from(headerRow.cells) : [];

  // Fallback al cuerpo para saber cuántas columnas hay
  const tbody = table.tBodies[0];
  const firstRow = tbody ? tbody.rows[0] : null;
  const colCount = headers.length || (firstRow ? firstRow.cells.length : 0);
  if(!colCount) return;

  // Crea/inserta un <style> idempotente para reglas nth-child
  const STYLE_ID = 'productsTable-dynamic-widths';
  let styleTag = document.getElementById(STYLE_ID);
  if(!styleTag){
    styleTag = document.createElement('style');
    styleTag.id = STYLE_ID;
    document.head.appendChild(styleTag);
  }

  // Utils
  const rootCS = getComputedStyle(document.documentElement);
  const px = v => parseFloat(String(v).replace('px','')) || 0;
  const varPx = (name, fb) => {
    const v = rootCS.getPropertyValue(name);
    return v ? px(v) : fb;
  };

  function idxBy(predicate){
    for(let i=0;i<colCount;i++){
      const th = headers[i];
      const td = firstRow ? firstRow.cells[i] : null;
      if(predicate({i, th, td})) return i;
    }
    return -1;
  }

  // Detecta columnas especiales
  const idxSelect = idxBy(({th, td}) =>
    !!(th && th.querySelector('input[type="checkbox"]')) ||
    !!(td && td.querySelector('input[type="checkbox"]'))
  );

  const idxId = idxBy(({th})=>{
    if(!th) return false;
    const t = th.textContent.trim().toLowerCase();
    return t === 'id';
  });

  const idxImg = idxBy(({th, td})=>{
    const thMatch = th && /imagen|image|img|picture|photo/i.test(th.textContent);
    const tdMatch = td && !!td.querySelector('img');
    return !!(thMatch || tdMatch);
  });

  // Mapa de pesos por título aproximado
  const weightByKey = (txt)=>{
    const t = (txt||'').toLowerCase();
    if(/nombre|name|title/.test(t)) return 1.8;
    if(/categor[ií]a|category/.test(t)) return 1.2;
    if(/precio|price/.test(t)) return 1.0;
    if(/rating/.test(t)) return 1.0;
    if(/unidades|units|vendid/.test(t)) return 1.2;
    if(/ingres|revenue|sales/.test(t)) return 1.3;
    if(/conversi|conversion/.test(t)) return 1.0;
    if(/fecha.*lanz/.test(t)) return 1.0;
    if(/rango.*fech/.test(t)) return 1.1;
    if(/desire|awaren|aware|competition|winner/.test(t)) return 1.0;
    return 1.0;
  };

  function layout(){
    if(!table.parentElement) return;
    const containerW = table.parentElement.clientWidth;

    // Anchuras fijas/semifijas
    const idW   = varPx('--col-id', 56);
    const imgMin = varPx('--col-img-min', 120);
    const imgMax = varPx('--col-img-max', 188);
    const imgW  = Math.max(imgMin, Math.min(imgMax, Math.round(containerW * 0.14)));
    const checkW = idxSelect >= 0 ? 32 : 0;

    // Reserva fija
    let fixed = 0;
    const widths = new Array(colCount).fill(null);

    if(idxSelect >= 0){ widths[idxSelect] = checkW; fixed += checkW; }
    if(idxId >= 0){     widths[idxId]     = idW;    fixed += idW;    }
    if(idxImg >= 0){    widths[idxImg]    = imgW;   fixed += imgW;   }

    // Estimación de padding/bordes por celda
    const pad = 22;
    const paddingTotal = pad * colCount;

    // Columnas auto y pesos
    let totalWeight = 0;
    const autoIdx = [];
    for(let i=0;i<colCount;i++){
      if(widths[i] == null){
        const name = headers[i]?.textContent || '';
        const w = weightByKey(name);
        totalWeight += w;
        autoIdx.push([i, w]);
      }
    }

    const avail = Math.max(0, containerW - fixed - paddingTotal);
    const MIN = 84;
    const usable = Math.max(avail, MIN * autoIdx.length);

    autoIdx.forEach(([i, w])=>{
      const target = Math.floor(usable * (w / (totalWeight || 1)));
      widths[i] = Math.max(MIN, target);
    });

    // Normaliza si por redondeo nos pasamos
    const sum = widths.reduce((a,b)=>a+(b||0),0) + paddingTotal;
    if(sum > containerW && sum > 0){
      const scale = (containerW - paddingTotal) / (sum - paddingTotal);
      for(let i=0;i<colCount;i++){
        widths[i] = Math.max(MIN, Math.floor(widths[i]*scale));
      }
    }

    // Inyecta reglas nth-child para th/td
    let css = '';
    for(let i=0;i<colCount;i++){
      const w = widths[i];
      const nth = i+1;
      css += `
.products-table th:nth-child(${nth}),
.products-table td:nth-child(${nth}){ width: ${w}px; max-width:${w}px; }`;
    }
    // Truncar texto en no-imagen/no-checkbox
    // (Añade class="truncate" en server-side si prefieres)
    css += `
.products-table td:not(.cell-img){ overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }`;

    styleTag.textContent = css;

    // Marca celdas de imagen (si no venían marcadas)
    if(firstRow){
      for(const row of tbody.rows){
        const cell = row.cells[idxImg];
        if(cell && !cell.classList.contains('cell-img')) cell.classList.add('cell-img');
      }
    }
  }

  // Observa cambios de tamaño y recalcula
  const ro = new ResizeObserver(()=>layout());
  ro.observe(table.parentElement || table);
  window.addEventListener('resize', layout);
  if(document.readyState === 'complete') layout();
  else window.addEventListener('load', layout);
})();


// product_research_app/static/js/auto_columns_v2.js
(function(){
  const table = document.getElementById('productsTable');
  if(!table) return;

  const wrap = table.parentElement;
  const ensureColgroup = () => {
    let cg = document.getElementById('products-colgroup');
    if(!cg){
      cg = document.createElement('colgroup');
      cg.id = 'products-colgroup';
      table.insertBefore(cg, table.tHead || table.firstChild);
    }
    return cg;
  };
  const colgroup = ensureColgroup();

  const thead = table.tHead || table.querySelector('thead');
  if(!thead) return;
  const headers = Array.from(thead.querySelectorAll('th'));

  // Asegura .th-inner (para padding-right de fallback)
  headers.forEach(th=>{
    if(!th.querySelector('.th-inner')){
      const span = document.createElement('span');
      span.className = 'th-inner';
      while(th.firstChild) span.appendChild(th.firstChild);
      th.appendChild(span);
    }
  });

  // Construye <col> si falta o difiere el número
  if(colgroup.children.length !== headers.length){
    colgroup.innerHTML = '';
    headers.forEach(th=>{
      const c = document.createElement('col');
      const key = (th.dataset.key
                  || th.getAttribute('data-col')
                  || th.textContent.trim().toLowerCase())
                  .replace(/\s+/g,'-');
      c.setAttribute('data-key', key);
      colgroup.appendChild(c);
    });
  }

  // Utils
  const root = document.documentElement;
  const css  = getComputedStyle(root);
  const px   = v => parseFloat(String(v).replace('px',''))||0;
  const varPx = (name, fb) => { const v = css.getPropertyValue(name); return v ? px(v) : fb; };

  function scrollbarWidth(){
    // Ancho real de scrollbar vertical para fallback
    return Math.max(0, wrap.offsetWidth - wrap.clientWidth);
  }

  function layout(){
    const cols  = Array.from(colgroup.children);
    const tbody = table.tBodies[0];
    if(!tbody || !cols.length) return;

    // Fijas
    const imgW = varPx('--col-img', 160);
    const idW  = varPx('--col-id', 56);
    const checkboxW = headers[0]?.querySelector('input[type="checkbox"]') ? 36 : 0;

    cols.forEach(c=>{
      const key = c.getAttribute('data-key');
      if(/^(select|checkbox)$/i.test(key)) c.style.width = checkboxW + 'px';
      else if(/^id$/i.test(key))           c.style.width = idW + 'px';
      else if(/^(img|imagen|image|picture|photo)$/i.test(key)) c.style.width = imgW + 'px';
      else c.style.width = ''; // auto
    });

    // Reparto del resto (pesos)
    const total = (table.parentElement || table).clientWidth;
    const paddingPerCell = 24;
    const fixed = cols.reduce((s,c)=> s + (c.style.width ? px(c.style.width) : 0), 0);
    const autoCols = cols.filter(c => !c.style.width);

    const weights = {
      name:1.6, nombre:1.6, title:1.6,
      category:1.2, categoría:1.2,
      price:1.0, precio:1.0,
      rating:1.0, unidades:1.0, ingresos:1.0, conversion:1.0, "tasa-conversión":1.0,
      "fecha-lanzamiento":1.0, "rango-fechas":1.0,
      desire:1.0, "desire-magnitude":1.0, "desire-magnetitude":1.0, "awerness-level":1.0, "awareness-level":1.0,
      competition:1.0, "winner-score":1.0
    };

    const sumW = autoCols.reduce((s,c)=> s + (weights[c.getAttribute('data-key')] || 1), 0) || 1;
    const padding = paddingPerCell * headers.length;
    const avail = Math.max(0, total - fixed - padding);
    const MIN = 88;
    const usable = Math.max(avail, MIN * autoCols.length);

    autoCols.forEach(c=>{
      const k = c.getAttribute('data-key');
      const w = (weights[k] || 1) / sumW;
      const target = Math.floor(usable * w);
      c.style.width = Math.max(MIN, target) + 'px';
    });

    // Truncar celdas de texto y marcar clases por columna
    Array.from(tbody.rows).forEach(tr=>{
      Array.from(tr.cells).forEach((td, i)=>{
        const key = cols[i]?.getAttribute('data-key') || '';
        if(/^(id)$/i.test(key)) td.classList.add('col-id');
        if(/^(img|imagen|image|picture|photo)$/i.test(key)) td.classList.add('col-img');
        if(!/^(img|imagen|image|picture|photo|checkbox|select|id)$/i.test(key)){
          td.classList.add('truncate');
        }
      });
    });

    // Fallback anti-descuadre: compensa padding de headers con ancho real de scrollbar
    root.style.setProperty('--sbw', scrollbarWidth() + 'px');
  }

  // Observadores
  const ro = new ResizeObserver(()=>layout());
  ro.observe(wrap);
  window.addEventListener('resize', layout);
  if(document.readyState === 'complete') layout();
  else window.addEventListener('load', layout);
})();


// Notas rápidas
// Ajusta el tamaño de Imagen cambiando --col-img en el CSS.
// Los pesos de columnas se controlan en el objeto weights.
// Todo es idempotente: si ya existe colgroup o .th-inner, no se duplica.

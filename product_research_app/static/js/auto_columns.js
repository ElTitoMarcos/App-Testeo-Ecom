// product_research_app/static/js/auto_columns.js
(function(){
  const table = document.getElementById('productTable');
  if(!table) return;
  const colgroup = document.getElementById('products-colgroup') || (()=> {
    const cg = document.createElement('colgroup');
    cg.id = 'products-colgroup';
    table.insertBefore(cg, table.tHead || table.firstChild);
    return cg;
  })();

  // Obtiene headers y crea <col> por columna (idempotente)
  const thead = table.tHead || table.querySelector('thead');
  if(!thead) return;
  const headers = Array.from(thead.querySelectorAll('th'));
  if(colgroup.children.length !== headers.length){
    colgroup.innerHTML = '';
    headers.forEach((th, i)=>{
      const c = document.createElement('col');
      // Normaliza clave de columna por texto/clases
      const key = (th.dataset.key
                  || th.getAttribute('data-col')
                  || th.textContent.trim().toLowerCase())
                  .replace(/\s+/g,'-');
      c.setAttribute('data-key', key);
      colgroup.appendChild(c);
    });
  }

  const rootStyle = getComputedStyle(document.documentElement);
  const px = v => parseFloat(String(v).replace('px',''))||0;

  function varPx(name, fallback){
    const v = rootStyle.getPropertyValue(name);
    return v ? px(v) : fallback;
  }

  function layout(){
    const cols = Array.from(colgroup.children);
    const tbody = table.tBodies[0];
    if(!tbody) return;

    const container = table.parentElement || table;
    const total = container.clientWidth;

    // Fijas
    const imgW = varPx('--col-img', 140);
    const idW  = varPx('--col-id', 56);
    const checkboxW = headers[0]?.querySelector('input[type="checkbox"]') ? 36 : 0;

    // Asigna fijas por data-key aproximada
    cols.forEach((c,i)=>{
      const key = c.getAttribute('data-key');
      if(/^(select|checkbox)$/i.test(key)) c.style.width = checkboxW+'px';
      else if(/^(id)$/i.test(key))       c.style.width = idW+'px';
      else if(/^(img|imagen|image|picture|photo)$/i.test(key)) c.style.width = imgW+'px';
      else c.style.width = ''; // se calculará abajo
    });

    const fixed = cols.reduce((sum,c)=>{
      return sum + (c.style.width ? px(c.style.width) : 0);
    }, 0);

    // Columnas auto con pesos
    const weights = { name:1.6, nombre:1.6, title:1.6, category:1.2, categoría:1.2 };
    const autoCols = cols.filter(c => !c.style.width);
    const sumW = autoCols.reduce((s,c)=>{
      const k = c.getAttribute('data-key');
      return s + (weights[k] || 1);
    }, 0);

    // Relleno seguro para que siempre quepan todas
    const padding = 24 * headers.length; // estimación de paddings/bordes
    const avail = Math.max(0, total - fixed - padding);
    const MIN = 80; // mínimo de cada columna para que siempre entren
    const hardMinTotal = MIN * autoCols.length;
    const usable = Math.max(avail, hardMinTotal); // si falta espacio, reducimos a MIN y elipsis

    autoCols.forEach(c=>{
      const k = c.getAttribute('data-key');
      const w = (weights[k] || 1) / sumW;
      const target = Math.floor(usable * w);
      c.style.width = Math.max(MIN, target) + 'px';
    });

    // Aplica clase truncate a celdas de texto
    const rows = Array.from(tbody.rows);
    rows.forEach(tr=>{
      Array.from(tr.cells).forEach((td, i)=>{
        const key = cols[i]?.getAttribute('data-key') || '';
        if(!/^(img|imagen|image|picture|photo|checkbox|select|id)$/i.test(key)){
          td.classList.add('truncate');
        }
        if(/^(id)$/i.test(key)) td.classList.add('col-id');
        if(/^(img|imagen|image|picture|photo)$/i.test(key)) td.classList.add('col-img');
      });
    });

    // Añade clases a headers
    headers.forEach((th,i)=>{
      const key = cols[i]?.getAttribute('data-key') || '';
      if(!/^(img|imagen|image|picture|photo|checkbox|select|id)$/i.test(key)){
        th.classList.add('truncate');
      }
      if(/^(id)$/i.test(key)) th.classList.add('col-id');
      if(/^(img|imagen|image|picture|photo)$/i.test(key)) th.classList.add('col-img');
    });
  }

  // Observadores
  const ro = new ResizeObserver(()=>layout());
  ro.observe(table.parentElement || table);
  window.addEventListener('resize', layout);

  // Primera pasada tras carga
  if(document.readyState === 'complete') layout();
  else window.addEventListener('load', layout);
})();

export async function fetchJson(url, opts){
  try{
    const r = await fetch(url, Object.assign({headers:{'Content-Type':'application/json'}}, opts||{}));
    const j = await r.json();
    if(!r.ok || j.ok===false){ const lp=j.log_path||''; toast.error(j.message||'Error', {actionText: lp?'Copiar ruta':'', onAction: ()=>navigator.clipboard.writeText(lp)}); throw new Error(j.message||'Error'); }
    return j;
  }catch(err){ toast.error('Error de red'); throw err; }
}

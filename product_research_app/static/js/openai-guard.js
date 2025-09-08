window.EC = window.EC || {};
(function(ns){
  let toastEl = null;
  async function check(){
    let ok = false;
    try {
      const res = await fetch('/api/openai/status');
      const data = await res.json();
      ok = !!(data && data.ok);
    } catch {
      ok = false;
    }
    window.OPENAI_ENABLED = ok;
    toggle();
    if(!ok){
      if(!toastEl && window.toast && toast.error){
        toastEl = toast.error('Falta API de OpenAI. Añádela en Configuración para usar IA.', {duration:0});
      }
    } else if(toastEl && toastEl.parentNode){
      toastEl.parentNode.removeChild(toastEl);
      toastEl = null;
    }
    return ok;
  }
  function toggle(){
    const ctrls = document.querySelectorAll('[data-requires-openai]');
    ctrls.forEach(el=>{
      if(window.OPENAI_ENABLED){
        el.disabled = false;
        el.removeAttribute('aria-disabled');
      } else {
        el.disabled = true;
        el.setAttribute('aria-disabled','true');
      }
    });
  }
  ns.recheckOpenAI = function(){ return check(); };
  if(document.readyState === 'loading'){
    document.addEventListener('DOMContentLoaded', check, {once:true});
  } else {
    check();
  }
})(window.EC);

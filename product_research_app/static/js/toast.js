(function(){
  const c = document.createElement('div'); c.id='toast-container'; document.body.appendChild(c);
  function make(type, msg, opt={}){
    const el = document.createElement('div'); el.className='toast '+type;
    const span = document.createElement('div'); span.textContent = msg; el.appendChild(span);
    if(opt.actionText && opt.onAction){
      const b = document.createElement('button'); b.className='action'; b.textContent=opt.actionText; b.onclick=()=>{ opt.onAction(); c.removeChild(el); };
      el.appendChild(b);
    }
    if(opt.actionText2 && opt.onAction2){
      const b2 = document.createElement('button'); b2.className='action'; b2.textContent=opt.actionText2; b2.onclick=()=>{ opt.onAction2(); c.removeChild(el); };
      el.appendChild(b2);
    }
    c.appendChild(el);
    setTimeout(()=>{ if(el.parentNode) c.removeChild(el); }, opt.duration||5000);
  }
  window.toast = { success:(m,o)=>make('success',m,o), error:(m,o)=>make('error',m,o), info:(m,o)=>make('info',m,o) };
})();

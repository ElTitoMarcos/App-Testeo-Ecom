const IMPORT_TASK_LS_KEY = 'last_import_task';
const DEFAULT_IMPORT_PHASE = 'Importando catálogo';

window.currentTaskId = window.currentTaskId || null;

(function setupInlineProgress() {
  const host = document.getElementById('progress-info');
  const fill = host?.querySelector('.pi-fill');
  const pctEl = host?.querySelector('.pi-pct');
  const phaseEl = host?.querySelector('.pi-phase');
  const subEl = host?.querySelector('.pi-sub');
  const sepEl = host?.querySelector('.pi-sep');
  const cancelBtn = document.getElementById('btn-cancel-import');

  const clamp = (value) => {
    const num = Number(value);
    if (!Number.isFinite(num)) return 0;
    return Math.max(0, Math.min(100, num));
  };

  const currentPhaseText = () => {
    const txt = phaseEl?.textContent?.trim();
    return txt && txt.length ? txt : DEFAULT_IMPORT_PHASE;
  };

  const currentSubText = () => subEl?.textContent || '';

  const show = (visible) => {
    if (!host) return;
    host.hidden = !visible;
    if (!visible) {
      if (fill) fill.style.width = '0%';
      if (pctEl) pctEl.textContent = '0%';
    }
  };

  const setProgress = (pct, phase, sub) => {
    const clamped = clamp(pct ?? 0);
    if (fill) fill.style.width = `${clamped}%`;
    if (pctEl) pctEl.textContent = `${clamped}%`;
    if (phase !== undefined && phaseEl) phaseEl.textContent = phase || '';
    if (sub !== undefined && subEl) {
      const text = sub || '';
      subEl.textContent = text;
      if (sepEl) sepEl.hidden = text.length === 0;
    }
  };

  window.onImportStart = (taskId) => {
    window.currentTaskId = taskId || null;
    setProgress(0, DEFAULT_IMPORT_PHASE, 'Preparando…');
    show(true);
    if (cancelBtn) cancelBtn.disabled = false;
  };

  window.onImportTick = (pct = 0, phase, sub) => {
    const phaseText = typeof phase === 'string' && phase.trim() ? phase.trim() : currentPhaseText();
    let subText;
    if (sub !== undefined) {
      subText = typeof sub === 'string' ? sub : String(sub ?? '');
    } else {
      subText = currentSubText();
    }
    setProgress(pct, phaseText, subText);
  };

  window.onImportEnd = () => {
    window.currentTaskId = null;
    try { localStorage.removeItem(IMPORT_TASK_LS_KEY); }
    catch (e) {}
    if (cancelBtn) cancelBtn.disabled = false;
    setTimeout(() => show(false), 160);
  };

  cancelBtn?.addEventListener('click', async () => {
    if (!window.currentTaskId) return;
    const taskId = window.currentTaskId;
    cancelBtn.disabled = true;
    try {
      await fetch('/_import_cancel', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ task_id: taskId })
      });
    } catch (e) {}
    try {
      window.stopImportPolling?.();
    } catch (e) {}
    try { localStorage.removeItem(IMPORT_TASK_LS_KEY); }
    catch (e) {}
    window.currentTaskId = null;
    setProgress(100, 'Cancelado', '');
    setTimeout(() => show(false), 600);
    cancelBtn.disabled = false;
  });
})();

(function calcStickyOffset() {
  const root = document.documentElement;
  const px = (n) => `${n || 0}px`;
  function recalc() {
    const topbar = document.querySelector('.topbar');
    const toolbar = document.querySelector('.toolbar');
    root.style.setProperty('--topbar-h', px(topbar?.offsetHeight || 0));
    root.style.setProperty('--toolbar-h', px(toolbar?.offsetHeight || 0));
  }
  window.addEventListener('resize', recalc, { passive: true });
  new MutationObserver(recalc).observe(document.body, {
    subtree: true,
    attributes: true,
    attributeFilter: ['class', 'style', 'hidden']
  });
  requestAnimationFrame(recalc);
})();

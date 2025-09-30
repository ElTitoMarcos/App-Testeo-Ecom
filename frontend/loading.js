const DEFAULT_JOB_KEY = '__default__';
const jobPct = new Map();

function clampPct(value) {
  const num = Number(value);
  if (!Number.isFinite(num)) return null;
  return Math.max(0, Math.min(100, Math.round(num)));
}

function phaseLabel(phase) {
  switch ((phase || '').toString()) {
    case 'triage':
      return 'Analizando lotes';
    case 'main':
      return 'IA generando…';
    case 'post':
      return 'Finalizando…';
    default:
      return 'Procesando…';
  }
}

function updateProgressBar(pct, jobId, phase) {
  const key = jobId ? String(jobId) : DEFAULT_JOB_KEY;
  jobPct.set(key, pct);

  const host = document.getElementById('progress-slot-global');
  if (!host) return;

  const fill = host.querySelector('.progress-fill');
  if (fill) fill.style.width = `${pct}%`;

  const pctEl = host.querySelector('.progress-percent');
  if (pctEl) pctEl.textContent = `${pct}%`;

  const stageEl = host.querySelector('.progress-stage');
  if (stageEl && phase) stageEl.textContent = phaseLabel(phase);
}

export function handleImportProgressMessage(message, fallback) {
  if (!message || typeof message !== 'object') {
    if (typeof fallback === 'function') fallback(message);
    return;
  }

  const { job_id, pct, status, phase } = message;
  const normalized = clampPct(pct);
  if (normalized !== null) {
    updateProgressBar(normalized, job_id, phase);
  } else if (typeof fallback === 'function') {
    fallback(message);
  }

  if (String(status || '').toLowerCase() === 'done') {
    updateProgressBar(100, job_id, phase || 'post');
  }
}

export function getLastPct(jobId) {
  const key = jobId ? String(jobId) : DEFAULT_JOB_KEY;
  return jobPct.get(key) ?? null;
}

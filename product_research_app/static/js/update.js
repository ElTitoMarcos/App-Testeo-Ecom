import { fetchJson, post } from './net.js';

const banner = document.getElementById('updateBanner');
if (banner) {
  const POLL_INTERVAL = 1000 * 60 * 15;
  let pollTimer = null;

  async function refreshStatus(options = {}) {
    if (!banner) return;
    try {
      const data = await fetchJson('/api/update/status', { method: 'GET' });
      renderStatus(data.status || {}, options);
    } catch (err) {
      if (options?.silent) return;
      console.debug('update status failed', err);
    }
  }

  function formatBytes(bytes) {
    if (!bytes || Number.isNaN(Number(bytes))) return '';
    const thresh = 1024;
    if (Math.abs(bytes) < thresh) {
      return `${bytes} B`;
    }
    const units = ['KB', 'MB', 'GB'];
    let u = -1;
    do {
      bytes /= thresh;
      ++u;
    } while (Math.abs(bytes) >= thresh && u < units.length - 1);
    return `${bytes.toFixed(1)} ${units[u]}`;
  }

  function renderStatus(status, options = {}) {
    if (!status || status.enabled === false) {
      banner.style.display = 'none';
      banner.innerHTML = '';
      return;
    }
    const available = Boolean(status.update_available && status.staged_path);
    const pendingRestart = Boolean(status.pending_restart);
    const latest = status.latest_version || status.current_version;
    const downloadSize = formatBytes(status.download_size);
    const notes = (status.release_notes || '').split('\n').filter(Boolean).slice(0, 5);

    if (!available && !pendingRestart && options?.hideWhenIdle !== false) {
      banner.style.display = 'none';
      banner.innerHTML = '';
      return;
    }

    const parts = [];
    parts.push('<strong>Actualización del escritorio</strong>');
    parts.push(`<p>Versión instalada: <code>${status.current_version || 'desconocida'}</code></p>`);
    if (available) {
      parts.push(`<p>Nueva versión: <code>${latest}</code>${downloadSize ? ` · ${downloadSize}` : ''}</p>`);
      if (notes.length) {
        parts.push('<details><summary>Notas de la versión</summary>');
        parts.push('<ul>');
        for (const line of notes) {
          parts.push(`<li>${line}</li>`);
        }
        parts.push('</ul></details>');
      }
      parts.push('<div class="actions">');
      parts.push('<button type="button" id="updateApplyBtn">Instalar y reiniciar</button>');
      parts.push('<button type="button" id="updateLaterBtn">Recordar después</button>');
      parts.push('<button type="button" id="updateRefreshBtn">Buscar otra vez</button>');
      parts.push('</div>');
    } else if (pendingRestart) {
      parts.push('<p>La actualización se instaló correctamente. Reinicia la aplicación para finalizar.</p>');
      parts.push('<div class="actions">');
      parts.push('<button type="button" id="updateAckBtn">Aceptar</button>');
      parts.push('</div>');
    }

    banner.innerHTML = parts.join('');
    banner.style.display = 'block';

    const applyBtn = document.getElementById('updateApplyBtn');
    const laterBtn = document.getElementById('updateLaterBtn');
    const refreshBtn = document.getElementById('updateRefreshBtn');
    const ackBtn = document.getElementById('updateAckBtn');

    if (applyBtn) {
      applyBtn.addEventListener('click', async () => {
        applyBtn.disabled = true;
        try {
          await post('/api/update/apply', {});
          toast.success('Actualización instalada. Reinicia la app.');
          renderStatus(Object.assign({}, status, { pending_restart: true, update_available: false }), { hideWhenIdle: false });
        } catch (err) {
          applyBtn.disabled = false;
        }
      });
    }
    if (laterBtn) {
      laterBtn.addEventListener('click', () => {
        banner.style.display = 'none';
      });
    }
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => refreshStatus({ silent: true }));
    }
    if (ackBtn) {
      ackBtn.addEventListener('click', async () => {
        await post('/api/update/ack-restart', {});
        banner.style.display = 'none';
      });
    }
  }

  refreshStatus({ hideWhenIdle: false });
  pollTimer = setInterval(() => refreshStatus({ silent: true }), POLL_INTERVAL);

  window.addEventListener('beforeunload', () => {
    if (pollTimer) {
      clearInterval(pollTimer);
    }
  });
}

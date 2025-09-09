function showApiKeyModal(firstRun = false) {
  const modal = document.getElementById("apiKeyModal");
  const title = document.getElementById("apiKeyModalTitle");
  const help  = document.getElementById("apiKeyModalHelp");
  const input = document.getElementById("apiKeyInput");
  title.textContent = firstRun ? "Añadir API Key" : "Cambiar API Key";
  help.textContent  = firstRun
    ? "Necesitas una clave de API para usar la IA. Se guarda localmente (config.json)."
    : "Introduce la nueva clave para actualizarla.";
  input.value = "";
  modal.classList.remove("hidden");
  input.focus();
}
function hideApiKeyModal() {
  document.getElementById("apiKeyModal").classList.add("hidden");
}
async function saveApiKey() {
  const key = document.getElementById("apiKeyInput").value.trim();
  if (!key) { alert("Introduce una API key válida"); return; }
  const res = await fetch("/api/config/api-key", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({ api_key: key })
  });
  const data = await res.json();
  if (data.ok) {
    hideApiKeyModal();
    document.dispatchEvent(new CustomEvent("api-key:updated"));
    if (window.toast) toast("API Key guardada");
  } else {
    alert(data.error || "Error guardando la API key");
  }
}

// Exponer helpers globales
window.showApiKeyModal = showApiKeyModal;
window.requireApiKeyOrAsk = async function () {
  try {
    const r = await fetch("/api/config");
    const j = await r.json();
    if (!j.has_api_key) { showApiKeyModal(true); return false; }
    return true;
  } catch {
    showApiKeyModal(true);
    return false;
  }
};

document.addEventListener("DOMContentLoaded", () => {
  const cancel = document.getElementById("apiKeyCancel");
  const save   = document.getElementById("apiKeySave");
  if (cancel) cancel.addEventListener("click", hideApiKeyModal);
  if (save)   save.addEventListener("click", saveApiKey);

  // Chequeo al arrancar: bloquea hasta tener clave
  (async () => {
    try {
      const r = await fetch("/api/config");
      const j = await r.json();
      if (!j.has_api_key) showApiKeyModal(true);
    } catch (e) { showApiKeyModal(true); }
  })();
});

import React, { useEffect, useRef, useState } from "react";

// Order matters - used to render sliders
const FIELDS = [
  { key: "price", label: "Price" },
  { key: "rating", label: "Rating" },
  { key: "units_sold", label: "Units Sold" },
  { key: "revenue", label: "Revenue" },
  { key: "desire", label: "Desire" },
  { key: "competition", label: "Competition" },
  { key: "oldness", label: "Oldness" },
];

const DEFAULTS_50: Record<string, number> = {
  price: 50,
  rating: 50,
  units_sold: 50,
  revenue: 50,
  desire: 50,
  competition: 50,
  oldness: 50,
};

export default function SettingsModal() {
  const [weights, setWeights] = useState<Record<string, number>>(DEFAULTS_50);
  const loadedRef = useRef(false); // evita PUTs antes de cargar

  // 2) Carga al montar (y solo entonces habilita autosave)
  useEffect(() => {
    let alive = true;
    fetch("/api/config/weights")
      .then((r) => r.json())
      .then((d) => {
        if (!alive) return;
        const w = d && d.weights ? d.weights : {};
        setWeights({ ...DEFAULTS_50, ...w });
        loadedRef.current = true;
      })
      .catch(() => {
        // Fallback: mantenemos DEFAULTS_50 en memoria
        loadedRef.current = true;
      });
    return () => {
      alive = false;
    };
  }, []);

  // 3) Debounce de guardado (solo si loadedRef.current === true)
  const saveDebounced = useRef<ReturnType<typeof setTimeout> | null>(null);

  function persistWeights(next: Record<string, number>, immediate = false) {
    if (!loadedRef.current) return; // no guardes antes de cargar
    const send = () => {
      fetch("/api/config/weights", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ weights: next }),
        keepalive: true,
      }).catch(() => {});
    };
    if (immediate) {
      if (saveDebounced.current) clearTimeout(saveDebounced.current);
      send();
    } else {
      if (saveDebounced.current) clearTimeout(saveDebounced.current);
      saveDebounced.current = setTimeout(send, 300);
    }
  }

  // 4) Handler de sliders (enteros 0–100)
  const handleChange = (key: string, v: number) => {
    const value = Math.max(0, Math.min(100, Math.round(v)));
    const next = { ...weights, [key]: value };
    setWeights(next); // UI inmediata
    persistWeights(next, false); // guarda debounced
  };

  // 5) Reset: todos a 50 y guardado inmediato
  const onReset = () => {
    const next = { ...DEFAULTS_50 };
    setWeights(next);
    persistWeights(next, true); // PUT inmediato
  };

  // 6) Flush al cerrar modal / cambiar pestaña (por si hay debounce pendiente)
  useEffect(() => {
    const flush = () => {
      if (saveDebounced.current) {
        clearTimeout(saveDebounced.current);
        persistWeights(weights, true);
      }
    };
    window.addEventListener("beforeunload", flush);
    const visHandler = () => {
      if (document.visibilityState === "hidden") flush();
    };
    document.addEventListener("visibilitychange", visHandler);
    return () => {
      window.removeEventListener("beforeunload", flush);
      document.removeEventListener("visibilitychange", visHandler as any);
    };
  }, [weights]);

  return (
    <div className="settings-modal">
      {FIELDS.map((field) => (
        <div key={field.key} className="weight-row">
          <label className="weight-label" htmlFor={`weight-${field.key}`}>
            {field.label}
          </label>
          <input
            id={`weight-${field.key}`}
            className="weight-range"
            type="range"
            min={0}
            max={100}
            step={1}
            value={weights[field.key] ?? 50}
            onChange={(e) => handleChange(field.key, Number(e.target.value))}
          />
          <div className="weight-value">{weights[field.key] ?? 50}</div>
        </div>
      ))}
      <button onClick={onReset}>Reset</button>
    </div>
  );
}


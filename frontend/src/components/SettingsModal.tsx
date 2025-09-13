import React, { useEffect, useRef, useState } from "react";

// Campos válidos para Winner Score (orden no importa)
const WEIGHT_KEYS = [
  "price",
  "rating",
  "units_sold",
  "revenue",
  "desire",
  "competition",
  "oldness",
] as const;

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

const DEFAULTS_50: Record<(typeof WEIGHT_KEYS)[number], number> = WEIGHT_KEYS.reduce(
  (acc, k) => {
    acc[k] = 50;
    return acc;
  },
  {} as Record<(typeof WEIGHT_KEYS)[number], number>
);

export default function SettingsModal() {
  const [weights, setWeights] = useState<Record<string, number>>(DEFAULTS_50);
  const loadedRef = useRef(false);
  const saveDebounced = useRef<ReturnType<typeof setTimeout> | null>(null);

  // CARGA (RAW)
  useEffect(() => {
    let alive = true;
    fetch("/api/config/winner-weights")
      .then((r) => r.json())
      .then((d) => {
        if (!alive) return;
        const raw = d && d.weights ? d.weights : DEFAULTS_50;
        setWeights({ ...DEFAULTS_50, ...raw });
        loadedRef.current = true;
      })
      .catch(() => {
        loadedRef.current = true;
      });
    return () => {
      alive = false;
    };
  }, []);

  // PATCH debounced (RAW). NO aplicar respuesta al estado
  function patchWeights(next: Record<string, number>, immediate = false) {
    if (!loadedRef.current) return;
    const send = () => {
      fetch("/api/config/winner-weights", {
        method: "PATCH",
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

  const onChangeSlider = (key: string, v: number) => {
    const value = Math.max(0, Math.min(100, Math.round(v)));
    const next = { ...weights, [key]: value };
    setWeights(next);
    patchWeights(next, false);
  };

  const onReset = async () => {
    // 1) Cancela cualquier guardado debounced pendiente que pudiera pisarte el reset
    if (saveDebounced?.current) clearTimeout(saveDebounced.current);

    // 2) Construye el objeto RAW con todos a 50 (neutro)
    const fifty: Record<string, number> = WEIGHT_KEYS.reduce((acc, k) => {
      acc[k] = 50;
      return acc;
    }, {} as Record<string, number>);

    // 3) Refleja el cambio en la UI inmediatamente
    setWeights((prev) => ({ ...prev, ...fifty }));

    // 4) Persiste en tu endpoint actual (RAW). Ignoramos la respuesta a propósito.
    try {
      await fetch("/api/config/winner-weights", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ weights: fifty }),
        keepalive: true,
      });
    } catch {
      /* opcional: toast de error silencioso */
    }
  };

  // Flush si se cierra rápido el modal
  useEffect(() => {
    const flush = () => {
      if (saveDebounced.current) {
        clearTimeout(saveDebounced.current);
        patchWeights(weights, true);
      }
    };
    return () => flush();
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
            onChange={(e) => onChangeSlider(field.key, Number(e.target.value))}
          />
          <div className="weight-value">{weights[field.key] ?? 50}</div>
        </div>
      ))}
      <button onClick={onReset}>Reset</button>
    </div>
  );
}


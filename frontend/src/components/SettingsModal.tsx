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
  { key: "awareness", label: "Awareness" },
];

const DEFAULTS_50 = {
  price: 50,
  rating: 50,
  units_sold: 50,
  revenue: 50,
  desire: 50,
  competition: 50,
  oldness: 50,
  awareness: 50,
};

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
        const raw =
          d && typeof d === "object"
            ? (d.weights || d.winner_weights || d)
            : {};
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
      saveDebounced.current = setTimeout(send, 700);
    }
  }

  const onChangeSlider = (key: string, v: number) => {
    const value = Math.max(0, Math.min(100, Math.round(v)));
    const next = { ...weights, [key]: value };
    setWeights(next);
    patchWeights(next, false);
  };

  const onReset = () => {
    const next = { ...DEFAULTS_50 };
    setWeights(next);
    patchWeights(next, true);
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


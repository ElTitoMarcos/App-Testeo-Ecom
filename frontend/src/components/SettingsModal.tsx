import React, { useState, useEffect, useRef } from "react";

const DEFAULTS_50: Record<string, number> = {
  price: 50,
  rating: 50,
  units_sold: 50,
  revenue: 50,
  desire: 50,
  competition: 50,
  oldness: 50,
};

const SettingsModal: React.FC = () => {
  // null indicates we're still loading
  const [weights, setWeights] = useState<Record<string, number> | null>(null);

  // load weights on mount
  useEffect(() => {
    let alive = true;
    fetch("/api/config/weights")
      .then((r) => r.json())
      .then((d) => {
        if (!alive) return;
        const w = d && d.weights ? d.weights : {};
        setWeights({ ...DEFAULTS_50, ...w });
      })
      .catch(() => {
        if (!alive) return;
        setWeights({ ...DEFAULTS_50 });
      });
    return () => {
      alive = false;
    };
  }, []);

  // debounced persistence
  const saveDebounced = useRef<ReturnType<typeof setTimeout> | null>(null);
  function persistWeights(next: Record<string, number>) {
    if (saveDebounced.current) clearTimeout(saveDebounced.current);
    saveDebounced.current = setTimeout(() => {
      fetch("/api/config/weights", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ weights: next }),
        keepalive: true,
      }).catch(() => {});
    }, 300);
  }

  // flush pending save when unmounting
  useEffect(() => {
    return () => {
      if (saveDebounced.current) {
        clearTimeout(saveDebounced.current);
        if (weights) {
          fetch("/api/config/weights", {
            method: "PUT",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ weights }),
            keepalive: true,
          }).catch(() => {});
        }
      }
    };
  }, [weights]);

  const handleChange = (key: string, v: number) => {
    if (!weights) return;
    const value = Math.max(0, Math.min(100, Math.round(v)));
    const next = { ...weights, [key]: value };
    setWeights(next);
    persistWeights(next);
  };

  const onReset = () => {
    const next = { ...DEFAULTS_50 };
    setWeights(next);
    fetch("/api/config/weights", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ weights: next }),
      keepalive: true,
    }).catch(() => {});
  };

  const fields = Object.keys(DEFAULTS_50);

  if (!weights) {
    return <div>Loading...</div>;
  }

  return (
    <div className="settings-modal">
      <button type="button" onClick={onReset}>
        Reset
      </button>
      {fields.map((field) => (
        <div key={field} className="weight-field">
          <label>{field}</label>
          <input
            className="weight-range"
            type="range"
            min={0}
            max={100}
            step={1}
            value={weights?.[field] ?? 50}
            onChange={(e) => handleChange(field, Number(e.target.value))}
            onMouseUp={(e) => handleChange(field, Number((e.target as HTMLInputElement).value))}
            onTouchEnd={(e) => handleChange(field, Number((e.target as HTMLInputElement).value))}
          />
          <span>{weights[field]}</span>
        </div>
      ))}
    </div>
  );
};

export default SettingsModal;


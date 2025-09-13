import { useEffect, useRef, useState } from "react";

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
  const saveDebounced = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    fetch("/api/config/weights")
      .then((r) => r.json())
      .then((d) => {
        const w = d && d.weights ? d.weights : {};
        const merged = { ...DEFAULTS_50, ...w };
        setWeights(merged);
      })
      .catch(() => {
        setWeights(DEFAULTS_50);
      });
  }, []);

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

  const handleChange = (key: string, v: number) => {
    const value = Math.max(0, Math.min(100, Math.round(v)));
    const next = { ...weights, [key]: value };
    setWeights(next);
    persistWeights(next);
  };

  const onReset = () => {
    setWeights(DEFAULTS_50);
    persistWeights(DEFAULTS_50);
  };

  const fields = [
    { key: "price", label: "Price" },
    { key: "rating", label: "Rating" },
    { key: "units_sold", label: "Units Sold" },
    { key: "revenue", label: "Revenue" },
    { key: "desire", label: "Desire" },
    { key: "competition", label: "Competition" },
    { key: "oldness", label: "Oldness" },
  ];

  return (
    <div>
      {fields.map((field) => (
        <div key={field.key}>
          <label>{field.label}</label>
          <input
            className="weight-range"
            type="range"
            min={0}
            max={100}
            step={1}
            value={weights[field.key] ?? 50}
            onChange={(e) => handleChange(field.key, Number(e.target.value))}
          />
        </div>
      ))}
      <button onClick={onReset}>Reset</button>
    </div>
  );
}


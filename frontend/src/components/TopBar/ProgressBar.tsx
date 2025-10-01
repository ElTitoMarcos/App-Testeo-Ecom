
type ProgressBarProps = {
  progress: number;
  total: number;
  label: string;
};

export function ProgressBar({ progress, total, label }: ProgressBarProps) {
  const pct = total > 0 ? Math.min(100, Math.round((progress / total) * 100)) : 0;
  const complete = pct >= 100;
  return (
    <div className="topbar-progress">
      <div className="topbar-progress__bar">
        <div
          className={`topbar-progress__fill${complete ? " topbar-progress__fill--complete" : ""}`}
          style={{ width: `${Math.max(0, pct)}%` }}
        />
      </div>
      {/* Eliminamos el contador (0/100) que confundía */}
      <span>{label}</span>
    </div>
  );
}

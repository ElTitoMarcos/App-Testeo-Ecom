import { useEffect } from "react";
import { useUIStore } from "../store/ui";
import { useProductsStore } from "../store/products";

export function useAiEvents() {
  const setProgress = useUIStore((s) => s.setAiProgress);
  const setLabel = useUIStore((s) => s.setAiLabel);
  const setJob = useUIStore((s) => s.setAiJob);
  const markDone = useUIStore((s) => s.markAiDone);
  const refetch = useProductsStore((s) => s.refetch);

  useEffect(() => {
    const base = (import.meta as any).env?.VITE_API_BASE_URL ?? "";
    let es: EventSource | undefined;

    try {
      es = new EventSource(`${base}/_ai_fill/stream`, { withCredentials: true } as EventSourceInit);
      es.onmessage = (event) => {
        if (!event.data) return;
        try {
          const payload = JSON.parse(event.data);
          if (payload?.job_id && payload.status === "running") {
            setJob(payload.job_id);
            const pct = Math.round((payload.progress ?? 0) * 100);
            setLabel("IA generando…");
            setProgress(pct);
          } else if (payload?.status === "done") {
            setJob(undefined);
            setProgress(100);
            setLabel("Listo");
            markDone();
            refetch({ force: true });
          }
        } catch {
          // ignore malformed events
        }
      };
    } catch {
      // ignore EventSource errors
    }

    // Polling de estado (monótono y sin resets a 0)
    const poll = window.setInterval(async () => {
      try {
        const res = await fetch(`${base}/api/ai/progress`, { credentials: "include" });
        const j = await res.json();
        if (j?.job_id && j.status === "running") {
          setJob(j.job_id);
          const pct = Math.round((j.progress ?? 0) * 100);
          setLabel("IA generando…");
          setProgress(pct);
        } else if (j?.status === "done") {
          setJob(undefined);
          setProgress(100);
          setLabel("Listo");
          markDone();
          refetch({ force: true });
        }
        // Si está "idle" no tocamos nada para no pisar la barra a 0.
      } catch {
        // ignora errores de red
      }
    }, 2000);

    return () => {
      if (es) {
        es.close();
      }
      window.clearInterval(poll);
    };
  }, [setProgress, setLabel, setJob, markDone, refetch]);
}

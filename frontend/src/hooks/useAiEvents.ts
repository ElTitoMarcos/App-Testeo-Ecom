import { useEffect, useRef } from "react";
import { useUIStore } from "../store/ui";
import { useProductsStore } from "../store/products";

export function useAiEvents() {
  const setProgress = useUIStore((s) => s.setAiProgress);
  const setLabel = useUIStore((s) => s.setAiLabel);
  const markDone = useUIStore((s) => s.markAiDone);
  const resetAi = useUIStore((s) => s.resetAi);
  const refetch = useProductsStore((s) => s.refetch);
  // Flag para saber si SSE está abierto (y entonces desactivar polling)
  const sseOpenRef = useRef(false);

  useEffect(() => {
    const base = (import.meta as any).env?.VITE_API_BASE_URL ?? "";
    const es = new EventSource(`${base}/events/ai`, { withCredentials: true });
    sseOpenRef.current = false;

    const handle = (msg: any) => {
      // Normaliza inicio de run → reset a 0% solo aquí
      if (msg?.type === "ai.progress" && (msg?.message === "starting" || msg?.reset === true)) {
        resetAi();
      }
      switch (msg?.type) {
        case "ai.progress": {
          const pct = Math.round(((msg.progress ?? 0) as number) * 100);
          setProgress(pct);
          setLabel("IA Generando...");
          break;
        }
        case "ai.done": {
          setProgress(100);
          setLabel("Listo");
          markDone();
          refetch({ force: true });
          break;
        }
        case "products.updated":
        case "products.reload": {
          refetch({ force: true });
          break;
        }
        case "ai.error": {
          setLabel("Error");
          break;
        }
      }
    };

    es.onopen = () => { sseOpenRef.current = true; };
    es.onmessage = (ev) => {
      try { handle(JSON.parse(ev.data)); } catch { /* ignore */ }
    };
    es.onerror = () => { sseOpenRef.current = false; /* polling tomará el relevo */ };

    // Polling de respaldo SOLO cuando SSE no está abierto
    const poll = window.setInterval(async () => {
      if (sseOpenRef.current) return;
      try {
        const res = await fetch(`${base}/api/ai/progress`, { credentials: "include" });
        const j = await res.json();
        const pct = Math.round(((j.progress ?? 0) as number) * 100);
        if (j.status === "running") {
          setLabel("IA Generando...");
          setProgress(pct);          // monótono (el store impide bajar)
        } else {
          if (pct >= 100 || j.status === "done") {
            setProgress(100);
            setLabel("Listo");
            markDone();
            refetch({ force: true });
          }
        }
        if (j.reset === true || j.message === "starting") resetAi();
      } catch {
        /* ignore */
      }
    }, 2000);

    return () => { window.clearInterval(poll); es.close(); };
  }, [setProgress, setLabel, markDone, resetAi, refetch]);
}

import { create } from "zustand";

type UIState = {
  aiProgress: number;   // 0..100
  aiLabel: string;      // "IA generando…" | "Listo" | "Error"
  aiJobId?: string;     // identifica el run activo para evitar resets
  setAiProgress: (p: number) => void;
  setAiLabel: (s: string) => void;
  setAiJob: (id?: string) => void;
  markAiDone: () => void;
};

export const useUIStore = create<UIState>((set) => ({
  aiProgress: 0,
  aiLabel: "Listo",
  aiJobId: undefined,
  // No permitir regresión del porcentaje dentro del mismo job
  setAiProgress: (p) =>
    set((s) => {
      const next = Math.max(0, Math.min(100, Math.round(p)));
      if (next < s.aiProgress) return {}; // ignora bajadas
      return { aiProgress: next };
    }),
  setAiLabel: (s) => set({ aiLabel: s }),
  setAiJob: (id) => set({ aiJobId: id, aiProgress: id ? 0 : 100 }),
  markAiDone: () => set({ aiProgress: 100, aiLabel: "Listo" }),
}));

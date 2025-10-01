import create from "zustand";

const clamp = (n: number) => Math.max(0, Math.min(100, Math.round(n)));

type UIState = {
  aiProgress: number;           // 0..100
  aiLabel: string;              // "IA Generando..." | "Listo" | "Error"
  aiRunning: boolean;           // control de monotonicidad
  setAiProgress: (p: number) => void;
  setAiLabel: (s: string) => void;
  markAiDone: () => void;
  resetAi: () => void;          // marcar nuevo run
};

export const useUIStore = create<UIState>((set) => ({
  aiProgress: 0,
  aiLabel: "Listo",
  aiRunning: false,
  // Monótono mientras aiRunning=true (no bajar el %)
  setAiProgress: (p) =>
    set((s) => {
      const next = clamp(p);
      if (s.aiRunning && next < s.aiProgress) return {};
      return { aiProgress: next };
    }),
  setAiLabel: (s) => set({ aiLabel: s }),
  resetAi: () => set({ aiProgress: 0, aiLabel: "IA Generando...", aiRunning: true }),
  markAiDone: () => set({ aiProgress: 100, aiLabel: "Listo", aiRunning: false }),
}));

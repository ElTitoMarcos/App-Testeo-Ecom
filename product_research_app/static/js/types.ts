export type AppConfig = {
  weights: Record<string, number>;
  oldness_preference_pct: number; // 0..100
};

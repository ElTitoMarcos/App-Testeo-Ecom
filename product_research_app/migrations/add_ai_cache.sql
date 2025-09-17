CREATE TABLE IF NOT EXISTS ai_cache (
  task_type TEXT,
  cache_key TEXT,
  payload_json TEXT,
  model_version TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (task_type, cache_key)
);

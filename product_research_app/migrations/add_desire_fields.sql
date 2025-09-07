-- Idempotent migration for desire/awareness/competition fields
ALTER TABLE products ADD COLUMN IF NOT EXISTS desire TEXT;
ALTER TABLE products ADD COLUMN IF NOT EXISTS desire_magnitude TEXT;
ALTER TABLE products ADD COLUMN IF NOT EXISTS awareness_level TEXT;
ALTER TABLE products ADD COLUMN IF NOT EXISTS competition_level TEXT;

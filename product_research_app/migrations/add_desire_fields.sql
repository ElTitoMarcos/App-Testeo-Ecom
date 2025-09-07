-- Migration for new desire/awareness/competition fields
-- The application performs this migration automatically, but the following
-- statements can be executed manually on older databases. Run once.
ALTER TABLE products ADD COLUMN desire_text TEXT;
ALTER TABLE products ADD COLUMN desire_magnitude TEXT;
ALTER TABLE products ADD COLUMN awareness_level TEXT;
ALTER TABLE products ADD COLUMN competition_level TEXT;

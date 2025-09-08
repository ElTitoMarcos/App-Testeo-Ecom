-- Migration for date_range field
-- Rename existing Spanish column if present
ALTER TABLE products RENAME COLUMN rango_fechas TO date_range;
-- Add date_range column if missing
ALTER TABLE products ADD COLUMN date_range TEXT;

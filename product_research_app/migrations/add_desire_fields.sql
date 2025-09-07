-- Migration for desire/awareness/competition fields
-- Rename existing Spanish columns if present
ALTER TABLE products RENAME COLUMN magnitud_deseo TO desire_magnitude;
ALTER TABLE products RENAME COLUMN nivel_consciencia TO awareness_level;
ALTER TABLE products RENAME COLUMN saturacion_mercado TO competition_level;
-- Add new desire column
ALTER TABLE products ADD COLUMN desire TEXT;
-- Drop obsolete columns if they exist
ALTER TABLE products DROP COLUMN facilidad_anuncio;
ALTER TABLE products DROP COLUMN facilidad_logistica;
ALTER TABLE products DROP COLUMN escalabilidad;
ALTER TABLE products DROP COLUMN engagement_shareability;
ALTER TABLE products DROP COLUMN durabilidad_recurrencia;

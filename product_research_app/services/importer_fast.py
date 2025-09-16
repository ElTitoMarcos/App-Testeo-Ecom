import csv, io
from product_research_app.db import get_db

STAGING_SCHEMA = """
CREATE TEMP TABLE IF NOT EXISTS staging_products (
  id INTEGER PRIMARY KEY,
  name TEXT,
  category_path TEXT,
  price REAL,
  rating REAL,
  units_sold REAL,
  revenue REAL,
  conversion_rate REAL,
  launch_date TEXT,
  date_range TEXT,
  desire_magnitude TEXT,
  awareness_level TEXT,
  competition_level TEXT,
  winner_score INTEGER,
  image_url TEXT,
  desire TEXT
);
DELETE FROM staging_products;
"""

UPSERT_SELECT = """
INSERT INTO products (
  id, name, category_path, price, rating, units_sold, revenue,
  conversion_rate, launch_date, date_range, desire_magnitude,
  awareness_level, competition_level, winner_score, image_url, desire
)
SELECT
  id, name, category_path, price, rating, units_sold, revenue,
  conversion_rate, launch_date, date_range, desire_magnitude,
  awareness_level, competition_level, winner_score, image_url, desire
FROM staging_products
ON CONFLICT(id) DO UPDATE SET
  name=excluded.name,
  category_path=excluded.category_path,
  price=excluded.price,
  rating=excluded.rating,
  units_sold=excluded.units_sold,
  revenue=excluded.revenue,
  conversion_rate=excluded.conversion_rate,
  launch_date=excluded.launch_date,
  date_range=excluded.date_range,
  desire_magnitude=excluded.desire_magnitude,
  awareness_level=excluded.awareness_level,
  competition_level=excluded.competition_level,
  winner_score=COALESCE(excluded.winner_score, products.winner_score),
  image_url=excluded.image_url,
  desire=COALESCE(excluded.desire, products.desire);
"""

BATCH_SIZE = 5000

def _num(x):
  if x is None: return 0.0
  s = str(x).strip()
  mul = 1
  if s.lower().endswith('m'): mul, s = 1e6, s[:-1]
  if s.lower().endswith('k'): mul, s = 1e3, s[:-1]
  s = s.replace('€','').replace('$','').replace('%','').replace('.','').replace(',','.')
  try: return float(s) * mul
  except: return 0.0

def _int_or_default(x, default=0):
  if x in (None, ''): return default
  try: return int(_num(x)) or default
  except: return default

def _rows_from_csv(csv_bytes):
  txt = csv_bytes.decode('utf-8', errors='ignore')
  rdr = csv.DictReader(io.StringIO(txt))
  for r in rdr:
    yield (
      int(r.get('id') or r.get('ID') or 0),
      r.get('name') or r.get('Nombre') or '',
      r.get('category_path') or r.get('Categoría') or r.get('categoria') or '',
      _num(r.get('price')),
      _num(r.get('rating')),
      _num(r.get('units_sold') or r.get('unidades')),
      _num(r.get('revenue') or r.get('ingresos')),
      _num(r.get('conversion_rate') or r.get('tasa_conversion')),
      (r.get('launch_date') or r.get('fecha_lanzamiento') or '')[:10],
      r.get('date_range') or r.get('rango_fechas') or '',
      r.get('desire_magnitude') or r.get('desireMag') or '',
      r.get('awareness_level') or r.get('awareness') or '',
      r.get('competition_level') or r.get('competition') or '',
      None if (r.get('winner_score') in (None,'')) else int(_num(r.get('winner_score'))),
      r.get('image_url') or r.get('imagen') or '',
      r.get('desire') or ''
    )

def _rows_from_records(records):
  for r in records:
    if not isinstance(r, dict):
      continue
    yield (
      _int_or_default(r.get('id') or r.get('ID')),
      r.get('name') or r.get('Nombre') or '',
      r.get('category_path') or r.get('Categoría') or r.get('categoria') or r.get('category') or '',
      _num(r.get('price') or r.get('precio')),
      _num(r.get('rating') or r.get('valoracion')),
      _num(r.get('units_sold') or r.get('unidades') or r.get('units')),
      _num(r.get('revenue') or r.get('ingresos') or r.get('sales')),
      _num(r.get('conversion_rate') or r.get('tasa_conversion') or r.get('conversion')),
      (r.get('launch_date') or r.get('fecha_lanzamiento') or r.get('launchDate') or '')[:10],
      r.get('date_range') or r.get('rango_fechas') or r.get('Date Range') or '',
      r.get('desire_magnitude') or r.get('desireMag') or '',
      r.get('awareness_level') or r.get('awareness') or '',
      r.get('competition_level') or r.get('competition') or '',
      None if (r.get('winner_score') in (None,'')) else int(_num(r.get('winner_score'))),
      r.get('image_url') or r.get('imagen') or r.get('image') or '',
      r.get('desire') or ''
    )

def _snapshot_and_drop(db, table='products'):
  idx = db.execute(
    "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL;",
    (table,)
  ).fetchall()
  trg = db.execute(
    "SELECT name, sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?;",
    (table,)
  ).fetchall()
  for (name, _) in idx: db.execute(f'DROP INDEX IF EXISTS "{name}";')
  for (name, _) in trg: db.execute(f'DROP TRIGGER IF EXISTS "{name}";')
  return idx, trg

def _recreate(db, items):
  for (name, sql) in items:
    if sql: db.execute(sql)

def _bulk_import(row_iter, status_cb):
  db = get_db()

  # PRAGMAs (solo durante import)
  db.execute("PRAGMA journal_mode=WAL;")
  db.execute("PRAGMA synchronous=OFF;")
  db.execute("PRAGMA temp_store=MEMORY;")
  db.execute("PRAGMA cache_size=-50000;")      # ~50MB de cache
  db.execute("PRAGMA locking_mode=EXCLUSIVE;")
  db.execute("PRAGMA foreign_keys=OFF;")

  db.execute("BEGIN IMMEDIATE;")
  total = 0
  try:
    # 1) staging en memoria
    db.executescript(STAGING_SCHEMA)

    # 2) ingestión en grandes lotes (sin retener todo en RAM)
    insert_staging = (
      "INSERT INTO staging_products "
      "(id,name,category_path,price,rating,units_sold,revenue,conversion_rate," \
      "launch_date,date_range,desire_magnitude,awareness_level,competition_level," \
      "winner_score,image_url,desire) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"
    )
    batch = []; done = 0
    for row in row_iter:
      batch.append(row); total += 1
      if len(batch) >= BATCH_SIZE:
        db.executemany(insert_staging, batch); done += len(batch); batch.clear()
        status_cb(stage="staging", done=done, total=max(total, done))
    if batch:
      db.executemany(insert_staging, batch); done += len(batch); batch.clear()
      status_cb(stage="staging", done=done, total=max(total, done))

    # 3) desactivar índices/triggers de destino y upsert masivo
    idx, trg = _snapshot_and_drop(db, 'products')
    db.execute("SAVEPOINT upsert_bulk;")
    db.execute(UPSERT_SELECT)
    db.execute("RELEASE upsert_bulk;")

    # 4) recrear índices/triggers y optimizar stats
    _recreate(db, idx); _recreate(db, trg)
    db.execute("ANALYZE products;")

    db.execute("COMMIT;")
    status_cb(stage="done", done=total, total=total)
    return total
  except Exception:
    db.execute("ROLLBACK;")
    raise
  finally:
    # volver a modo seguro
    db.execute("PRAGMA synchronous=NORMAL;")
    db.execute("PRAGMA foreign_keys=ON;")
    db.execute("PRAGMA locking_mode=NORMAL;")

def fast_import(csv_bytes, status_cb=lambda **k: None, source=None):
  """
  Import rápido para grandes volúmenes:
  - staging TEMP en memoria
  - PRAGMAs agresivos durante la ventana de import
  - desactivar índices/triggers del destino
  - UPSERT masivo SELECT->products
  """
  return _bulk_import(_rows_from_csv(csv_bytes), status_cb)

def fast_import_records(records, status_cb=lambda **k: None, source=None):
  return _bulk_import(_rows_from_records(records), status_cb)

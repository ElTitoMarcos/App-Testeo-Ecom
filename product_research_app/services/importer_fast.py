from __future__ import annotations

import csv
import io
import time
from typing import Iterable, Iterator, Sequence

from product_research_app.db import get_db


def _count_lines(b: bytes) -> int:
    """Return a quick line count estimate for CSV payloads."""

    if not b:
        return 0
    return max(b.count(b"\n") - 1, 0)


def _num(x):
    if x is None:
        return 0.0
    s = str(x).strip()
    mul = 1
    if s.lower().endswith("m"):
        mul, s = 1e6, s[:-1]
    if s.lower().endswith("k"):
        mul, s = 1e3, s[:-1]
    s = (
        s.replace("€", "")
        .replace("$", "")
        .replace("%", "")
        .replace(".", "")
        .replace(",", ".")
    )
    try:
        return float(s) * mul
    except Exception:
        return 0.0


def _rows_from_csv(csv_bytes: bytes) -> Iterator[tuple]:
    txt = csv_bytes.decode("utf-8", errors="ignore")
    rdr = csv.DictReader(io.StringIO(txt))
    for r in rdr:
        yield (
            int(r.get("id") or r.get("ID") or 0),
            r.get("name") or r.get("Nombre") or "",
            r.get("category_path")
            or r.get("Categoría")
            or r.get("categoria")
            or "",
            _num(r.get("price")),
            _num(r.get("rating")),
            _num(r.get("units_sold") or r.get("unidades")),
            _num(r.get("revenue") or r.get("ingresos")),
            _num(r.get("conversion_rate") or r.get("tasa_conversion")),
            (r.get("launch_date") or r.get("fecha_lanzamiento") or "")[:10],
            r.get("date_range") or r.get("rango_fechas") or "",
            r.get("desire_magnitude") or r.get("desireMag") or "",
            r.get("awareness_level") or r.get("awareness") or "",
            r.get("competition_level") or r.get("competition") or "",
            None
            if (r.get("winner_score") in (None, ""))
            else int(_num(r.get("winner_score"))),
            r.get("image_url") or r.get("imagen") or "",
            r.get("desire") or "",
        )


def _rows_from_records(records: Iterable[dict]) -> Iterator[tuple]:
    for r in records:
        if not isinstance(r, dict):
            continue
        yield (
            int(_num(r.get("id") or r.get("ID"))),
            r.get("name") or r.get("Nombre") or "",
            r.get("category_path")
            or r.get("Categoría")
            or r.get("categoria")
            or r.get("category")
            or "",
            _num(r.get("price") or r.get("precio")),
            _num(r.get("rating") or r.get("valoracion")),
            _num(r.get("units_sold") or r.get("unidades") or r.get("units")),
            _num(r.get("revenue") or r.get("ingresos") or r.get("sales")),
            _num(
                r.get("conversion_rate")
                or r.get("tasa_conversion")
                or r.get("conversion")
            ),
            (r.get("launch_date") or r.get("fecha_lanzamiento") or r.get("launchDate") or "")[
                :10
            ],
            r.get("date_range")
            or r.get("rango_fechas")
            or r.get("Date Range")
            or "",
            r.get("desire_magnitude") or r.get("desireMag") or "",
            r.get("awareness_level") or r.get("awareness") or "",
            r.get("competition_level") or r.get("competition") or "",
            None
            if (r.get("winner_score") in (None, ""))
            else int(_num(r.get("winner_score"))),
            r.get("image_url") or r.get("imagen") or r.get("image") or "",
            r.get("desire") or "",
        )


def _snapshot_and_drop(db, table: str = "products") -> tuple[Sequence[tuple], Sequence[tuple]]:
    idx = db.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='index' AND tbl_name=? AND sql IS NOT NULL;",
        (table,),
    ).fetchall()
    trg = db.execute(
        "SELECT name, sql FROM sqlite_master WHERE type='trigger' AND tbl_name=?;",
        (table,),
    ).fetchall()
    for (name, _) in idx:
        db.execute(f'DROP INDEX IF EXISTS "{name}";')
    for (name, _) in trg:
        db.execute(f'DROP TRIGGER IF EXISTS "{name}";')
    return idx, trg


def _recreate(db, items: Sequence[tuple]) -> None:
    for (name, sql) in items or ():
        if sql:
            db.execute(sql)


def _round_ms(delta: float) -> int:
    return max(int(round(delta * 1000)), 0)


def _push_pragmas(db):
    original = {}
    try:
        original["journal_mode"] = db.execute("PRAGMA journal_mode;").fetchone()[0]
    except Exception:
        original["journal_mode"] = None
    try:
        original["synchronous"] = db.execute("PRAGMA synchronous;").fetchone()[0]
    except Exception:
        original["synchronous"] = None
    try:
        original["temp_store"] = db.execute("PRAGMA temp_store;").fetchone()[0]
    except Exception:
        original["temp_store"] = None
    try:
        original["cache_size"] = db.execute("PRAGMA cache_size;").fetchone()[0]
    except Exception:
        original["cache_size"] = None
    try:
        original["locking_mode"] = db.execute("PRAGMA locking_mode;").fetchone()[0]
    except Exception:
        original["locking_mode"] = None
    try:
        original["foreign_keys"] = db.execute("PRAGMA foreign_keys;").fetchone()[0]
    except Exception:
        original["foreign_keys"] = None
    try:
        original["busy_timeout"] = db.execute("PRAGMA busy_timeout;").fetchone()[0]
    except Exception:
        original["busy_timeout"] = None
    try:
        original["mmap_size"] = db.execute("PRAGMA mmap_size;").fetchone()[0]
    except Exception:
        original["mmap_size"] = None

    db.execute("PRAGMA journal_mode=WAL;")
    db.execute("PRAGMA synchronous=OFF;")
    db.execute("PRAGMA temp_store=MEMORY;")
    db.execute("PRAGMA cache_size=-60000;")
    db.execute("PRAGMA locking_mode=EXCLUSIVE;")
    db.execute("PRAGMA foreign_keys=OFF;")
    db.execute("PRAGMA busy_timeout=2000;")
    db.execute("PRAGMA mmap_size=268435456;")

    return original


def _restore_pragmas(db, original) -> None:
    if not original:
        return
    try:
        jm = original.get("journal_mode")
        if jm:
            db.execute(f"PRAGMA journal_mode={jm};")
    except Exception:
        pass
    try:
        sync = original.get("synchronous")
        if sync is not None:
            db.execute(f"PRAGMA synchronous={sync};")
    except Exception:
        pass
    try:
        temp_store = original.get("temp_store")
        if temp_store is not None:
            db.execute(f"PRAGMA temp_store={temp_store};")
    except Exception:
        pass
    try:
        cache_size = original.get("cache_size")
        if cache_size is not None:
            db.execute(f"PRAGMA cache_size={cache_size};")
    except Exception:
        pass
    try:
        locking_mode = original.get("locking_mode")
        if locking_mode:
            db.execute(f"PRAGMA locking_mode={locking_mode};")
    except Exception:
        pass
    try:
        fk = original.get("foreign_keys")
        if fk is not None:
            db.execute(f"PRAGMA foreign_keys={'ON' if fk else 'OFF'};")
    except Exception:
        pass
    try:
        busy = original.get("busy_timeout")
        if busy is not None:
            db.execute(f"PRAGMA busy_timeout={int(busy)};")
    except Exception:
        pass
    try:
        mmap = original.get("mmap_size")
        if mmap is not None:
            db.execute(f"PRAGMA mmap_size={int(mmap)};")
    except Exception:
        pass


STAGING_SCHEMA = """
CREATE TEMP TABLE IF NOT EXISTS staging_products (
  id INTEGER PRIMARY KEY,
  name TEXT, category_path TEXT, price REAL, rating REAL,
  units_sold REAL, revenue REAL, conversion_rate REAL,
  launch_date TEXT, date_range TEXT,
  desire_magnitude TEXT, awareness_level TEXT, competition_level TEXT,
  winner_score INTEGER, image_url TEXT, desire TEXT
);
DELETE FROM staging_products;
"""


UPSERT_DIRECT = """
INSERT INTO products (
  id, name, category_path, price, rating, units_sold, revenue,
  conversion_rate, launch_date, date_range, desire_magnitude,
  awareness_level, competition_level, winner_score, image_url, desire
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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


UPSERT_FROM_STAGING = """
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


_BATCH_SIZE = 8000


def _should_use_staging(n_rows_est: int, current_rows: int) -> bool:
    if n_rows_est <= 0:
        return False
    if n_rows_est >= 1000:
        return True
    if n_rows_est < 200:
        return False
    base = max(current_rows, 1)
    relative = n_rows_est / base
    return relative > 0.03


def _import_rows(
    db,
    rows: Iterator[tuple],
    total_est: int,
    status_cb,
    use_staging: bool,
    t0: float,
):
    total_hint = int(total_est or 0)
    if total_hint < 0:
        total_hint = 0
    done = 0
    actual_total = total_hint

    def update_progress(stage: str | None = None, final: bool = False) -> None:
        nonlocal actual_total
        actual_total = max(actual_total, done)
        payload = {"done": done, "total": actual_total}
        if stage is not None:
            payload["stage"] = stage
        if final:
            payload["imported"] = done
        status_cb(**payload)

    t_parse = time.time()
    db.execute("BEGIN IMMEDIATE;")
    try:
        if use_staging:
            db.executescript(STAGING_SCHEMA)
            insert_staging = (
                "INSERT INTO staging_products "
                "(id,name,category_path,price,rating,units_sold,revenue,conversion_rate,"
                "launch_date,date_range,desire_magnitude,awareness_level,competition_level,"
                "winner_score,image_url,desire) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?);"
            )
            batch = []
            for row in rows:
                batch.append(row)
                if len(batch) >= _BATCH_SIZE:
                    db.executemany(insert_staging, batch)
                    done += len(batch)
                    batch.clear()
                    update_progress("staging")
            if batch:
                db.executemany(insert_staging, batch)
                done += len(batch)
                batch.clear()
                update_progress("staging")

            idx, trg = _snapshot_and_drop(db, "products")
            t_staging = time.time()

            db.execute("SAVEPOINT upsert_bulk;")
            db.execute(UPSERT_FROM_STAGING)
            db.execute("RELEASE upsert_bulk;")
            t_upsert = time.time()
            update_progress("upsert")

            db.execute("COMMIT;")
            t_commit = time.time()
            update_progress("done", final=True)

            status_cb(
                t_parse=_round_ms(t_parse - t0),
                t_staging=_round_ms(t_staging - t_parse),
                t_upsert=_round_ms(t_upsert - t_staging),
                t_commit=_round_ms(t_commit - t_upsert),
            )

            idx = list(idx or [])
            trg = list(trg or [])
            already = False

            def optimize():
                nonlocal already
                if already:
                    return
                already = True
                if idx or trg:
                    db.execute("BEGIN;")
                    try:
                        _recreate(db, idx)
                        _recreate(db, trg)
                        db.execute("COMMIT;")
                    except Exception:
                        db.execute("ROLLBACK;")
                        raise
                db.execute("ANALYZE products;")

            optimize.rows_imported = done
            optimize.use_staging = True
            return optimize

        cur = db.cursor()
        try:
            batch = []
            for row in rows:
                batch.append(row)
                if len(batch) >= _BATCH_SIZE:
                    cur.executemany(UPSERT_DIRECT, batch)
                    done += len(batch)
                    batch.clear()
                    update_progress("upsert")
            if batch:
                cur.executemany(UPSERT_DIRECT, batch)
                done += len(batch)
                batch.clear()
                update_progress("upsert")
        finally:
            cur.close()

        t_upsert = time.time()
        db.execute("COMMIT;")
        t_commit = time.time()
        update_progress("done", final=True)

        status_cb(
            t_parse=_round_ms(t_parse - t0),
            t_staging=0,
            t_upsert=_round_ms(t_upsert - t_parse),
            t_commit=_round_ms(t_commit - t_upsert),
        )

        already = False

        def optimize():
            nonlocal already
            if already:
                return
            already = True

        optimize.rows_imported = done
        optimize.use_staging = False
        return optimize

    except Exception:
        db.execute("ROLLBACK;")
        raise


def fast_import_adaptive(csv_bytes: bytes, status_cb=lambda **k: None):
    """Import CSV data using an adaptive strategy.

    Returns a callable ``optimize()`` that performs heavy post-processing
    (index/trigger recreation and ANALYZE). Callers may execute the returned
    callable in a background thread to keep the UI responsive.
    """

    if not isinstance(csv_bytes, (bytes, bytearray)):
        csv_bytes = bytes(csv_bytes)

    db = get_db()
    t0 = time.time()
    n_rows_est = _count_lines(csv_bytes)
    status_cb(total=n_rows_est)

    try:
        cur_count = db.execute("SELECT COUNT(*) FROM products;").fetchone()[0]
    except Exception:
        cur_count = 0

    use_staging = _should_use_staging(n_rows_est, cur_count)
    pragmas = _push_pragmas(db)
    optimize = None
    try:
        optimize = _import_rows(
            db,
            _rows_from_csv(csv_bytes),
            n_rows_est,
            status_cb,
            use_staging,
            t0,
        )
    finally:
        _restore_pragmas(db, pragmas)

    if optimize is None:
        def _noop():
            return None

        _noop.rows_imported = 0
        _noop.use_staging = False
        optimize = _noop

    return optimize


def fast_import(csv_bytes, status_cb=lambda **k: None, source=None):
    optimize = fast_import_adaptive(csv_bytes, status_cb=status_cb)
    try:
        optimize()
    except Exception:
        raise
    return int(getattr(optimize, "rows_imported", 0) or 0)


def fast_import_records(records, status_cb=lambda **k: None, source=None):
    if not isinstance(records, Sequence):
        records = list(records)
    total = len(records)
    status_cb(total=total)

    db = get_db()
    t0 = time.time()
    try:
        cur_count = db.execute("SELECT COUNT(*) FROM products;").fetchone()[0]
    except Exception:
        cur_count = 0

    use_staging = _should_use_staging(total, cur_count)
    pragmas = _push_pragmas(db)
    optimize = None
    try:
        optimize = _import_rows(
            db,
            _rows_from_records(records),
            total,
            status_cb,
            use_staging,
            t0,
        )
    finally:
        _restore_pragmas(db, pragmas)

    if optimize is None:
        def _noop():
            return None

        _noop.rows_imported = 0
        _noop.use_staging = False
        optimize = _noop

    optimize()
    return int(getattr(optimize, "rows_imported", 0) or 0)

import sqlite3


def row_to_dict(row):
    if row is None:
        return None
    if isinstance(row, dict):
        return row
    if isinstance(row, sqlite3.Row):
        return {k: row[k] for k in row.keys()}
    try:
        return dict(row)
    except Exception:
        return None


def rget(row, key, default=None):
    if row is None:
        return default
    if isinstance(row, dict):
        return row.get(key, default)
    if isinstance(row, sqlite3.Row):
        return row[key] if key in row.keys() else default
    try:
        d = dict(row)
        return d.get(key, default)
    except Exception:
        return default

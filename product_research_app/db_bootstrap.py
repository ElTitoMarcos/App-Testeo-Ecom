"""SQLite schema bootstrap and seeding utilities for the dev pipeline."""

from __future__ import annotations

import logging
import random
import sqlite3
from typing import Iterable

logger = logging.getLogger(__name__)


_PRODUCT_SQL = """
CREATE TABLE IF NOT EXISTS product (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  title TEXT NOT NULL,
  name TEXT,
  brand TEXT,
  category TEXT,
  description TEXT,
  price REAL,
  units_sold INTEGER,
  rating REAL,
  oldness REAL,
  revenue REAL,
  parent_id INTEGER,
  awareness TEXT,
  awareness_magnitude INTEGER,
  desire TEXT,
  desire_magnitude INTEGER,
  competition TEXT,
  competition_magnitude INTEGER,
  awareness_level TEXT,
  competition_level TEXT,
  ai_desire TEXT,
  ai_desire_label TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  updated_at TEXT DEFAULT CURRENT_TIMESTAMP
);
""".strip()

_EXTRAS_SQL = """
CREATE TABLE IF NOT EXISTS extras (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  product_id INTEGER UNIQUE REFERENCES product(id) ON DELETE CASCADE,
  desire TEXT,
  desire_magnitude INTEGER,
  awareness TEXT,
  awareness_magnitude INTEGER,
  competition TEXT,
  competition_magnitude INTEGER
);
""".strip()


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cursor.fetchone() is not None


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Ensure the minimal schema required for the dev pipeline exists."""

    tables_sql: dict[str, str] = {"product": _PRODUCT_SQL, "extras": _EXTRAS_SQL}
    created: list[str] = []

    for table_name, create_sql in tables_sql.items():
        if not _table_exists(conn, table_name):
            conn.execute(create_sql)
            created.append(table_name)
        else:
            conn.execute(create_sql)

    if created:
        logger.info("db_bootstrap: created tables: %s", ", ".join(created))


def drop_all(conn: sqlite3.Connection) -> None:
    """Drop all known tables created by :func:`ensure_schema`."""

    dropped: list[str] = []
    for table_name in ("extras", "product"):
        if _table_exists(conn, table_name):
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            dropped.append(table_name)
    if dropped:
        logger.info("db_bootstrap: dropped tables: %s", ", ".join(dropped))


def _generate_fake_titles(n: int) -> Iterable[str]:
    adjectives = [
        "Smart",
        "Eco",
        "Ultra",
        "Compact",
        "Portable",
        "Premium",
        "Wireless",
        "Turbo",
        "Pro",
        "Essential",
    ]
    nouns = [
        "Blender",
        "Speaker",
        "Notebook",
        "Bottle",
        "Vacuum",
        "Lamp",
        "Charger",
        "Backpack",
        "Watch",
        "Headphones",
    ]
    suffixes = [
        "Max",
        "Lite",
        "Plus",
        "X",
        "Edge",
        "Air",
        "Flex",
        "Core",
        "Prime",
        "Nova",
    ]

    for index in range(n):
        title = " ".join(
            (
                random.choice(adjectives),
                random.choice(nouns),
                random.choice(suffixes),
                str(100 + index),
            )
        )
        yield title


def seed_fake_products(conn: sqlite3.Connection, n: int) -> int:
    """Seed ``n`` synthetic products for development and testing."""

    if n <= 0:
        logger.info("db_bootstrap: seed requested with non-positive count (%s)", n)
        return 0

    ensure_schema(conn)

    brands = [
        "Acme",
        "Globex",
        "Soylent",
        "Initech",
        "Umbra",
        "Stark",
        "Wayne",
        "Wonka",
    ]
    categories = [
        "Home",
        "Electronics",
        "Outdoors",
        "Fitness",
        "Kitchen",
        "Office",
        "Travel",
    ]

    rows = []
    for title in _generate_fake_titles(n):
        price = round(random.uniform(9.99, 249.99), 2)
        units_sold = random.randint(25, 5000)
        rating = round(random.uniform(3.0, 5.0), 2)
        oldness = round(random.uniform(0.0, 1.0), 3)
        revenue = round(price * units_sold, 2)
        brand = random.choice(brands)
        category = random.choice(categories)
        description = (
            f"{title} by {brand} combines modern design with practical features, ideal for {category.lower()} use."
        )
        rows.append(
            (
                title,
                title,
                brand,
                category,
                description,
                price,
                units_sold,
                rating,
                oldness,
                revenue,
                None,
            )
        )

    conn.executemany(
        """
        INSERT INTO product (
            title,
            name,
            brand,
            category,
            description,
            price,
            units_sold,
            rating,
            oldness,
            revenue,
            parent_id,
            awareness,
            awareness_magnitude,
            desire,
            desire_magnitude,
            competition,
            competition_magnitude,
            ai_desire,
            ai_desire_label
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL)
        """.strip(),
        rows,
    )

    logger.info("db_bootstrap: seeded %s products", len(rows))
    return len(rows)

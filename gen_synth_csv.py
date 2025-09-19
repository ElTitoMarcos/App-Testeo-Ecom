#!/usr/bin/env python3
"""Generate synthetic catalog CSV files for stress and benchmark runs."""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path


CATEGORIES = [
    "home",
    "beauty",
    "fitness",
    "kitchen",
    "electronics",
    "outdoors",
    "pets",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Destination CSV file")
    parser.add_argument("--rows", type=int, default=10_000, help="Number of rows to generate")
    parser.add_argument(
        "--clusters",
        type=int,
        default=120,
        help="Number of base product clusters to create",
    )
    parser.add_argument(
        "--redundancy",
        type=float,
        default=0.4,
        help="Fraction of rows that are near-duplicates (0-1)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _base_products(clusters: int) -> list[dict[str, str]]:
    products: list[dict[str, str]] = []
    for idx in range(clusters):
        category = random.choice(CATEGORIES)
        brand = f"Brand{idx % 50:02d}"
        base_title = f"{category.title()} Gadget {idx:04d}"
        base_desc = (
            f"Versatile {category} product {idx} ideal for growth experiments and rapid tests."
        )
        base_price = round(random.uniform(9.0, 120.0), 2)
        base_rating = round(random.uniform(3.2, 4.9), 2)
        base_units = random.randint(20, 2500)
        products.append(
            {
                "title": base_title,
                "description": base_desc,
                "category": category,
                "brand": brand,
                "base_price": base_price,
                "base_rating": base_rating,
                "base_units": base_units,
            }
        )
    return products


def _render_row(idx: int, base: dict[str, str], duplicate: bool) -> dict[str, str]:
    price_variation = random.uniform(0.9, 1.1) if not duplicate else random.uniform(0.98, 1.02)
    rating_variation = random.uniform(-0.25, 0.25) if not duplicate else random.uniform(-0.05, 0.05)
    units_variation = random.randint(-80, 120) if not duplicate else random.randint(-15, 25)
    base_price = float(base["base_price"])
    base_rating = float(base["base_rating"])
    base_units = int(base["base_units"])
    final_price = max(1.0, round(base_price * price_variation, 2))
    final_rating = min(5.0, max(1.0, round(base_rating + rating_variation, 2)))
    final_units = max(0, base_units + units_variation)
    launch_delta = random.randint(0, 720 if not duplicate else 180)
    launch_date = date.today() - timedelta(days=launch_delta)
    asin = f"SYNTH{idx:08d}"
    variation_tag = "Pro" if idx % 5 == 0 else "Lite"
    title = base["title"] if duplicate else f"{base['title']} {variation_tag}"
    description = (
        base["description"]
        if duplicate
        else base["description"] + f" Variant {variation_tag} tailored for rapid launch."
    )
    return {
        "title": title,
        "description": description,
        "category": base["category"],
        "brand": base["brand"],
        "price": f"{final_price:.2f}",
        "rating": f"{final_rating:.2f}",
        "units_sold": str(final_units),
        "launch_date": launch_date.isoformat(),
        "asin": asin,
        "url": f"https://example.test/products/{asin.lower()}",
    }


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    base_products = _base_products(max(1, args.clusters))
    redundancy_cutoff = max(0.0, min(1.0, args.redundancy))
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "title",
            "description",
            "category",
            "brand",
            "price",
            "rating",
            "units_sold",
            "launch_date",
            "asin",
            "url",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx in range(args.rows):
            base = random.choice(base_products)
            duplicate = random.random() < redundancy_cutoff
            row = _render_row(idx, base, duplicate)
            writer.writerow(row)
    print(f"Generated {args.rows} rows at {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

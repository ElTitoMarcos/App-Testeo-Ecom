"""
Command line interface for the Product Research Copilot.

This script ties together all subsystems (database, OpenAI integration, scraping
and configuration) and presents a simple menu driven interface to the user.  It
allows importing product data from CSV/JSON files, adding products manually or
by scraping a URL, listing and viewing products, evaluating them with GPT via
the OpenAI API, managing lists, exporting data and generating simple PDF
reports.

Run this script with Python 3.11 or later.  To install dependencies you may
need ``pip install requests beautifulsoup4 pillow`` on your target system.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

from . import config
from . import database
from . import gpt
from . import scraper
from .utils.db import row_to_dict, rget

import sqlite3


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("gpt.api").setLevel(logging.INFO)
logging.getLogger("gpt.ratelimit").setLevel(logging.INFO)


APP_DIR = Path(__file__).resolve().parent
DB_PATH = APP_DIR / "data.sqlite3"

WINNER_SCORE_FIELDS = [
    "magnitud_deseo",
    "nivel_consciencia",
    "saturacion_mercado",
    "facilidad_anuncio",
    "facilidad_logistica",
    "escalabilidad",
    "engagement_shareability",
    "durabilidad_recurrencia",
]


def ensure_database() -> sqlite3.Connection:
    """Create or open the database and ensure tables exist."""
    conn = database.get_connection(DB_PATH)
    database.initialize_database(conn)
    return conn


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def prompt_user(prompt: str) -> str:
    try:
        return input(prompt)
    except EOFError:
        return ""


def import_data(conn: database.sqlite3.Connection) -> None:
    path = prompt_user("Ruta del archivo CSV/JSON a importar: ").strip()
    if not path:
        print("Ruta vacía, cancelando.")
        return
    file_path = Path(path).expanduser()
    if not file_path.exists():
        print(f"El archivo {file_path} no existe.")
        return
    source = prompt_user("Nombre de la fuente (ej. PiPiADS, Dropispy, Manual): ").strip() or None
    inserted = 0
    try:
        if file_path.suffix.lower() == ".csv":
            with open(file_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                headers = reader.fieldnames or []
                print(f"Columnas detectadas: {', '.join(headers)}")
                # Heuristic mapping
                def find_col(names: List[str], options: List[str]) -> Optional[str]:
                    for opt in options:
                        for n in names:
                            if opt.lower() in n.lower():
                                return n
                    return None

                name_col = find_col(headers, ["name", "nombre", "product", "title"])
                desc_col = find_col(headers, ["description", "descripcion", "desc"])
                cat_col = find_col(headers, ["category", "categoria"])
                price_col = find_col(headers, ["price", "precio", "cost"])
                currency_col = find_col(headers, ["currency", "moneda"])
                image_col = find_col(headers, ["image", "imagen", "img", "picture"])
                date_range_col = find_col(headers, ["date range", "daterange", "fecha rango", "rango fechas"])
                desire_col = find_col(headers, ["desire"])
                desire_mag_col = find_col(headers, ["desire_magnitude", "desire magnitude", "magnitud_deseo"])
                awareness_col = find_col(headers, ["awareness_level", "awareness level", "nivel_consciencia"])
                competition_col = find_col(headers, ["competition_level", "competition level", "saturacion_mercado"])
                # allow user override
                override = prompt_user(
                    "¿Desea mapear columnas manualmente? (s/n): "
                ).strip().lower()
                if override == "s":
                    name_col = prompt_user("Columna para el nombre del producto (enter para inferido): ") or name_col
                    desc_col = prompt_user("Columna para la descripción (enter para inferido): ") or desc_col
                    cat_col = prompt_user("Columna para la categoría (enter para inferido): ") or cat_col
                    price_col = prompt_user("Columna para el precio (enter para inferido): ") or price_col
                    currency_col = prompt_user("Columna para la moneda (enter para inferido): ") or currency_col
                    image_col = prompt_user("Columna para la imagen (enter para inferido): ") or image_col
                    date_range_col = prompt_user("Columna para Date Range (enter para inferido): ") or date_range_col
                    desire_col = prompt_user("Columna para Desire (enter para inferido): ") or desire_col
                    desire_mag_col = (
                        prompt_user("Columna para Desire Magnitude (enter para inferido): ")
                        or desire_mag_col
                    )
                    awareness_col = (
                        prompt_user("Columna para Awareness Level (enter para inferido): ")
                        or awareness_col
                    )
                    competition_col = (
                        prompt_user("Columna para Competition Level (enter para inferido): ")
                        or competition_col
                    )
                for row in reader:
                    name = (row.get(name_col) or "").strip() if name_col else None
                    if not name:
                        # Skip rows without a name
                        continue
                    description = (row.get(desc_col) or "").strip() if desc_col else None
                    category = (row.get(cat_col) or "").strip() if cat_col else None
                    price = None
                    if price_col and row.get(price_col):
                        try:
                            price = float(str(row.get(price_col)).replace(",", "."))
                        except ValueError:
                            price = None
                    currency = (row.get(currency_col) or "").strip() if currency_col else None
                    image_url = (row.get(image_col) or "").strip() if image_col else None
                    date_range = (row.get(date_range_col) or "").strip() if date_range_col else ""
                    desire = (row.get(desire_col) or "").strip() if desire_col else None
                    desire = desire or None
                    desire_mag = (row.get(desire_mag_col) or "").strip() if desire_mag_col else None
                    desire_mag = desire_mag or None
                    awareness = (row.get(awareness_col) or "").strip() if awareness_col else None
                    awareness = awareness or None
                    competition = (row.get(competition_col) or "").strip() if competition_col else None
                    competition = competition or None
                    extra_cols = {
                        k: v
                        for k, v in row.items()
                        if k
                        not in {
                            name_col,
                            desc_col,
                            cat_col,
                            price_col,
                            currency_col,
                            image_col,
                            desire_col,
                            desire_mag_col,
                            awareness_col,
                            competition_col,
                            date_range_col,
                        }
                        and k.lower() not in {"product name", "source", "decision"}
                    }
                    _ = database.insert_product(
                        conn,
                        name=name,
                        description=description,
                        category=category,
                        price=price,
                        currency=currency,
                        image_url=image_url,
                        date_range=date_range,
                        source=source,
                        desire=desire,
                        desire_magnitude=desire_mag,
                        awareness_level=awareness,
                        competition_level=competition,
                        extra=extra_cols,
                    )
                    inserted += 1
        elif file_path.suffix.lower() == ".json":
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                print("El JSON debe contener una lista de objetos.")
                return
            for item in data:
                if not isinstance(item, dict):
                    continue
                name = item.get("name") or item.get("nombre") or item.get("product") or item.get("title")
                if not name:
                    continue
                description = item.get("description") or item.get("descripcion")
                category = item.get("category") or item.get("categoria")
                price = None
                for key in ["price", "precio", "cost"]:
                    if key in item:
                        try:
                            price = float(str(item[key]).replace(",", "."))
                            break
                        except ValueError:
                            continue
                currency = item.get("currency") or item.get("moneda")
                image_url = item.get("image") or item.get("imagen") or item.get("img")
                date_range = (
                    item.get("date_range")
                    or item.get("Date Range")
                    or item.get("DateRange")
                    or item.get("Fecha Rango")
                    or item.get("Rango Fechas")
                    or ""
                )
                desire = item.get("desire") or item.get("desire_text")
                desire_mag = item.get("desire_magnitude") or item.get("magnitud_deseo")
                awareness = item.get("awareness_level") or item.get("nivel_consciencia")
                competition = item.get("competition_level") or item.get("saturacion_mercado")
                extra = {
                    k: v
                    for k, v in item.items()
                    if k
                    not in {
                        "name",
                        "nombre",
                        "product",
                        "title",
                        "description",
                        "descripcion",
                        "category",
                        "categoria",
                        "price",
                        "precio",
                        "cost",
                        "currency",
                        "moneda",
                        "image",
                        "imagen",
                        "img",
                        "desire",
                        "desire_text",
                        "desire_magnitude",
                        "awareness_level",
                        "competition_level",
                        "magnitud_deseo",
                        "nivel_consciencia",
                        "saturacion_mercado",
                        "source",
                        "decision",
                    }
                }
                _ = database.insert_product(
                    conn,
                    name=name,
                    description=description,
                    category=category,
                    price=price,
                    currency=currency,
                    image_url=image_url,
                    date_range=date_range,
                    desire=desire,
                    desire_magnitude=desire_mag,
                    awareness_level=awareness,
                    competition_level=competition,
                    source=source,
                    extra=extra,
                )
                inserted += 1
        else:
            print("Formato de archivo no soportado. Utilice .csv o .json.")
            return
        print(f"Se importaron {inserted} productos correctamente.")
    except Exception as exc:
        print(f"Error al importar datos: {exc}")


def add_product_manually(conn: database.sqlite3.Connection) -> None:
    print("Añadir producto manualmente o mediante URL")
    choice = prompt_user("1) Introducir datos manualmente\n2) Proporcionar URL para scraping\nSeleccione opción (1/2): ").strip()
    if choice not in {"1", "2"}:
        print("Opción no válida.")
        return
    if choice == "2":
        url = prompt_user("Introduzca la URL del producto: ").strip()
        if not url:
            print("URL vacía.")
            return
        scraped = scraper.scrape_product_from_url(url)
        if not scraped:
            print("No se pudo obtener información de la URL. Introduzca los datos manualmente.")
            choice = "1"
        else:
            name = scraped.get("name") or prompt_user("Nombre del producto: ").strip()
            description = scraped.get("description") or prompt_user("Descripción del producto: ").strip()
            category = prompt_user("Categoría (opcional): ").strip() or None
            price = scraped.get("price")
            if price is None:
                price_str = prompt_user("Precio (opcional, solo números): ").strip()
                if price_str:
                    try:
                        price = float(price_str.replace(",", "."))
                    except ValueError:
                        price = None
            image_url = prompt_user("URL de imagen (opcional): ").strip() or None
            source = prompt_user("Fuente (opcional): ").strip() or None
            product_id = database.insert_product(
                conn,
                name=name,
                description=description,
                category=category,
                price=price,
                currency=None,
                image_url=image_url,
                source=source,
                extra=None,
            )
            print(f"Producto añadido con ID {product_id}.")
            return
    # manual entry
    name = prompt_user("Nombre del producto: ").strip()
    if not name:
        print("Debe introducir un nombre.")
        return
    description = prompt_user("Descripción del producto (opcional): ").strip() or None
    category = prompt_user("Categoría (opcional): ").strip() or None
    price = None
    price_str = prompt_user("Precio (opcional, solo números): ").strip()
    if price_str:
        try:
            price = float(price_str.replace(",", "."))
        except ValueError:
            price = None
    image_url = prompt_user("URL de imagen (opcional): ").strip() or None
    source = prompt_user("Fuente (opcional): ").strip() or None
    product_id = database.insert_product(
        conn,
        name=name,
        description=description,
        category=category,
        price=price,
        currency=None,
        image_url=image_url,
        source=source,
        extra=None,
    )
    print(f"Producto añadido con ID {product_id}.")


def list_products(conn: database.sqlite3.Connection) -> None:
    products = database.list_products(conn)
    if not products:
        print("No hay productos importados todavía.")
        return
    print("Lista de productos:")
    print("ID | Nombre | Precio | Fuente | Fecha de importación")
    for p in products:
        price = p["price"] if p["price"] is not None else "-"
        print(f"{p['id']:>3} | {p['name'][:30]:<30} | {price!s:<10} | {p['source'] or '-':<10} | {p['import_date'][:19]}")


def view_product(conn: database.sqlite3.Connection) -> None:
    pid_str = prompt_user("ID del producto a ver: ").strip()
    try:
        pid = int(pid_str)
    except ValueError:
        print("ID inválido.")
        return
    product = row_to_dict(database.get_product(conn, pid))
    if not product:
        print("Producto no encontrado.")
        return
    print(f"\n--- Detalles del producto {pid} ---")
    for key in ["name", "description", "category", "price", "currency", "image_url", "source", "import_date"]:
        print(f"{key.capitalize()}: {product[key] if product[key] is not None else '-'}")
    # Show latest score if available
    scores = database.get_scores_for_product(conn, pid)
    if scores:
        score = scores[0]
        print("\nÚltimo BA‑Score:")
        print(f"Modelo: {score['model']}")
        print(f"Total: {score['total_score']}")
        print(f"Momentum: {score['momentum']}, Saturación: {score['saturation']}, Diferenciación: {score['differentiation']}")
        print(f"Prueba Social: {score['social_proof']}, Margen: {score['margin']}, Logística: {score['logistics']}")
        print(f"Resumen: {score['summary'][:200]}...")
    else:
        print("\nEste producto todavía no ha sido evaluado.")


def evaluate_product(conn: database.sqlite3.Connection) -> None:
    pid_str = prompt_user("ID del producto a evaluar: ").strip()
    try:
        pid = int(pid_str)
    except ValueError:
        print("ID inválido.")
        return
    product = database.get_product(conn, pid)
    if not product:
        print("Producto no encontrado.")
        return
    api_key = config.get_api_key()
    if not api_key:
        print("No se ha configurado la API Key de OpenAI. Use la opción correspondiente para guardarla.")
        return
    model = config.get_model()
    print(f"Evaluando producto '{product['name']}' con el modelo {model}...")
    try:
        try:
            extra = json.loads(rget(product, "extra") or "{}")
        except Exception:
            extra = {}
        resp = gpt.evaluate_winner_score(
            api_key,
            model,
            {
                "title": rget(product, "name"),
                "description": rget(product, "description"),
                "category": rget(product, "category"),
                "metrics": extra,
            },
        )
        weights_map = config.get_weights()
        scores = resp.get("scores", {})
        justifs = resp.get("justifications", {})
        weighted = sum(
            scores.get(f, 3) * weights_map.get(f, 0.0) for f in WINNER_SCORE_FIELDS
        )
        raw_score = weighted * 8.0
        pct = ((raw_score - 8.0) / 32.0) * 100.0
        pct = max(0, min(100, round(pct)))
        breakdown = {
            "scores": scores,
            "justifications": justifs,
            "weights": weights_map,
        }
        database.insert_score(
            conn,
            product_id=pid,
            model=model,
            total_score=0,
            momentum=0,
            saturation=0,
            differentiation=0,
            social_proof=0,
            margin=0,
            logistics=0,
            summary="",
            explanations={},
            winner_score_raw=raw_score,
            winner_score=pct,
            winner_score_breakdown=breakdown,
        )
        print("Evaluación completada y guardada.")
    except gpt.OpenAIError as exc:
        print(f"Error al evaluar con OpenAI: {exc}")


def configure_api() -> None:
    api_key = prompt_user("Introduzca su OpenAI API Key: ").strip()
    if not api_key:
        print("API Key vacía. No se guardó.")
        return
    model = prompt_user("Nombre del modelo de OpenAI a usar (ej. gpt-4o): ").strip() or "gpt-4o"
    cfg = config.load_config()
    cfg["api_key"] = api_key
    cfg["model"] = model
    config.save_config(cfg)
    print("API Key y modelo guardados correctamente.")


def export_data(conn: database.sqlite3.Connection) -> None:
    products = database.list_products(conn)
    if not products:
        print("No hay productos para exportar.")
        return
    fmt = prompt_user("Formato de exportación (csv/json): ").strip().lower()
    out_path = prompt_user("Ruta de archivo destino: ").strip()
    if not out_path:
        print("Ruta vacía.")
        return
    out_file = Path(out_path).expanduser()
    try:
        if fmt == "csv":
            with open(out_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "id",
                    "name",
                    "description",
                    "category",
                    "price",
                    "currency",
                    "image_url",
                    "source",
                    "import_date",
                    "Desire",
                    "Desire Magnitude",
                    "Awareness Level",
                    "Competition Level",
                ])
                for p in products:
                    writer.writerow([
                        p["id"],
                        p["name"],
                        p["description"],
                        p["category"],
                        p["price"],
                        p["currency"],
                        p["image_url"],
                        p["source"],
                        p["import_date"],
                        p["desire"],
                        p["desire_magnitude"],
                        p["awareness_level"],
                        p["competition_level"],
                    ])
            print(f"Datos exportados a {out_file}")
        elif fmt == "json":
            json_data = []
            for p in products:
                json_data.append(
                    {
                        "id": p["id"],
                        "name": p["name"],
                        "description": p["description"],
                        "category": p["category"],
                        "price": p["price"],
                        "currency": p["currency"],
                        "image_url": p["image_url"],
                        "source": p["source"],
                        "import_date": p["import_date"],
                        "desire": p["desire"],
                        "desire_magnitude": p["desire_magnitude"],
                        "awareness_level": p["awareness_level"],
                        "competition_level": p["competition_level"],
                    }
                )
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"Datos exportados a {out_file}")
        else:
            print("Formato no soportado. Use 'csv' o 'json'.")
    except Exception as exc:
        print(f"Error al exportar datos: {exc}")


def export_product_pdf(conn: database.sqlite3.Connection) -> None:
    pid_str = prompt_user("ID del producto para exportar a PDF: ").strip()
    try:
        pid = int(pid_str)
    except ValueError:
        print("ID inválido.")
        return
    product = database.get_product(conn, pid)
    if not product:
        print("Producto no encontrado.")
        return
    scores = database.get_scores_for_product(conn, pid)
    score = scores[0] if scores else None
    out_path = prompt_user("Ruta del archivo PDF a generar (ej. reporte.pdf): ").strip()
    if not out_path:
        print("Ruta vacía.")
        return
    out_file = Path(out_path).expanduser()
    try:
        generate_pdf_report(product, score, out_file)
        print(f"Reporte generado en {out_file}")
    except Exception as exc:
        print(f"Error al generar PDF: {exc}")


def generate_pdf_report(product: Any, score: Optional[Any], out_file: Path) -> None:
    """Generate a simple PDF report for a product using Pillow.

    The report is created by drawing text on a blank image and then saving
    it as a single page PDF.  This avoids the need for external PDF libraries.
    """
    # Define page size (approx A4 at 72 DPI)
    width, height = 595, 842
    img = Image.new("RGB", (width, height), color="white")
    draw = ImageDraw.Draw(img)
    # Load default font
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 16)
    except Exception:
        font = ImageFont.load_default()
        title_font = font
    y = 20
    draw.text((20, y), f"Reporte de Producto ID {product['id']}", font=title_font, fill="black")
    y += 30
    draw.text((20, y), f"Nombre: {product['name']}", font=font, fill="black")
    y += 20
    if product['description']:
        draw.text((20, y), f"Descripción: {product['description'][:200]}", font=font, fill="black")
        y += 40
    if product['category']:
        draw.text((20, y), f"Categoría: {product['category']}", font=font, fill="black")
        y += 20
    if product['price'] is not None:
        draw.text((20, y), f"Precio: {product['price']}", font=font, fill="black")
        y += 20
    if product['source']:
        draw.text((20, y), f"Fuente: {product['source']}", font=font, fill="black")
        y += 20
    if score:
        y += 10
        draw.text((20, y), "BA‑Score:", font=title_font, fill="black")
        y += 25
        draw.text((20, y), f"Modelo: {score['model']}", font=font, fill="black")
        y += 20
        draw.text((20, y), f"Total: {score['total_score']}", font=font, fill="black")
        y += 20
        draw.text((20, y), f"Momentum: {score['momentum']}\nSaturación: {score['saturation']}\nDiferenciación: {score['differentiation']}", font=font, fill="black")
        y += 45
        draw.text((20, y), f"Prueba social: {score['social_proof']}\nMargen: {score['margin']}\nLogística: {score['logistics']}", font=font, fill="black")
        y += 45
        # Summary cropping to fit page
        summary = score['summary'] if score['summary'] else ''
        summary_lines = text_wrap(summary, font, width - 40)
        draw.text((20, y), "Resumen:", font=font, fill="black")
        y += 20
        for line in summary_lines:
            if y > height - 40:
                break
            draw.text((25, y), line, font=font, fill="black")
            y += 15
    # Save as PDF
    img.save(out_file, "PDF", resolution=100.0)


def text_wrap(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Utility to wrap text into lines that fit within max_width."""
    lines: List[str] = []
    if not text:
        return lines
    words = text.split()
    line = ""
    for word in words:
        test_line = line + (" " if line else "") + word
        width, _ = font.getsize(test_line)
        if width <= max_width:
            line = test_line
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def manage_lists(conn: database.sqlite3.Connection) -> None:
    print("Gestión de listas de productos")
    print("1) Ver listas existentes")
    print("2) Crear nueva lista")
    print("3) Añadir producto a una lista")
    print("4) Ver productos de una lista")
    option = prompt_user("Seleccione opción: ").strip()
    if option == "1":
        lists = database.get_lists(conn)
        if not lists:
            print("No hay listas definidas.")
            return
        print("Listas:")
        for lst in lists:
            print(f"{lst['id']}: {lst['name']}")
    elif option == "2":
        name = prompt_user("Nombre de la nueva lista: ").strip()
        if not name:
            print("Nombre no puede ser vacío.")
            return
        list_id = database.create_list(conn, name)
        print(f"Lista creada con ID {list_id}.")
    elif option == "3":
        lists = database.get_lists(conn)
        if not lists:
            print("Primero cree una lista.")
            return
        print("Seleccione una lista por ID:")
        for lst in lists:
            print(f"{lst['id']}: {lst['name']}")
        list_id_str = prompt_user("ID de la lista: ").strip()
        try:
            list_id = int(list_id_str)
        except ValueError:
            print("ID inválido.")
            return
        product_id_str = prompt_user("ID del producto a añadir: ").strip()
        try:
            product_id = int(product_id_str)
        except ValueError:
            print("ID inválido.")
            return
        database.add_product_to_list(conn, list_id, product_id)
        print("Producto añadido a la lista.")
    elif option == "4":
        lists = database.get_lists(conn)
        if not lists:
            print("No hay listas definidas.")
            return
        for lst in lists:
            print(f"{lst['id']}: {lst['name']}")
        list_id_str = prompt_user("ID de la lista a ver: ").strip()
        try:
            list_id = int(list_id_str)
        except ValueError:
            print("ID inválido.")
            return
        products = database.get_products_in_list(conn, list_id)
        if not products:
            print("La lista está vacía.")
            return
        print(f"Productos en la lista {list_id}:")
        for p in products:
            print(f"{p['id']}: {p['name']} (BA‑Score: {database.get_scores_for_product(conn, p['id'])[0]['total_score'] if database.get_scores_for_product(conn, p['id']) else 'n/a'})")
    else:
        print("Opción no reconocida.")


def compare_products(conn: database.sqlite3.Connection) -> None:
    ids_str = prompt_user("Introduzca IDs de productos a comparar separados por comas (máximo 4): ").strip()
    if not ids_str:
        print("No se introdujeron IDs.")
        return
    id_list = []
    for part in ids_str.split(","):
        try:
            pid = int(part.strip())
            id_list.append(pid)
        except ValueError:
            continue
    if not id_list:
        print("No se reconocieron IDs válidos.")
        return
    if len(id_list) > 4:
        print("Seleccione hasta 4 productos.")
        id_list = id_list[:4]
    products: List[Dict[str, Any]] = []
    for pid in id_list:
        prod = database.get_product(conn, pid)
        if prod:
            products.append(prod)
    if not products:
        print("No se encontraron productos para comparar.")
        return
    # Display comparison table
    print("\nComparación de productos:")
    headers = ["Métrica"] + [p["name"][:20] for p in products]
    rows = []
    # Retrieve latest scores for each product
    scores_map: Dict[int, Optional[sqlite3.Row]] = {}
    for p in products:
        scores = database.get_scores_for_product(conn, p["id"])
        scores_map[p["id"]] = scores[0] if scores else None
    # Create rows for total score and submetrics
    metrics = [
        ("Total", "total_score"),
        ("Momentum", "momentum"),
        ("Saturación", "saturation"),
        ("Diferenciación", "differentiation"),
        ("Prueba social", "social_proof"),
        ("Margen", "margin"),
        ("Logística", "logistics"),
    ]
    for label, field in metrics:
        row = [label]
        for p in products:
            sc = scores_map.get(p["id"])
            value = sc[field] if sc else "n/a"
            row.append(value)
        rows.append(row)
    # Print table
    col_widths = [max(len(str(cell)) for cell in col) for col in zip(*([headers] + rows))]
    def print_row(row: List[Any]) -> None:
        print(" | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths)))
    print_row(headers)
    print("-+-".join("-" * w for w in col_widths))
    for row in rows:
        print_row(row)


def run_batch(
    directory: str,
    export_path: str,
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> None:
    """
    Execute a non‑interactive batch import and evaluation.

    This function scans the given directory for CSV and JSON files, imports all
    detected products, evaluates them with GPT if they have no score yet and
    finally exports the ranked list to a CSV file.  It prints a summary of the
    top 10 products to the console.

    Args:
        directory: Path to the directory containing CSV/JSON files.
        export_path: Destination file for the CSV results.
        api_key: Optional OpenAI API key.  Falls back to config or env.
        model: Optional model name.  Falls back to config or default.
    """
    dir_path = Path(directory).expanduser()
    if not dir_path.is_dir():
        print(f"Directorio no encontrado: {dir_path}")
        return
    conn = ensure_database()
    # Import all files
    imported_files = 0
    inserted = 0
    for file_path in dir_path.iterdir():
        if file_path.suffix.lower() not in {".csv", ".json"}:
            continue
        imported_files += 1
        source = file_path.stem
        try:
            if file_path.suffix.lower() == ".csv":
                with open(file_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    headers = reader.fieldnames or []
                    # heuristics
                    def find_col(names: List[str], options: List[str]) -> Optional[str]:
                        for opt in options:
                            for n in names:
                                if opt.lower() in n.lower():
                                    return n
                        return None
                    name_col = find_col(headers, ["name", "nombre", "product", "title"])
                    desc_col = find_col(headers, ["description", "descripcion", "desc"])
                    cat_col = find_col(headers, ["category", "categoria"])
                    price_col = find_col(headers, ["price", "precio", "cost"])
                    currency_col = find_col(headers, ["currency", "moneda"])
                    image_col = find_col(headers, ["image", "imagen", "img", "picture"])
                    for row in reader:
                        name = (row.get(name_col) or "").strip() if name_col else None
                        if not name:
                            continue
                        description = (row.get(desc_col) or "").strip() if desc_col else None
                        category = (row.get(cat_col) or "").strip() if cat_col else None
                        price = None
                        if price_col and row.get(price_col):
                            try:
                                price = float(str(row.get(price_col)).replace(",", "."))
                            except ValueError:
                                price = None
                        currency = (row.get(currency_col) or "").strip() if currency_col else None
                        image_url = (row.get(image_col) or "").strip() if image_col else None
                        extra_cols = {k: v for k, v in row.items() if k not in {name_col, desc_col, cat_col, price_col, currency_col, image_col}}
                        _ = database.insert_product(
                            conn,
                            name=name,
                            description=description,
                            category=category,
                            price=price,
                            currency=currency,
                            image_url=image_url,
                            source=source,
                            extra=extra_cols,
                        )
                        inserted += 1
            elif file_path.suffix.lower() == ".json":
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    continue
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("nombre") or item.get("product") or item.get("title")
                    if not name:
                        continue
                    description = item.get("description") or item.get("descripcion")
                    category = item.get("category") or item.get("categoria")
                    price = None
                    for key in ["price", "precio", "cost"]:
                        if key in item:
                            try:
                                price = float(str(item[key]).replace(",", "."))
                                break
                            except ValueError:
                                continue
                    currency = item.get("currency") or item.get("moneda")
                    image_url = item.get("image") or item.get("imagen") or item.get("img")
                    extra = {k: v for k, v in item.items() if k not in {"name", "nombre", "product", "title", "description", "descripcion", "category", "categoria", "price", "precio", "cost", "currency", "moneda", "image", "imagen", "img"}}
                    _ = database.insert_product(
                        conn,
                        name=name,
                        description=description,
                        category=category,
                        price=price,
                        currency=currency,
                        image_url=image_url,
                        source=source,
                        extra=extra,
                    )
                    inserted += 1
        except Exception as exc:
            print(f"Error al importar {file_path}: {exc}")
    print(f"Importados {inserted} productos desde {imported_files} archivos.")
    # Evaluate products
    api = api_key or config.get_api_key() or os.environ.get("OPENAI_API_KEY")
    model_name = model or config.get_model()
    if not api:
        print("No se ha configurado la API Key de OpenAI. Se omiten las evaluaciones de IA.")
    else:
        prods = database.list_products(conn)
        evaluated = 0
        for p in prods:
            # check if there is already a score
            if database.get_scores_for_product(conn, p["id"]):
                continue
            try:
                result = gpt.evaluate_product(api, model_name, dict(p))
                total_score = float(result.get("totalScore"))
                momentum = float(result.get("momentum_score"))
                saturation = float(result.get("saturation_score"))
                differentiation = float(result.get("differentiation_score"))
                social = float(result.get("social_proof_score"))
                margin_val = float(result.get("margin_score"))
                logistics = float(result.get("logistics_score"))
                summary = result.get("summary", "")
                explanations = {
                    "momentum": result.get("momentum_explanation"),
                    "saturation": result.get("saturation_explanation"),
                    "differentiation": result.get("differentiation_explanation"),
                    "social_proof": result.get("social_proof_explanation"),
                    "margin": result.get("margin_explanation"),
                    "logistics": result.get("logistics_explanation"),
                }
                database.insert_score(
                    conn,
                    product_id=p["id"],
                    model=model_name,
                    total_score=total_score,
                    momentum=momentum,
                    saturation=saturation,
                    differentiation=differentiation,
                    social_proof=social,
                    margin=margin_val,
                    logistics=logistics,
                    summary=summary,
                    explanations=explanations,
                )
                evaluated += 1
            except gpt.OpenAIError as exc:
                print(f"Error al evaluar producto {p['id']}: {exc}")
        print(f"Evaluados {evaluated} productos con IA.")
    # Gather latest scores and sort
    rows = []
    for p in database.list_products(conn):
        scs = database.get_scores_for_product(conn, p["id"])
        if scs:
            sc = scs[0]
            rows.append({
                "id": p["id"],
                "name": p["name"],
                "total_score": sc["total_score"],
                "momentum": sc["momentum"],
                "saturation": sc["saturation"],
                "differentiation": sc["differentiation"],
                "social_proof": sc["social_proof"],
                "margin": sc["margin"],
                "logistics": sc["logistics"],
                "summary": sc["summary"],
            })
    # sort by total score descending
    rows.sort(key=lambda x: x.get("total_score", 0), reverse=True)
    # Export
    if rows:
        try:
            with open(export_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["id", "name", "total_score", "momentum", "saturation", "differentiation", "social_proof", "margin", "logistics", "summary"])
                for r in rows:
                    writer.writerow([
                        r["id"], r["name"], r["total_score"], r["momentum"], r["saturation"], r["differentiation"], r["social_proof"], r["margin"], r["logistics"], r["summary"]
                    ])
            print(f"Resultados exportados a {export_path}")
        except Exception as exc:
            print(f"No se pudieron exportar los resultados: {exc}")
    # Print top 10
    print("\nTop 10 productos por BA‑Score:")
    for idx, r in enumerate(rows[:10], start=1):
        print(f"{idx}. {r['name']} (ID {r['id']}): {r['total_score']}")


def interactive_menu(conn: sqlite3.Connection) -> None:
    """Run the interactive menu loop."""
    while True:
        print("\n=== Product Research Copilot ===")
        print("1) Importar datos desde CSV/JSON")
        print("2) Añadir producto manualmente/por URL")
        print("3) Listar productos")
        print("4) Ver detalles de un producto")
        print("5) Evaluar producto con GPT")
        print("6) Configurar API Key y modelo de OpenAI")
        print("7) Exportar datos (CSV/JSON)")
        print("8) Exportar reporte PDF de un producto")
        print("9) Gestionar listas de productos")
        print("10) Comparar productos")
        print("0) Salir")
        option = prompt_user("Seleccione una opción: ").strip()
        if option == "1":
            import_data(conn)
        elif option == "2":
            add_product_manually(conn)
        elif option == "3":
            list_products(conn)
        elif option == "4":
            view_product(conn)
        elif option == "5":
            evaluate_product(conn)
        elif option == "6":
            configure_api()
        elif option == "7":
            export_data(conn)
        elif option == "8":
            export_product_pdf(conn)
        elif option == "9":
            manage_lists(conn)
        elif option == "10":
            compare_products(conn)
        elif option == "0":
            print("Hasta luego!")
            break
        else:
            print("Opción no válida. Intente de nuevo.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Product Research Copilot")
    parser.add_argument("--batch", help="Ejecutar en modo batch importando todos los CSV/JSON de un directorio y evaluando los productos")
    parser.add_argument("--export", help="Archivo de salida para el modo batch", default="batch_results.csv")
    parser.add_argument("--api-key", help="API Key de OpenAI para el modo batch")
    parser.add_argument("--model", help="Modelo de OpenAI a usar en modo batch")
    args = parser.parse_args()
    if args.batch:
        run_batch(args.batch, args.export, args.api_key, args.model)
        return
    conn = ensure_database()
    interactive_menu(conn)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupción por el usuario. Adiós.")
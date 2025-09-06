import argparse
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List

PREFIX = "[SQLite-Maintenance]"


def log(msg: str) -> None:
    print(f"{PREFIX} {msg}")


def locate_db(repo_root: Path) -> Path:
    default = repo_root / "product_research_app" / "data.sqlite3"
    if default.exists():
        return default
    for path in repo_root.rglob("*.sqlite*"):
        if path.is_file():
            return path
    raise FileNotFoundError("No se encontró ningún archivo SQLite")


def find_foreign_key_refs(conn: sqlite3.Connection, target: str) -> List[str]:
    cur = conn.cursor()
    tables = [row[0] for row in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    refs = []
    for tbl in tables:
        fk_cur = cur.execute(f"PRAGMA foreign_key_list({tbl})")
        for fk in fk_cur.fetchall():
            # fk[2] is the table referenced
            if fk[2] == target:
                refs.append(tbl)
                break
    return refs


def _insert_dummy(conn: sqlite3.Connection, table: str) -> int:
    cur = conn.cursor()
    info = cur.execute(f"PRAGMA table_info({table})").fetchall()
    cols: List[str] = []
    vals: List[object] = []
    for _, name, coltype, notnull, dflt, pk in info:
        if pk:
            continue
        if notnull and dflt is None:
            cols.append(name)
            t = coltype.upper() if coltype else ""
            if any(x in t for x in ["INT", "REAL", "NUM"]):
                vals.append(0)
            else:
                # Generic placeholder
                vals.append("test")
    if cols:
        placeholders = ",".join(["?"] * len(cols))
        columns = ",".join(cols)
        cur.execute(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", vals)
    else:
        cur.execute(f"INSERT INTO {table} DEFAULT VALUES")
    return cur.lastrowid


def purge_and_reset(args: argparse.Namespace) -> int:
    env = os.getenv("APP_ENV") or os.getenv("ENV")
    allowed_envs = {"development", "local"}
    if env not in allowed_envs and not args.dangerously_on_prod:
        log("Entorno no permitido. Use --dangerously-on-prod si está seguro.")
        return 1
    if args.no_prompt:
        if not (args.yes and args.i_know):
            log("Faltan confirmaciones --yes y --i-know-what-im-doing.")
            return 2
    else:
        if not args.yes:
            resp = input("Esta acción eliminará datos. Escribe 'yes' para continuar: ")
            if resp.strip().lower() != "yes":
                log("Operación cancelada por el usuario.")
                return 2
        if not args.i_know:
            resp = input("Escribe 'I know what I'm doing' para continuar: ")
            if resp.strip() != "I know what I'm doing":
                log("Operación cancelada por el usuario.")
                return 2
    repo_root = Path(__file__).resolve().parent.parent
    try:
        db_path = locate_db(repo_root)
    except FileNotFoundError as e:
        log(str(e))
        return 1
    log(f"Base de datos objetivo: {db_path}")
    backup_dir = repo_root / "backups"
    backup_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    backup_path = backup_dir / f"{db_path.name}.{timestamp}.bak"
    shutil.copy2(db_path, backup_path)
    log(f"Copia de seguridad creada en {backup_path}")
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        cur = conn.cursor()
        table = args.table
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?", (table,)
        )
        if not cur.fetchone():
            log(f"La tabla '{table}' no existe.")
            return 1
        refs = find_foreign_key_refs(conn, table)
        if refs and not args.force:
            log(f"Existen claves foráneas apuntando a {table}: {', '.join(refs)}")
            log("Use --force para continuar de todas maneras.")
            return 1
        if refs and args.force:
            log("ADVERTENCIA: Forzando a pesar de claves foráneas activas.")
        cur.execute(f"SELECT COUNT(*) FROM {table}")
        rows = cur.fetchone()[0]
        cur.execute("SELECT seq FROM sqlite_sequence WHERE name=?", (table,))
        seq_row = cur.fetchone()
        seq_before = seq_row[0] if seq_row else 0
        if rows == 0 and seq_before in (0, None):
            log("La tabla ya está vacía y el contador reiniciado. Nada que hacer.")
            return 0
        try:
            cur.execute("BEGIN")
            cur.execute(f"DELETE FROM {table}")
            cur.execute("DELETE FROM sqlite_sequence WHERE name=?", (table,))
            conn.commit()
        except sqlite3.DatabaseError as e:
            conn.rollback()
            log(f"Error al limpiar la tabla: {e}")
            return 3
        try:
            conn.execute("VACUUM")
        except sqlite3.DatabaseError as e:
            log(f"Error en VACUUM: {e}")
            return 3
        cur.execute("SELECT seq FROM sqlite_sequence WHERE name=?", (table,))
        seq_row = cur.fetchone()
        seq_after = seq_row[0] if seq_row else 0
        try:
            cur.execute("BEGIN")
            new_id = _insert_dummy(conn, table)
            cur.execute("ROLLBACK")
        except Exception as e:
            cur.execute("ROLLBACK")
            log(f"Error verificando inserción: {e}")
            return 3
        log(f"Filas eliminadas: {rows}")
        log(f"Contador antes: {seq_before} / después: {seq_after}")
        log(f"Verificación final, próxima id será {new_id}")
        conn.close()
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Herramientas de mantenimiento SQLite")
    sub = parser.add_subparsers(dest="command")
    p = sub.add_parser("purge-and-reset", help="Vacía tabla y reinicia IDs")
    p.add_argument("--table", default="products")
    p.add_argument("--yes", action="store_true", help="Confirmar acción")
    p.add_argument("--i-know-what-im-doing", dest="i_know", action="store_true", help="Confirmación adicional")
    p.add_argument("--no-prompt", action="store_true", help="Modo no interactivo")
    p.add_argument("--force", action="store_true", help="Ignorar claves foráneas")
    p.add_argument("--dangerously-on-prod", action="store_true", dest="dangerously_on_prod", help="Permite ejecución en producción")
    args = parser.parse_args(argv)
    if args.command == "purge-and-reset":
        return purge_and_reset(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations
import argparse, os, re, sys, time, json
from pathlib import Path

EXCLUDE_DIRS = {'.git','.github','.venv','venv','env','python_embed','node_modules','build','dist','logs','.idea','.vscode','__pycache__','.mypy_cache','.pytest_cache'}

FILE_GLOB_EXTS = [
    '*.bak','*.bkp','*.backup','*.bk','*.old','*.orig','*.save','*.sav',
    '*.tmp','*.tmp~','*.swp','*.swo','*~'
]

# Regex que disparan por NOMBRE (no solo extensión)
NAME_REGEXES = [
    re.compile(r'(?i)(?:^|[\s._-])copy of\s+'),          # "Copy of X"
    re.compile(r'(?i)(?:^|[\s._-])copia de\s+'),         # "Copia de X"
    re.compile(r'(?i)\((?:copy|copia|backup)\)'),        # "(copy)" "(copia)" "(backup)"
    re.compile(r'(?i)\s-\s*(?:copy|copia)$'),            # " - copy" / " - copia"
    re.compile(r'\s\(\d+\)(\.[^.]+)?$'),                 # " (1).ext"
]

# Comprimidos solo si en el nombre aparece "backup|copia|copy"
ARCHIVE_REGEX = re.compile(r'(?i).*\b(backup|copia|copy)\b.*\.(zip|7z|rar|tar|tar\.gz|tgz)$')

# Directorios "backup"
DIR_REGEXES = [
    re.compile(r'(?i)^(backup|backups|bak|bk)$'),
    re.compile(r'(?i).*(?:^|[_\-.])backup(?:$|[_\-.]).*'),
]

# Evitar falsos positivos útiles de sqlite
SQLITE_SKIP = re.compile(r'.*\.(sqlite|sqlite3)$|.*-(wal|shm)$', re.IGNORECASE)

def is_excluded(path: Path) -> bool:
    parts = {p.name for p in path.parts}
    return any(ex in parts for ex in EXCLUDE_DIRS)

def glob_backup_files(root: Path) -> list[tuple[Path,str]]:
    found: list[tuple[Path,str]] = []
    for pat in FILE_GLOB_EXTS:
        for p in root.rglob(pat):
            if p.is_file() and not is_excluded(p):
                if SQLITE_SKIP.search(str(p)):  # solo salta si NO es .bak etc.
                    # si además tiene .bak/.old ya lo cazamos por glob; si no, se omite
                    if p.suffix.lower() not in {'.bak','.bkp','.backup','.bk','.old','.orig','.save','.sav'}:
                        continue
                found.append((p, f'glob:{pat}'))
    # Nombres por regex
    for p in root.rglob('*'):
        if not p.is_file() or is_excluded(p): 
            continue
        name = p.name
        if ARCHIVE_REGEX.match(name):
            found.append((p, 'archive-name:backup|copia|copy'))
            continue
        for rx in NAME_REGEXES:
            if rx.search(name):
                found.append((p, f'regex:{rx.pattern}'))
                break
    # dedup
    uniq = {}
    for p, why in found:
        uniq[str(p)] = (p, why)  # última razón
    return list(uniq.values())

def find_backup_dirs(root: Path) -> list[tuple[Path,str]]:
    cands: list[tuple[Path,str]] = []
    for p in root.rglob('*'):
        if p.is_dir() and not is_excluded(p):
            for rx in DIR_REGEXES:
                if rx.match(p.name) or rx.search(p.name):
                    cands.append((p, f'dir:{rx.pattern}'))
                    break
    # ordenar de más profundo a menos (útil para borrado)
    cands = sorted(set((str(p), why) for p,why in cands), key=lambda t: t[0].count(os.sep), reverse=True)
    return [(Path(s), why) for s,why in cands]

def write_report(report_path: Path, files: list[tuple[Path,str]], dirs: list[tuple[Path,str]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('w', encoding='utf-8') as f:
        f.write('# Backups encontrados\n\n')
        f.write(f'Generado: {time.strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write(f'Archivos: {len(files)} | Directorios: {len(dirs)}\n\n')
        if files:
            f.write('## Archivos\n\n')
            f.write('| Ruta | Regla |\n|---|---|\n')
            for p, why in sorted(files, key=lambda t: t[0].as_posix().lower()):
                f.write(f'| `{p.as_posix()}` | `{why}` |\n')
            f.write('\n')
        if dirs:
            f.write('## Directorios\n\n')
            f.write('| Ruta | Regla |\n|---|---|\n')
            for p, why in sorted(dirs, key=lambda t: t[0].as_posix().lower()):
                f.write(f'| `{p.as_posix()}` | `{why}` |\n')
            f.write('\n')

def main():
    ap = argparse.ArgumentParser(description='Escanea y elimina backups del repo.')
    ap.add_argument('--purge', action='store_true', help='Borra en vez de sólo informar (dry-run por defecto).')
    ap.add_argument('--root', default='.', help='Raíz a escanear (por defecto, .)')
    args = ap.parse_args()

    root = Path(args.root).resolve()
    files = glob_backup_files(root)
    dirs  = find_backup_dirs(root)

    report = root / 'backups_report.md'
    write_report(report, files, dirs)

    print(json.dumps({
        'root': str(root),
        'files': len(files),
        'dirs': len(dirs),
        'report': str(report)
    }, ensure_ascii=False, indent=2))

    if not args.purge:
        print('\n[DRY-RUN] No se eliminó nada. Revisa backups_report.md')
        return

    # Borrado real: primero archivos, luego directorios
    logdir = root / 'logs'
    logdir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    logf = logdir / f'cleanup_backups_{stamp}.log'
    with logf.open('w', encoding='utf-8') as lf:
        lf.write(f'Cleanup started {time.ctime()}\n')
        # archivos
        for p,_ in files:
            try:
                p.unlink(missing_ok=True)
                lf.write(f'DEL FILE {p}\n')
            except Exception as e:
                lf.write(f'ERR FILE {p}: {e}\n')
        # carpetas (profundas primero)
        for p,_ in dirs:
            try:
                # borra en cascada si está vacía; si no, fuerza recursivo
                if p.exists():
                    for sub in sorted(p.rglob('*'), key=lambda q: q.as_posix().count('/'), reverse=True):
                        try:
                            if sub.is_file() or sub.is_symlink(): sub.unlink(missing_ok=True)
                            elif sub.is_dir(): sub.rmdir()
                        except Exception:
                            pass
                    p.rmdir()
                lf.write(f'DEL DIR  {p}\n')
            except Exception as e:
                lf.write(f'ERR DIR  {p}: {e}\n')
        lf.write('Cleanup finished\n')
    print(f'\n[PURGE] Eliminación completada. Log: {logf}')

if __name__ == '__main__':
    main()

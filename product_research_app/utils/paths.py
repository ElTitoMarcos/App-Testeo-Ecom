"""Utility helpers for runtime paths.

This module centralises logic to resolve directories used by the
application so we can keep path handling cross-platform.  Using
:class:`pathlib.Path` everywhere avoids assumptions about the path
separator and makes it easier to fall back to user-writable locations on
systems such as macOS where the application bundle may be read-only.
"""

from __future__ import annotations

import os
import platform
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Iterable

_APP_STORAGE_ENV = "APP_STORAGE_DIR"
_APP_LOG_ENV = "APP_LOG_DIR"
_APP_DB_ENV = "PRAPP_DB_PATH"
_APP_NAMESPACE = "com.ecomtesting"


def _can_write(path: Path) -> bool:
    """Return ``True`` if ``path`` can be created and written to."""

    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".__permcheck_{os.getpid()}"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return True
    except OSError:
        return False


@lru_cache(maxsize=None)
def package_dir() -> Path:
    """Return the directory that contains the Python package."""

    return Path(__file__).resolve().parents[1]


def _platform_data_home() -> Iterable[Path]:
    """Yield platform-specific user data directories."""

    home = Path.home()
    system = platform.system()
    if system == "Darwin":
        yield home / "Library" / "Application Support" / _APP_NAMESPACE
    elif system == "Windows":
        base = Path(os.environ.get("APPDATA") or (home / "AppData" / "Roaming"))
        yield base / "EcomTesting"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or (home / ".local" / "share"))
        yield base / "ecomtesting"
    # As an ultimate fallback use the system temporary directory.
    yield Path(tempfile.gettempdir()) / "product_research_app"


def _platform_log_home() -> Iterable[Path]:
    """Yield candidate directories for log files ordered by preference."""

    home = Path.home()
    system = platform.system()
    if system == "Darwin":
        yield home / "Library" / "Logs" / "EcomTesting"
    elif system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA") or (home / "AppData" / "Local"))
        yield base / "EcomTesting" / "Logs"
    else:
        base = Path(os.environ.get("XDG_STATE_HOME") or (home / ".local" / "state"))
        yield base / "ecomtesting" / "logs"
    yield from _platform_data_home()


@lru_cache(maxsize=None)
def data_root() -> Path:
    """Return a writable directory to persist database and config files."""

    env_dir = os.environ.get(_APP_STORAGE_ENV)
    if env_dir:
        candidate = Path(env_dir).expanduser()
        if _can_write(candidate):
            return candidate
    # Prefer the package directory for backwards compatibility if writable.
    pkg_dir = package_dir()
    if _can_write(pkg_dir):
        return pkg_dir
    for candidate in _platform_data_home():
        if _can_write(candidate):
            return candidate
    # Last resort: use the temporary directory.
    fallback = Path(tempfile.gettempdir()) / "product_research_app"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


@lru_cache(maxsize=None)
def get_data_dir() -> Path:
    """Return the directory where mutable data should be stored."""

    root = data_root()
    # When using the package directory keep files next to the code to avoid
    # breaking existing installations.  Otherwise store under ``data``.
    if root == package_dir():
        return root
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_database_path() -> Path:
    """Return the default SQLite database path."""

    env_path = os.environ.get(_APP_DB_ENV)
    if env_path:
        return Path(env_path).expanduser().resolve()
    candidate = package_dir() / "data.sqlite3"
    if candidate.exists() or _can_write(candidate.parent):
        return candidate
    fallback = get_data_dir() / "data.sqlite3"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


@lru_cache(maxsize=None)
def get_config_file() -> Path:
    """Return the path to the JSON configuration file."""

    candidate = package_dir() / "config.json"
    if candidate.exists() or _can_write(candidate.parent):
        return candidate
    fallback = get_data_dir() / "config.json"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


@lru_cache(maxsize=None)
def get_calibration_cache_file() -> Path:
    """Return the path used to cache calibration payloads."""

    candidate = package_dir() / "ai_calibration_cache.json"
    if candidate.exists() or _can_write(candidate.parent):
        return candidate
    fallback = get_data_dir() / "ai_calibration_cache.json"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    return fallback


@lru_cache(maxsize=None)
def get_log_dir() -> Path:
    """Return the directory where log files should be written."""

    env_dir = os.environ.get(_APP_LOG_ENV)
    if env_dir:
        candidate = Path(env_dir).expanduser()
        if _can_write(candidate):
            return candidate
    repo_logs = package_dir().parent / "logs"
    if _can_write(repo_logs):
        return repo_logs
    for candidate in _platform_log_home():
        if _can_write(candidate):
            return candidate
    fallback = Path(tempfile.gettempdir()) / "product_research_logs"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


@lru_cache(maxsize=None)
def get_upload_temp_dir() -> Path:
    """Return a temporary directory for file uploads."""

    base = Path(tempfile.gettempdir()) / "product_research_uploads"
    base.mkdir(parents=True, exist_ok=True)
    return base


def normalize_for_storage(path: Path | str | None) -> str | None:
    """Normalise ``path`` before storing it in SQLite."""

    if path is None:
        return None
    return os.path.normpath(str(Path(path)))

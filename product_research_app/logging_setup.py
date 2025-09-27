from __future__ import annotations

import json
import logging
import os
import socket
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

_SESSION_ID = uuid4().hex
_ADAPTER: Optional[logging.LoggerAdapter] = None
_LOG_DIR: Optional[Path] = None


class _JsonLineFormatter(logging.Formatter):
    """Formatter that serialises log records as JSON lines."""

    _RESERVED = {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - logging
        data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .astimezone()
            .isoformat(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        extra: Dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in self._RESERVED:
                continue
            extra[key] = self._serialise(value)
        if extra:
            data.update(extra)
        if record.exc_info:
            data.setdefault("traceback", self.formatException(record.exc_info))
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    def _serialise(self, value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [self._serialise(item) for item in value]
        if isinstance(value, dict):
            return {str(k): self._serialise(v) for k, v in value.items()}
        try:
            return json.loads(json.dumps(value, ensure_ascii=False))
        except Exception:
            return repr(value)


class _HumanFormatter(logging.Formatter):
    """Formatter that appends contextual extras without failing on absence."""

    def format(self, record: logging.LogRecord) -> str:  # pragma: no cover - logging
        base = super().format(record)
        meta_parts = []
        for key in ("app", "pid", "hostname", "session_id", "tz"):
            value = getattr(record, key, None)
            if value is not None:
                meta_parts.append(f"{key}={value}")
        if meta_parts:
            return f"{base} [{' '.join(meta_parts)}]"
        return base


def _get_level() -> int:
    level_name = os.environ.get("PRAPP_LOG_LEVEL", "INFO").strip().upper()
    return getattr(logging, level_name, logging.INFO)


def _tz_iso8601() -> str:
    now = datetime.now().astimezone()
    offset = now.utcoffset()
    if offset is None:
        return "+00:00"
    total_minutes = int(offset.total_seconds() // 60)
    sign = "+" if total_minutes >= 0 else "-"
    total_minutes = abs(total_minutes)
    hours, minutes = divmod(total_minutes, 60)
    return f"{sign}{hours:02d}:{minutes:02d}"


def _configure_handlers(log_dir: Path, json_enabled: bool) -> None:
    max_bytes = 5 * 1024 * 1024
    backup_count = 5

    text_handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    text_handler.setFormatter(
        _HumanFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )

    handlers = [text_handler]

    if json_enabled:
        json_handler = RotatingFileHandler(
            log_dir / "app.jsonl",
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        json_handler.setFormatter(_JsonLineFormatter())
        handlers.append(json_handler)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)
    root_logger.setLevel(_get_level())


def setup_logging(base_dir: Path | str | None = None, json_enabled: bool = True) -> logging.LoggerAdapter:
    """Configure application logging and return a logger adapter with base context."""

    global _ADAPTER, _LOG_DIR
    if _ADAPTER is not None:
        return _ADAPTER

    base_path = Path(base_dir) if base_dir else Path.cwd()
    log_dir = base_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    env_json = os.environ.get("PRAPP_LOG_JSON")
    if env_json is not None:
        json_enabled = env_json not in {"0", "false", "False"}

    _configure_handlers(log_dir, json_enabled=json_enabled)

    base_extra = {
        "app": "product_research_app",
        "pid": os.getpid(),
        "hostname": socket.gethostname(),
        "session_id": _SESSION_ID,
        "tz": _tz_iso8601(),
    }

    base_logger = logging.getLogger("product_research_app")
    _ADAPTER = logging.LoggerAdapter(base_logger, base_extra)
    _LOG_DIR = log_dir
    logging.captureWarnings(True)
    return _ADAPTER


def get_logger(name: Optional[str] = None) -> logging.LoggerAdapter:
    """Return a child logger adapter including the shared contextual extras."""

    adapter = setup_logging()
    logger = adapter.logger if name is None else logging.getLogger(name)
    extra = dict(adapter.extra)
    return logging.LoggerAdapter(logger, extra)


def get_log_dir() -> Path:
    """Return the directory where log files are written."""

    if _LOG_DIR is None:
        setup_logging()
    assert _LOG_DIR is not None
    return _LOG_DIR

"""Automatic update utilities for desktop bundles."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests
from packaging import version

from ..version import get_version

logger = logging.getLogger(__name__)


def _detect_platform_key() -> str:
    if sys.platform.startswith("darwin"):
        return "macos"
    if sys.platform.startswith("win"):
        return "windows"
    return sys.platform


def _default_repository() -> Optional[str]:
    repo = os.environ.get("APP_UPDATE_REPOSITORY")
    if repo:
        return repo
    config_path = Path(__file__).resolve().parent.parent / "update_config.json"
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text("utf-8"))
            repo = payload.get("repository")
            if repo:
                return repo
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("failed to load update_config.json", exc_info=True)
    return None


def _default_channel() -> str:
    return os.environ.get("APP_UPDATE_CHANNEL", "stable")


def _update_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    target = root / "updates"
    target.mkdir(exist_ok=True)
    return target


def _app_bundle_root() -> Optional[Path]:
    exe = Path(sys.executable).resolve()
    # PyInstaller on macOS: <bundle>.app/Contents/MacOS/binary
    for _ in range(6):
        if exe.suffix == ".app":
            return exe
        exe = exe.parent
    return None


@dataclass
class ReleaseAsset:
    name: str
    download_url: str
    size: int


@dataclass
class UpdatePayload:
    version: str
    notes: str
    download: ReleaseAsset
    checksum: Optional[str] = None


def _select_asset(assets: Iterable[Dict[str, Any]], platform_key: str) -> Optional[ReleaseAsset]:
    preferred = [platform_key]
    if platform_key == "macos":
        preferred.extend(["mac", "darwin", "osx"])
    elif platform_key == "windows":
        preferred.extend(["win", "windows"])
    for keyword in preferred:
        for entry in assets:
            name = str(entry.get("name", "")).lower()
            if keyword in name and name.endswith(('.zip', '.tar.gz')):
                return ReleaseAsset(
                    name=str(entry.get("name", "")),
                    download_url=str(entry.get("browser_download_url", "")),
                    size=int(entry.get("size", 0)),
                )
    return None


def _select_checksum(assets: Iterable[Dict[str, Any]], base_name: str) -> Optional[str]:
    checksum_name = f"{base_name}.sha256"
    for entry in assets:
        if str(entry.get("name")) == checksum_name:
            url = entry.get("browser_download_url")
            if not url:
                return None
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.text.strip().split()[0]
    return None


def fetch_latest_release(repository: str) -> Optional[UpdatePayload]:
    url = f"https://api.github.com/repos/{repository}/releases/latest"
    headers = {
        "Accept": "application/vnd.github+json",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("APP_UPDATE_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.get(url, headers=headers, timeout=30)
    if response.status_code >= 400:
        logger.warning("update check failed: status=%s", response.status_code)
        return None
    payload = response.json()
    assets = payload.get("assets") or []
    platform_key = _detect_platform_key()
    asset = _select_asset(assets, platform_key)
    if not asset:
        logger.info("no asset found for platform %s", platform_key)
        return None
    checksum = _select_checksum(assets, asset.name)
    return UpdatePayload(
        version=str(payload.get("tag_name") or payload.get("name") or "0.0.0"),
        notes=str(payload.get("body") or ""),
        download=asset,
        checksum=checksum,
    )


def _compute_sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _extract_zip(archive: Path, target: Path) -> Path:
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(target)
    apps = list(target.rglob("*.app"))
    if apps:
        return apps[0]
    return target


class AutoUpdater:
    """Background updater that fetches GitHub releases."""

    def __init__(self, repository: Optional[str] = None, check_interval: int = 6 * 60 * 60):
        self.repository = repository or _default_repository()
        self.channel = _default_channel()
        self.check_interval = max(60, int(check_interval))
        self.state_lock = threading.Lock()
        self.state: Dict[str, Any] = {
            "enabled": bool(self.repository),
            "channel": self.channel,
            "current_version": get_version(),
            "update_available": False,
            "last_checked": None,
            "error": None,
            "staged_path": None,
            "pending_restart": False,
        }
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._update_root = _update_root()
        self._load_state_file()

    def _load_state_file(self) -> None:
        state_file = self._update_root / "state.json"
        if state_file.exists():
            try:
                data = json.loads(state_file.read_text("utf-8"))
                with self.state_lock:
                    self.state.update(data)
            except Exception:  # pragma: no cover - defensive
                logger.debug("failed to load update state", exc_info=True)

    def _persist_state(self) -> None:
        state_file = self._update_root / "state.json"
        with state_file.open("w", encoding="utf-8") as fh:
            json.dump(self.state, fh, indent=2, sort_keys=True)

    # Public API -----------------------------------------------------------------
    def start(self) -> None:
        if not self.state.get("enabled"):
            logger.info("auto-update disabled; missing repository")
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run_loop, name="auto-updater", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)

    def status(self) -> Dict[str, Any]:
        with self.state_lock:
            return dict(self.state)

    def check_now(self) -> Dict[str, Any]:
        self._run_check()
        return self.status()

    def apply_update(self) -> bool:
        with self.state_lock:
            staged = self.state.get("staged_path")
        if not staged:
            raise RuntimeError("no staged update to apply")
        staged_path = Path(staged)
        if sys.platform.startswith("darwin"):
            return self._apply_mac_update(staged_path)
        logger.warning("update apply not implemented for platform %s", sys.platform)
        return False

    def acknowledge_restart(self) -> None:
        with self.state_lock:
            self.state["pending_restart"] = False
            self._persist_state()

    # Internal methods -----------------------------------------------------------
    def _run_loop(self) -> None:
        logger.info("auto-update loop started (interval=%s)", self.check_interval)
        while not self._stop_event.is_set():
            try:
                self._run_check()
            except Exception:  # pragma: no cover - defensive
                logger.exception("auto-update check failed")
            if self._stop_event.wait(self.check_interval):
                break

    def _run_check(self) -> None:
        payload = None
        if self.repository:
            payload = fetch_latest_release(self.repository)
        now_ts = int(time.time())
        with self.state_lock:
            self.state["last_checked"] = now_ts
        if not payload:
            with self.state_lock:
                self.state["error"] = "no_release"
                self.state["update_available"] = False
                self._persist_state()
            return
        current = version.parse(get_version())
        remote_version = version.parse(payload.version.lstrip("v"))
        if remote_version <= current:
            with self.state_lock:
                self.state["update_available"] = False
                self.state["latest_version"] = payload.version
                self.state["error"] = None
                self._persist_state()
            return
        staged_path = self._stage_update(payload)
        with self.state_lock:
            self.state.update(
                {
                    "update_available": True,
                    "latest_version": payload.version,
                    "release_notes": payload.notes,
                    "download_size": payload.download.size,
                    "staged_path": str(staged_path) if staged_path else None,
                    "staged_version": payload.version,
                    "error": None,
                }
            )
            self._persist_state()

    def _stage_update(self, payload: UpdatePayload) -> Optional[Path]:
        target_dir = self._update_root / payload.version.replace("/", "-")
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        archive_path = target_dir / payload.download.name
        logger.info("downloading update %s -> %s", payload.version, archive_path)
        resp = requests.get(payload.download.download_url, stream=True, timeout=60)
        resp.raise_for_status()
        with archive_path.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=1024 * 512):
                if chunk:
                    fh.write(chunk)
        if payload.checksum:
            checksum = _compute_sha256(archive_path)
            if checksum.lower() != payload.checksum.lower():
                raise ValueError("checksum mismatch for update archive")
        extract_dir = target_dir / "extracted"
        extract_dir.mkdir(exist_ok=True)
        staged_app = _extract_zip(archive_path, extract_dir)
        logger.info("update staged at %s", staged_app)
        return staged_app

    def _apply_mac_update(self, staged_path: Path) -> bool:
        bundle_root = _app_bundle_root()
        if not bundle_root:
            raise RuntimeError("not running inside a macOS app bundle")
        parent = bundle_root.parent
        backup = parent / f"{bundle_root.name}.bak"
        logger.info("applying macOS update from %s", staged_path)
        try:
            if backup.exists():
                shutil.rmtree(backup)
        except Exception:
            logger.debug("failed to remove previous backup", exc_info=True)
        temp_target = parent / staged_path.name
        if temp_target.exists():
            shutil.rmtree(temp_target)
        shutil.copytree(staged_path, temp_target)
        # rename current bundle to backup and move new in place
        try:
            bundle_root.rename(backup)
            temp_target.rename(parent / bundle_root.name)
        except Exception as exc:
            logger.error("failed to swap bundles: %s", exc)
            # attempt rollback
            if not bundle_root.exists() and backup.exists():
                backup.rename(bundle_root)
            raise
        with self.state_lock:
            self.state["update_available"] = False
            self.state["staged_path"] = None
            self.state["current_version"] = (
                self.state.get("staged_version") or staged_path.name.split(".app")[0]
            )
            self.state["pending_restart"] = True
            self._persist_state()
        logger.info("update applied, restart required")
        return True


AUTO_UPDATER = AutoUpdater()

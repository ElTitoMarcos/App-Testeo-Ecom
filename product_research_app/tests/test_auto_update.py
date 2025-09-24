from __future__ import annotations

from pathlib import Path

import product_research_app.utils.auto_update as auto_update


def test_auto_updater_stages_release(tmp_path, monkeypatch):
    archive_path = tmp_path / 'mock.zip'
    with auto_update.zipfile.ZipFile(archive_path, 'w') as zf:
        zf.writestr('ProductResearchCopilot.app/Contents/Info.plist', 'plist')

    archive_bytes = archive_path.read_bytes()

    class DummyResponse:
        status_code = 200

        def iter_content(self, chunk_size=8192):
            for idx in range(0, len(archive_bytes), chunk_size):
                yield archive_bytes[idx : idx + chunk_size]

        def raise_for_status(self):  # pragma: no cover - compatibility
            return None

    monkeypatch.setattr(auto_update, "_update_root", lambda: tmp_path)
    monkeypatch.setattr(auto_update.requests, "get", lambda *args, **kwargs: DummyResponse())

    payload = auto_update.UpdatePayload(
        version='2.0.0',
        notes='changes',
        download=auto_update.ReleaseAsset(
            name='ProductResearchCopilot-macos.zip',
            download_url='https://example.com/app.zip',
            size=len(archive_bytes),
        ),
        checksum=None,
    )

    monkeypatch.setattr(auto_update, "fetch_latest_release", lambda repo: payload)

    updater = auto_update.AutoUpdater(repository='example/repo', check_interval=60)
    status = updater.check_now()
    assert status['update_available'] is True
    staged = Path(status['staged_path'])
    assert staged.exists()
    assert staged.suffix == '.app'
    assert status['staged_version'] == '2.0.0'


def test_fetch_latest_release_selects_asset(monkeypatch):
    fake_payload = {
        'tag_name': 'v3.1.4',
        'body': 'Notes',
        'assets': [
            {
                'name': 'product_research_app-windows.zip',
                'browser_download_url': 'https://example.com/win.zip',
                'size': 1024,
            },
            {
                'name': 'product_research_app-macos.zip',
                'browser_download_url': 'https://example.com/mac.zip',
                'size': 2048,
            },
        ],
    }

    class Response:
        status_code = 200

        def json(self):
            return fake_payload

    monkeypatch.setattr(auto_update.requests, "get", lambda *args, **kwargs: Response())
    monkeypatch.setattr(auto_update, "_detect_platform_key", lambda: 'macos')

    result = auto_update.fetch_latest_release('example/repo')
    assert result is not None
    assert result.download.download_url.endswith('mac.zip')

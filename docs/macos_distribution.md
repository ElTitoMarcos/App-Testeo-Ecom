# macOS distribution and update pipeline

This document describes how the macOS desktop bundle is produced, published and kept in sync with the Windows build.

## One-command build with PyInstaller

Run the helper script from the project root on macOS:

```bash
./scripts/build_mac_app.sh
```

The script performs the following steps automatically:

1. Creates an isolated virtual environment under `.venv-mac-build`.
2. Installs all Python dependencies together with `pyinstaller`.
3. Generates the standalone `.app` bundle using `product_research_app/mac_app.spec`.
4. Packages the bundle as `dist/macos/product_research_app-macos.zip` and produces the companion SHA-256 file.

The resulting archive is ready to be notarised or uploaded directly to a release.

## GitHub Actions workflow

The workflow defined in `.github/workflows/macos-app.yml` keeps the Windows and macOS versions in lockstep:

* A three-way matrix job runs `pytest` on Ubuntu, macOS and Windows to ensure feature parity across platforms.
* After all tests pass, the macOS job executes `scripts/build_mac_app.sh`, publishes the zipped bundle as a build artifact and, when the push corresponds to a tag (`v*`), uploads the assets to the corresponding GitHub Release.

The workflow runs on every push to `main`, pull request and manual dispatch.

## Automatic in-app updates

The macOS bundle embeds `product_research_app.utils.auto_update.AutoUpdater`, which polls the latest GitHub Release for the repository indicated in `product_research_app/update_config.json` (or the `APP_UPDATE_REPOSITORY` environment variable). When a release with a macOS asset is detected, the updater downloads it, verifies the optional SHA-256 checksum and stages it under `<app>/updates`. The SPA shows the update status through `/api/update/status`, allowing the user to apply the staged build and restart the app.

For GitHub-hosted builds the release assets must contain:

* `product_research_app-macos.zip`
* `product_research_app-macos.zip.sha256`

The `softprops/action-gh-release` step in the workflow publishes both artifacts automatically, which enables the in-app updater without manual intervention.

## Functional parity validation

Parity between Windows and macOS is enforced in two layers:

1. **Shared test matrix** – the same pytest suite (`product_research_app/tests`) runs on Windows, macOS and Linux on every push.
2. **Update smoke tests** – `product_research_app/tests/test_auto_update.py` validates the update staging logic, ensuring the desktop bundle behaviour remains identical across platforms.

Together these checks guarantee that both distributions expose the same features and update flows.

# -*- mode: python ; coding: utf-8 -*-

from __future__ import annotations

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

from product_research_app.version import get_version

block_cipher = None

datas = collect_data_files(
    'product_research_app',
    includes=[
        'static/**',
        'prompts/**',
        'migrations/**',
        'services/**/*.json',
        'settings/**',
        'config.example.json',
        'update_config.json',
    ],
)

hiddenimports = collect_submodules('product_research_app')

VERSION = get_version()

a = Analysis(
    ['product_research_app/desktop_entry.py'],
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ProductResearchCopilot',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='ProductResearchCopilot',
)
app = BUNDLE(
    coll,
    name='ProductResearchCopilot.app',
    bundle_identifier='com.productresearch.copilot',
    icon=None,
    info_plist={
        'CFBundleName': 'Product Research Copilot',
        'CFBundleDisplayName': 'Product Research Copilot',
        'CFBundleIdentifier': 'com.productresearch.copilot',
        'CFBundleVersion': VERSION,
        'CFBundleShortVersionString': VERSION,
        'CFBundlePackageType': 'APPL',
        'NSHighResolutionCapable': True,
    },
)

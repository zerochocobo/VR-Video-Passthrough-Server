# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs

datas = [('resources', 'resources'), ('ui\\app_metadata.json', 'ui'), ('ui\\translations', 'ui\\translations'), ('ui\\styles', 'ui\\styles')]
binaries = []
datas += collect_data_files('PySide6')
binaries += collect_dynamic_libs('PySide6')
binaries += collect_dynamic_libs('shiboken6')


a = Analysis(
    ['ui\\app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=['PySide6.QtCore', 'PySide6.QtGui', 'PySide6.QtWidgets', 'shiboken6'],
    hookspath=['packaging\\hooks'],
    hooksconfig={},
    runtime_hooks=['packaging\\runtime_hook_cuda_dlls.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VR_Video_Passthrough_Server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['G:\\GIT\\debug\\PTMediaServer\\resources\\app.ico'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='VR_Video_Passthrough_Server',
)

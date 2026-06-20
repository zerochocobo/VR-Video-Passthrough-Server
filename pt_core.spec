# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import collect_all

datas = [('resources', 'resources')]
binaries = []
hiddenimports = ['offline.convert', 'offline.two_dvr', 'tools.offline_passthrough', 'tools.offline_alpha_passthrough', 'tools.warmup_offline_trt', 'tools.generate_yoloworld_person_txt_feats', 'cupy_backends.cuda._softlink']
datas += collect_data_files('osam')
hiddenimports += collect_submodules('offline')
hiddenimports += collect_submodules('pipeline')
hiddenimports += collect_submodules('http_app')
hiddenimports += collect_submodules('dlna')
hiddenimports += collect_submodules('utils')
hiddenimports += collect_submodules('cupy_backends')
tmp_ret = collect_all('onnxruntime')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cupy')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('cupy_backends')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
tmp_ret = collect_all('pynvvideocodec')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='pt_core',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='pt_core',
)

# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files

datas = [('facial_recognition.dat', '.'), ('attendance.csv', '.'), ('company_logo/*', 'company_logo/'), ('company_logo/KFCS.ico', '.'), ('company_logo/download.jpeg', 'company_logo/')]
datas += collect_data_files('face_recognition_models')


a = Analysis(
    ['v3.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=['cv2', 'dlib'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='KFCS Pro',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['company_logo\\KFCS.ico'],
)

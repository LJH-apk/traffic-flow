# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Windows 交通流检测系统"""
import sys
from pathlib import Path

PROJ = Path(".").resolve()

# 收集仪表盘静态文件
static_files = []
static_root = PROJ / "src/dashboard/static"
for f in static_root.rglob("*"):
    if f.is_file() and "__pycache__" not in str(f):
        dest_dir = Path("src/dashboard") / f.relative_to(PROJ / "src/dashboard").parent
        static_files.append((str(f), str(dest_dir)))

# 收集 calibrations
calib_files = []
calib_root = PROJ / "src/assets/calibrations"
if calib_root.exists():
    for f in calib_root.rglob("*.json"):
        if f.is_file():
            dest_dir = Path("src/assets") / f.relative_to(PROJ / "src/assets").parent
            calib_files.append((str(f), str(dest_dir)))

a = Analysis(
    [str(PROJ / "src" / "main.py")],
    pathex=[str(PROJ)],
    binaries=[],
    datas=[*static_files, *calib_files],
    hiddenimports=[
        "scipy.interpolate", "scipy.spatial", "scipy.ndimage",
        "numpy", "PIL",
        "src.config.settings",
        "src.dashboard.server",
        "src.dashboard.build_data",
        "src.dashboard.live",
        "src.detection.detector",
        "src.trajectory.tracker",
        "src.trajectory.run_all_entrances",
        "src.cross_section.counter",
        "src.cross_section.lane_detector",
        "src.cross_section.zebra_detector",
        "src.utils.video_io",
        "src.utils.visualization",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="Traffic-Dashboard",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False, upx=True, upx_exclude=[],
    name="Traffic-Dashboard",
)

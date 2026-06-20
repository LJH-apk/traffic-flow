#!/usr/bin/env python3
"""交通流检测系统 — 数据看板独立启动器（跨平台）"""

import os
import sys

# PyInstaller bundle 路径修正
if getattr(sys, 'frozen', False):
    _bundle = sys._MEIPASS
    os.chdir(_bundle)
    if _bundle not in sys.path:
        sys.path.insert(0, _bundle)
else:
    _bundle = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, _bundle)

print("=" * 60)
print("  交通流数据检测算法 — 数据看板")
print("=" * 60)
print()

# 检查数据目录（跨平台）
def check_dir(path, name):
    if os.path.isdir(path):
        items = os.listdir(path)
        if items:
            print(f"  [OK] {name} ({len(items)} 项)")
            return True
    print(f"  [WARN] {name} 不存在或为空")
    print(f"        请确保: {path}")
    return False

base = os.getcwd()
ok1 = check_dir(os.path.join(base, "outputs"), "outputs/（输出数据）")
ok2 = check_dir(os.path.join(base, "src/assets/data"), "src/assets/data/（视频文件）")
ok3 = check_dir(os.path.join(base, "src/assets/models"), "src/assets/models/（模型权重）")

if not (ok1 and ok2 and ok3):
    print()
    print("[提示] 数据目录缺失，请将以下文件夹放入 _internal/ 目录:")
    print("         outputs/           <- 输出数据 (CSV + JSON)")
    print("         src/assets/data/   <- 视频文件 (.mp4)")
    print("         src/assets/models/ <- 模型权重 (.pt)")
    print()
print()

# 自动构建仪表盘数据
from src.dashboard.build_data import main as build_data
from pathlib import Path

dashboard_meta = Path(base) / "outputs" / "dashboard" / "meta.json"
if not dashboard_meta.exists():
    print("[INFO] 仪表盘数据不存在，正在自动构建...")
    try:
        build_data()
    except Exception as e:
        print(f"[WARN] 数据构建失败: {e}")
        print("[INFO] 将继续启动看板，但部分数据可能不可用")
else:
    print("[INFO] 仪表盘数据已就绪")
print()

# 启动服务器
from src.dashboard.server import run

print("  启动 HTTP 服务器...")
run()

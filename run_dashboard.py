#!/usr/bin/env python3
"""一键启动交通流仪表盘"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "dashboard"))

from server import run
run()

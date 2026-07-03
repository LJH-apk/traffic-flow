"""
项目工程工具：给智能体补充代码检索、配置检查、输出文件与通用 CSV 分析能力。

这些工具只读取仓库内文件，不执行任意命令；目的是让主模型在交通数据分析之外，
也能基于项目事实回答代码、配置、实验产物和表格数据问题。
"""
from __future__ import annotations

import ast
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parents[2]
_SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
_TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".html", ".css", ".js", ".csv"
}


def _safe_path(path: str | None, default: Path | None = None) -> Path:
    raw = default if not path else ROOT / path
    resolved = raw.resolve()
    if resolved != ROOT and ROOT not in resolved.parents:
        raise ValueError("路径必须位于项目目录内")
    return resolved


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in _TEXT_SUFFIXES


def _iter_files(base: Path):
    for path in base.rglob("*"):
        if any(part in _SKIP_DIRS for part in path.parts):
            continue
        if path.is_file():
            yield path


def search_project_files(query: str, path: str | None = None, limit: int = 20) -> dict:
    """在项目文本文件中检索关键词，返回文件、行号和命中片段。"""
    q = (query or "").strip()
    if not q:
        return {"error": "query 不能为空"}
    base = _safe_path(path, ROOT)
    if base.is_file():
        files = [base]
    else:
        files = list(_iter_files(base))
    hits = []
    q_lower = q.lower()
    for file in files:
        if not _is_text_file(file):
            continue
        try:
            lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        for idx, line in enumerate(lines, start=1):
            if q_lower in line.lower():
                hits.append({
                    "文件": str(file.relative_to(ROOT)),
                    "行号": idx,
                    "内容": line.strip()[:240],
                })
                if len(hits) >= max(1, min(int(limit or 20), 80)):
                    return {"query": q, "命中数": len(hits), "结果": hits}
    return {"query": q, "命中数": len(hits), "结果": hits}


def read_project_file(path: str, start_line: int | None = None,
                      max_lines: int = 160) -> dict:
    """读取项目内文本文件的部分内容。"""
    file = _safe_path(path)
    if not file.is_file():
        return {"error": "文件不存在", "path": path}
    if not _is_text_file(file):
        return {"error": "仅支持读取文本文件", "path": path}
    lines = file.read_text(encoding="utf-8", errors="ignore").splitlines()
    start = max(1, int(start_line or 1))
    count = max(1, min(int(max_lines or 160), 300))
    end = min(len(lines), start + count - 1)
    body = "\n".join(f"{i}: {lines[i - 1]}" for i in range(start, end + 1))
    return {
        "文件": str(file.relative_to(ROOT)),
        "总行数": len(lines),
        "起始行": start,
        "结束行": end,
        "内容": body,
    }


def inspect_project_config() -> dict:
    """读取智能交通项目的关键配置，便于回答模型、路径、阈值和断面设置问题。"""
    from src.config import settings as s

    return {
        "模型": {
            "主模型": s.MODEL_NAME,
            "伪GT模型": s.MODEL_NAME_GT,
            "设备": s.DEVICE,
            "置信度阈值": s.CONF_THRESH,
            "NMS_IOU阈值": s.IOU_THRESH,
        },
        "路径": {
            "视频": str(s.VIDEO_PATH.relative_to(ROOT)) if s.VIDEO_PATH.is_relative_to(ROOT) else str(s.VIDEO_PATH),
            "输出目录": str(s.OUTPUT_DIR.relative_to(ROOT)) if s.OUTPUT_DIR.is_relative_to(ROOT) else str(s.OUTPUT_DIR),
            "轨迹CSV": str(s.TRAJ_CSV_PATH.relative_to(ROOT)),
            "断面CSV": str(s.CROSS_SECTION_CSV_PATH.relative_to(ROOT)),
            "智能体报告": str(s.AGENT_REPORT_PATH.relative_to(ROOT)),
        },
        "交通参数": {
            "轨迹采样FPS": s.TRAJ_SAMPLE_FPS,
            "兜底像素每米": s.PIXELS_PER_METER,
            "排队速度阈值_kmh": s.QUEUE_SPEED_THRESH_KMH,
            "有效速度上限_kmh": s.SPEED_VALID_MAX_KMH,
            "危险车头时距_s": s.HEADWAY_DANGEROUS_S,
            "饱和流率_pcu每小时每车道": s.SATURATION_FLOW_PCU_H,
        },
        "断面线": {
            entrance: [line[0] for line in lines]
            for entrance, lines in s.SECTION_LINES_MAP.items()
        },
    }


def list_output_files(limit: int = 80) -> dict:
    """列出 outputs 下的主要产物，帮助智能体判断有哪些实验结果可用。"""
    from src.config.settings import OUTPUT_DIR

    out = OUTPUT_DIR.resolve()
    if not out.exists():
        return {"data_ready": False, "error": "outputs 目录不存在"}
    rows = []
    for file in sorted(_iter_files(out), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            stat = file.stat()
        except OSError:
            continue
        rows.append({
            "文件": str(file.relative_to(ROOT)),
            "大小_KB": round(stat.st_size / 1024, 1),
            "后缀": file.suffix.lower() or "(无)",
        })
        if len(rows) >= max(1, min(int(limit or 80), 200)):
            break
    return {"data_ready": True, "文件数": len(rows), "结果": rows}


def analyze_csv_file(path: str, limit_columns: int = 30) -> dict:
    """通用 CSV 体检：行列数、字段缺失、数值范围、类别分布。"""
    file = _safe_path(path)
    if not file.is_file():
        return {"error": "CSV 文件不存在", "path": path}
    if file.suffix.lower() != ".csv":
        return {"error": "仅支持 CSV 文件", "path": path}

    with file.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    col_limit = max(1, min(int(limit_columns or 30), 80))
    profile = {}
    for col in fieldnames[:col_limit]:
        values = [(r.get(col) or "").strip() for r in rows]
        missing = sum(1 for v in values if v == "")
        nums = []
        for v in values:
            if not v:
                continue
            try:
                nums.append(float(v))
            except ValueError:
                pass
        if nums and len(nums) >= max(3, len(values) * 0.6):
            nums_sorted = sorted(nums)
            profile[col] = {
                "类型": "数值",
                "缺失数": missing,
                "最小值": round(nums_sorted[0], 3),
                "最大值": round(nums_sorted[-1], 3),
                "均值": round(sum(nums_sorted) / len(nums_sorted), 3),
            }
        else:
            counts = Counter(v for v in values if v)
            profile[col] = {
                "类型": "类别/文本",
                "缺失数": missing,
                "唯一值数": len(counts),
                "高频值": dict(counts.most_common(8)),
            }

    return {
        "文件": str(file.relative_to(ROOT)),
        "行数": len(rows),
        "列数": len(fieldnames),
        "字段": fieldnames,
        "字段画像": profile,
        "样例": rows[:5],
    }


def summarize_python_module(path: str) -> dict:
    """静态解析 Python 文件，返回类、函数和导入概览。"""
    file = _safe_path(path)
    if not file.is_file() or file.suffix != ".py":
        return {"error": "仅支持项目内 Python 文件", "path": path}
    source = file.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(source)
    classes = []
    functions = []
    imports = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes.append({"名称": node.name, "行号": node.lineno,
                            "方法": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append({"名称": node.name, "行号": node.lineno})
        elif isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            imports.extend(f"{mod}.{alias.name}" if mod else alias.name for alias in node.names)
    return {
        "文件": str(file.relative_to(ROOT)),
        "类": classes,
        "顶层函数": functions,
        "导入": imports[:80],
    }

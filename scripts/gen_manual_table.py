"""生成人工标注过车事件三线表 LaTeX + PDF"""
import csv, subprocess, os

ROOT = "/Users/liujiahang/科研/交通流算法"

with open(os.path.join(ROOT, "annotations/manual_crossing_clean.csv"), "r") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = [next(reader) for _ in range(10)]

cn_header = {
    "entrance": "进口", "video_name": "视频文件", "gt_frame_id": "帧ID",
    "section": "断面", "direction": "方向", "class_name": "类别",
    "lane_id": "车道", "track_id": "跟踪ID", "vehicle_label": "车辆类型", "note": "备注"
}
header_cn = [cn_header.get(h, h) for h in header]

def cn_val(col, val):
    if col == "entrance":
        return val.replace("北进口", "北").replace("南进口", "南").replace("东进口", "东")
    if col == "class_name":
        return {"car": "小汽车", "bus": "公交车", "truck": "货车",
                "non_motor": "非机动车", "motorcycle": "摩托车", "bicycle": "自行车"}.get(val, val)
    if col == "section":
        return val.replace("北进口", "").replace("东进口", "").replace("南进口", "")
    if col == "video_name":
        parts = val.split("_", 1)
        return parts[0] + "_…" if len(parts) > 1 else val[:12] + "…"
    if col in ("vehicle_label", "note"):
        return val if val else "—"
    return val

# 构建 LaTeX body 行
body_lines = []
for row in rows:
    mapped = [cn_val(header[i], row[i]) for i in range(len(header))]
    body_lines.append(" & ".join(str(v) for v in mapped) + r" \\")

tex_content = r"""\documentclass[12pt,a4paper]{ctexart}
\usepackage{booktabs}
\usepackage[left=1.8cm,right=1.8cm,top=2cm,bottom=2cm]{geometry}
\usepackage{caption}

\begin{document}

\begin{table}[h]
\centering
\caption{人工标注过车事件示例（前10条）}
\label{tab:manual_crossing}
\footnotesize
\begin{tabular}{cccccccccc}
\toprule
""" + " & ".join(header_cn) + r" \\" + "\n" + r"\midrule" + "\n" + \
"\n".join(body_lines) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}

\end{document}
"""

tex_path = os.path.join(ROOT, "outputs/manual_crossing_sample.tex")
with open(tex_path, "w") as f:
    f.write(tex_content)

# 编译 PDF
out_dir = os.path.join(ROOT, "outputs")
for _ in range(2):
    subprocess.run(
        ["xelatex", "-interaction=nonstopmode", "-output-directory=" + out_dir, tex_path],
        capture_output=True, cwd=out_dir
    )

# 清理
for ext in [".aux", ".log", ".out"]:
    tmp = os.path.join(out_dir, "manual_crossing_sample" + ext)
    if os.path.exists(tmp):
        os.unlink(tmp)

print("PDF 已生成: outputs/manual_crossing_sample.pdf")

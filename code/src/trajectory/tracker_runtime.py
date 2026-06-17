"""跟踪任务运行期辅助逻辑。"""
from __future__ import annotations

import time
from pathlib import Path

from src.config.settings import ENTRANCE_ALIASES, SECTION_LINES
from src.cross_section.section_calibration import load_all_section_lines, load_section_lines


class TrackerOutputLock:
    """跨进程输出锁，避免多个检测任务同时写 outputs。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._fh = None

    def __enter__(self):
        import fcntl

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", encoding="utf-8")
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._fh.close()
            self._fh = None
            raise RuntimeError("已有检测任务正在写 outputs，请先停止 dashboard 检测或等待完成") from exc
        self._fh.write(f"{time.time():.3f}\n")
        self._fh.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        import fcntl

        if self._fh is None:
            return
        fcntl.flock(self._fh, fcntl.LOCK_UN)
        self._fh.close()
        self._fh = None

    def __del__(self) -> None:
        self.__exit__(None, None, None)


def resolve_section_lines(video_path) -> list:
    stem = Path(video_path).stem
    for alias, canonical in ENTRANCE_ALIASES.items():
        if alias in stem:
            lines = load_section_lines(canonical)
            if lines:
                print(f"[断面] 识别到进口：{canonical}，加载 {len(lines)} 条断面线")
                return lines
    lines = load_all_section_lines() or SECTION_LINES
    print(f"[断面] 未识别进口名（文件：{Path(video_path).name}），加载全部 {len(lines)} 条断面线")
    return lines

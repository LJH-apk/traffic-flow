import ast
from pathlib import Path


TRACKER_PATH = Path("src/trajectory/tracker.py")


def _run_function():
    tree = ast.parse(TRACKER_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "run":
            return node
    raise AssertionError("TrajectoryTracker.run not found")


def test_tracker_run_accepts_live_publisher_and_stop_event():
    run_fn = _run_function()
    arg_names = [arg.arg for arg in run_fn.args.args + run_fn.args.kwonlyargs]

    assert "live_publisher" in arg_names
    assert "stop_event" in arg_names


def test_tracker_run_publishes_live_frames_stats_and_progress():
    source = TRACKER_PATH.read_text(encoding="utf-8")

    assert "publish_frame" in source
    assert "publish_stats" in source
    assert "update_progress" in source
    assert "stop_event.is_set()" in source


def test_tracker_live_preview_targets_smoother_frame_rate():
    source = TRACKER_PATH.read_text(encoding="utf-8")

    assert "_LIVE_PREVIEW_FPS = 18.0" in source
    assert "round(video_fps / _LIVE_PREVIEW_FPS)" in source

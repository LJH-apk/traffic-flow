import threading
import time

from dashboard.live import LiveState
from dashboard.server import DashboardApp


class BlockingTracker:
    def __init__(self) -> None:
        self.called = threading.Event()
        self.stop_event = None
        self.publisher = None

    def run(self, video_path, live_publisher=None, stop_event=None):
        self.publisher = live_publisher
        self.stop_event = stop_event
        self.called.set()
        stop_event.wait(timeout=2)


def test_dashboard_app_starts_fixed_video_once_and_stops():
    tracker = BlockingTracker()
    state = LiveState()
    app = DashboardApp(
        state=state,
        tracker_factory=lambda: tracker,
        video_path="fixed.mp4",
    )

    first = app.start_detection()
    assert first["started"] is True
    assert tracker.called.wait(timeout=1)

    second = app.start_detection()
    assert second["started"] is False
    assert "already" in second["message"].lower()

    stopped = app.stop_detection()
    assert stopped["stop_requested"] is True
    assert tracker.stop_event is state.stop_event
    assert tracker.publisher is state
    assert state.stop_event.is_set()

    app.join(timeout=2)
    snapshot = state.snapshot()
    assert snapshot["running"] is False
    assert snapshot["video"] == "fixed.mp4"


def test_dashboard_app_reports_tracker_errors():
    def factory():
        class FailingTracker:
            def run(self, video_path, live_publisher=None, stop_event=None):
                raise RuntimeError("tracker exploded")
        return FailingTracker()

    state = LiveState()
    app = DashboardApp(state=state, tracker_factory=factory, video_path="fixed.mp4")

    assert app.start_detection()["started"] is True
    deadline = time.time() + 2
    while time.time() < deadline and state.snapshot()["running"]:
        time.sleep(0.01)

    snapshot = state.snapshot()
    assert snapshot["running"] is False
    assert "tracker exploded" in snapshot["error"]

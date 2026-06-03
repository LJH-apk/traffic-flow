import threading

import numpy as np

from dashboard.live import LiveState


def test_live_state_tracks_status_stats_and_stop_signal():
    state = LiveState()

    state.start("demo.mp4", total_frames=300)
    state.publish_stats({"vehicles": 4, "events": 2, "avg_speed": 31.5})
    state.update_progress(frame_idx=30, timestamp_s=1.2, fps=16.7)
    state.request_stop()

    snapshot = state.snapshot()
    assert snapshot["running"] is True
    assert snapshot["stop_requested"] is True
    assert snapshot["video"] == "demo.mp4"
    assert snapshot["progress"]["frame_idx"] == 30
    assert snapshot["stats"]["vehicles"] == 4
    assert state.stop_event.is_set()


def test_live_state_keeps_latest_encoded_frame_only():
    state = LiveState()
    first = np.zeros((20, 40, 3), dtype=np.uint8)
    second = np.full((20, 40, 3), 255, dtype=np.uint8)

    assert state.publish_frame(first, width=32, quality=65) is True
    first_bytes = state.latest_frame()
    assert first_bytes is not None
    assert first_bytes.startswith(b"\xff\xd8")

    assert state.publish_frame(second, width=32, quality=65) is True
    second_bytes = state.latest_frame()
    assert second_bytes is not None
    assert second_bytes.startswith(b"\xff\xd8")
    assert second_bytes != first_bytes


def test_live_state_stream_waits_until_a_frame_is_available():
    state = LiveState()
    chunks = []

    def consume_one():
        chunks.append(next(state.mjpeg_chunks(timeout=1.0)))

    thread = threading.Thread(target=consume_one)
    thread.start()
    assert state.publish_frame(np.zeros((10, 10, 3), dtype=np.uint8)) is True
    thread.join(timeout=2)

    assert chunks
    assert b"Content-Type: image/jpeg" in chunks[0]

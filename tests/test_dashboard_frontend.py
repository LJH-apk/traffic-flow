from pathlib import Path


INDEX = Path("dashboard/index.html")


def test_dashboard_frontend_has_live_controls_and_stream():
    html = INDEX.read_text(encoding="utf-8")

    assert "/api/detect/start" in html
    assert "/api/detect/stop" in html
    assert "/api/live/status" in html
    assert "/api/live/stats" in html
    assert "/api/live/stream.mjpg" in html
    assert "/outputs/trajectory.mp4" in html
    assert 'id="startBtn"' in html
    assert 'id="stopBtn"' in html


def test_dashboard_frontend_uses_big_screen_layout_labels():
    html = INDEX.read_text(encoding="utf-8")

    assert "交通流数据可视化监管大屏" in html
    assert "实时预览" in html
    assert "累计车辆" in html
    assert "最近过车事件" in html
    assert "车道流量" in html


def test_dashboard_frontend_has_container_hud_not_detection_redraw():
    html = INDEX.read_text(encoding="utf-8")

    assert 'id="hudLayer"' in html
    assert 'class="hud-top"' in html
    assert 'id="hudFrameVal"' in html
    assert 'id="hudProgressVal"' in html
    assert "<canvas" not in html
    assert "bbox-overlay" not in html
    assert "drawBoundingBox" not in html


def test_dashboard_frontend_has_metric_animation_and_event_highlight():
    html = INDEX.read_text(encoding="utf-8")

    assert "function animateNumber" in html
    assert "function flashPanel" in html
    assert "event--fresh" in html
    assert "data-metric=" in html

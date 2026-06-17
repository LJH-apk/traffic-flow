from pathlib import Path


INDEX = Path("src/dashboard/static/index.html")
LIVE_VIEW = Path("src/dashboard/static/assets/views/live.js")
DEMO_VIEW = Path("src/dashboard/static/assets/views/demo.js")
DASHBOARD_VIEW = Path("src/dashboard/static/assets/views/dashboard.js")
ANIMATE_UTIL = Path("src/dashboard/static/assets/utils/animate.js")


def test_dashboard_frontend_has_live_controls_and_stream():
    html = INDEX.read_text(encoding="utf-8")
    live_js = LIVE_VIEW.read_text(encoding="utf-8")
    dashboard_js = DASHBOARD_VIEW.read_text(encoding="utf-8")

    assert "/dashboard/assets/views/live.js" in html
    assert "/api/live/status" in live_js
    assert "/api/live/stats" in live_js
    assert "/api/live/stream.mjpg" in live_js
    assert "/api/detect/rebuild" in dashboard_js
    assert 'id="liveMjpeg"' in live_js
    assert 'id="liveStatus"' in live_js


def test_dashboard_frontend_uses_big_screen_layout_labels():
    html = INDEX.read_text(encoding="utf-8")
    live_js = LIVE_VIEW.read_text(encoding="utf-8")
    demo_js = DEMO_VIEW.read_text(encoding="utf-8")
    dashboard_js = DASHBOARD_VIEW.read_text(encoding="utf-8")

    assert "交通流检测平台" in html
    assert "实时检测" in html
    assert "累计车辆" in live_js
    assert "最近过车事件" in dashboard_js
    assert "交通流检测系统" in demo_js


def test_dashboard_frontend_has_container_hud_not_detection_redraw():
    live_js = LIVE_VIEW.read_text(encoding="utf-8")
    demo_js = DEMO_VIEW.read_text(encoding="utf-8")

    assert 'id="liveHudTop"' in live_js
    assert 'id="liveHudBottom"' in live_js
    assert 'id="liveFrame"' in live_js
    assert 'id="liveProgress"' in live_js
    assert 'class="demo-video"' in demo_js
    assert "bbox-overlay" not in live_js
    assert "drawBoundingBox" not in live_js


def test_dashboard_frontend_has_metric_animation_and_event_highlight():
    live_js = LIVE_VIEW.read_text(encoding="utf-8")
    demo_js = DEMO_VIEW.read_text(encoding="utf-8")
    dashboard_js = DASHBOARD_VIEW.read_text(encoding="utf-8")
    animate_js = ANIMATE_UTIL.read_text(encoding="utf-8")

    assert "function interpolate" in animate_js
    assert "function pulse" in animate_js
    assert "function animateKPI" in live_js
    assert "function animateKPI" in dashboard_js
    assert "ov-event-pill" in demo_js

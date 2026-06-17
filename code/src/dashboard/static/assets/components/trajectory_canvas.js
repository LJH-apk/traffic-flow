/* ═══════════════════════════════════════════════════════════════════
   components/trajectory_canvas.js — Canvas 轨迹路径图
   ═══════════════════════════════════════════════════════════════════ */
window.TrajectoryCanvas = (() => {

  let _ctx = null;
  let _canvas = null;
  let _w = 0;
  let _h = 0;

  function init(canvasId) {
    _canvas = document.getElementById(canvasId);
    if (!_canvas) return;
    _ctx = _canvas.getContext('2d');
  }

  function render(data, meta) {
    if (!_ctx || !_canvas) return;
    const rect = _canvas.parentElement.getBoundingClientRect();
    _w = rect.width;
    _h = rect.height;
    _canvas.width = _w * (window.devicePixelRatio || 1);
    _canvas.height = _h * (window.devicePixelRatio || 1);
    _canvas.style.width = _w + 'px';
    _canvas.style.height = _h + 'px';
    _ctx.setTransform((window.devicePixelRatio || 1), 0, 0, (window.devicePixelRatio || 1), 0, 0);

    const srcW = data.canvas ? data.canvas.width : 3840;
    const srcH = data.canvas ? data.canvas.height : 2160;
    const scaleX = _w / srcW;
    const scaleY = _h / srcH;
    const scale = Math.min(scaleX, scaleY);

    // Off-center to match video
    const offsetX = (_w - srcW * scale) / 2;
    const offsetY = (_h - srcH * scale) / 2;

    // Background
    _ctx.fillStyle = '#020617';
    _ctx.fillRect(0, 0, _w, _h);

    // Draw tracks
    const tracks = data.tracks || [];
    tracks.forEach(track => {
      const pts = track.points || [];
      if (pts.length < 2) return;
      const color = CMap.forClass(track.class_name);
      _ctx.strokeStyle = color;
      _ctx.lineWidth = 1.2;
      _ctx.globalAlpha = 0.55;
      _ctx.beginPath();
      _ctx.moveTo(offsetX + pts[0][0] * scale, offsetY + pts[0][1] * scale);
      for (let i = 1; i < pts.length; i++) {
        _ctx.lineTo(offsetX + pts[i][0] * scale, offsetY + pts[i][1] * scale);
      }
      _ctx.stroke();
    });
    _ctx.globalAlpha = 1;
  }

  function destroy() {
    _ctx = null;
    _canvas = null;
  }

  return { init, render, destroy };

})();

/* ═══════════════════════════════════════════════════════════════════
   views/live.js — 实时模式：MJPEG 画面 + 轮询统计
   ═══════════════════════════════════════════════════════════════════ */
window.LiveView = (() => {

  let _pollTimer = null;
  let _prevStats = null;

  function render() {
    const container = document.getElementById('viewContainer');
    if (!container) return;

    container.innerHTML = `
      <div class="live-stage" id="liveStage">
        <!-- 实时画面 -->
        <div class="live-screen" id="liveScreen">
          <img class="live-mjpeg" id="liveMjpeg" alt="实时检测画面" />
          <div class="live-empty" id="liveEmpty">等待检测启动…</div>

          <!-- HUD 浮层 -->
          <div class="live-hud-top" id="liveHudTop">
            <span class="live-pill"><span class="live-dot"></span> LIVE</span>
            <span class="live-pill">FPS <b id="liveFps">0.0</b></span>
            <span class="live-pill">FRAME <b id="liveFrame">0</b></span>
          </div>
          <div class="live-hud-bottom" id="liveHudBottom">
            <span class="live-pill">PROGRESS <b id="liveProgress">0%</b></span>
          </div>
        </div>

        <!-- 右侧统计 -->
        <div class="live-sidebar">
          <div class="live-stat accent-cyan">
            <div class="live-stat-label">累计车辆</div>
            <div class="live-stat-val" id="liveVehicles">0</div>
          </div>
          <div class="live-stat accent-amber">
            <div class="live-stat-label">过车事件</div>
            <div class="live-stat-val" id="liveEvents">0</div>
          </div>
          <div class="live-stat accent-emerald">
            <div class="live-stat-label">实时均速</div>
            <div class="live-stat-val" id="liveSpeed">0.0 <small>km/h</small></div>
          </div>
          <div class="live-stat accent-blue">
            <div class="live-stat-label">活跃目标</div>
            <div class="live-stat-val" id="liveActive">0</div>
          </div>
          <div class="live-stat">
            <div class="live-stat-label">检测状态</div>
            <div class="live-stat-val" id="liveStatus" style="font-size:14px;">待机</div>
          </div>
          <div class="live-classes" id="liveClasses"></div>
        </div>
      </div>
    `;

    startPolling();
  }

  function startPolling() {
    // 启动 MJPEG 流
    const img = document.getElementById('liveMjpeg');
    if (img) {
      img.src = '/api/live/stream.mjpg?ts=' + Date.now();
      img.onload = function () {
        const empty = document.getElementById('liveEmpty');
        if (empty) empty.style.display = 'none';
      };
      img.onerror = function () {
        const empty = document.getElementById('liveEmpty');
        if (empty) empty.innerHTML = '<span>⚠ 检测未启动或 MJPEG 流不可用</span>';
      };
    }

    // 轮询状态（每 800ms）
    _pollTimer = setInterval(poll, 800);
    poll(); // 立即拉一次
  }

  async function poll() {
    try {
      const [snapRes, statsRes] = await Promise.all([
        fetch('/api/live/status').then(r => r.json()).catch(() => null),
        fetch('/api/live/stats').then(r => r.json()).catch(() => null),
      ]);

      if (snapRes) applyStatus(snapRes);
      if (statsRes) applyStats(statsRes);
    } catch (e) {
      // 静默处理
    }
  }

  function applyStatus(snap) {
    const running = !!snap.running;
    const empty = document.getElementById('liveEmpty');
    if (empty && running) empty.style.display = 'none';

    const el = (id, text) => { const e = document.getElementById(id); if (e) e.textContent = text; };
    el('liveFps', (snap.progress?.fps || 0).toFixed(1));
    el('liveFrame', snap.progress?.frame_idx || 0);
    el('liveProgress', (snap.progress?.percent || 0).toFixed(1) + '%');

    let statusText = '待机';
    if (running) statusText = snap.stop_requested ? '停止中…' : '运行中';
    else if (snap.error) statusText = '异常: ' + snap.error;
    else if (snap.finished_at) statusText = '已完成';
    el('liveStatus', statusText);
  }

  function applyStats(stats) {
    const prev = _prevStats || {};
    _prevStats = stats;

    animateKPI('liveVehicles', stats.vehicles || 0, 0);
    animateKPI('liveEvents', stats.events || 0, 0);
    animateKPI('liveSpeed', stats.avg_speed || 0, 1);
    animateKPI('liveActive', stats.active_tracks || 0, 0);

    // 车型分布
    renderClasses(stats.class_counts);
  }

  function animateKPI(id, target, digits) {
    const el = document.getElementById(id);
    if (!el) return;
    const curText = el.textContent.replace(' km/h', '').replace(/,/g, '');
    const from = parseFloat(curText) || 0;
    const to = Number(target) || 0;
    if (from === to) return;

    const start = performance.now();
    const dur = 350;
    function step(now) {
      const p = Math.min(1, (now - start) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      const val = from + (to - from) * e;
      const text = digits ? val.toFixed(digits) : Math.round(val).toLocaleString('zh-CN');
      if (id === 'liveSpeed') {
        el.innerHTML = text + ' <small>km/h</small>';
      } else {
        el.textContent = text;
      }
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function renderClasses(counts) {
    const container = document.getElementById('liveClasses');
    if (!container) return;
    if (!counts || !Object.keys(counts).length) {
      container.innerHTML = '';
      return;
    }
    const max = Math.max(1, ...Object.values(counts));
    container.innerHTML = Object.entries(counts).map(([name, v]) => {
      const pct = Math.round((v / max) * 100);
      const color = CMap.forClass(name);
      return `<div class="live-class-bar">
        <span class="live-class-name">${name}</span>
        <span class="live-class-track"><span class="live-class-fill" style="width:${pct}%;background:${color};"></span></span>
        <span class="live-class-val">${v}</span>
      </div>`;
    }).join('');
  }

  function destroy() {
    if (_pollTimer) clearInterval(_pollTimer);
    _pollTimer = null;
    _prevStats = null;
  }

  return { render, destroy };

})();

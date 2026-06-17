/* ═══════════════════════════════════════════════════════════════════
   views/demo.js — 演示模式：全屏视频 + 四角浮层 + 时间同步
   ═══════════════════════════════════════════════════════════════════ */
window.DemoView = (() => {

  let _video = null;
  let _animFrame = null;
  let _lastSnapIdx = -1;

  function render() {
    const container = document.getElementById('viewContainer');
    if (!container) return;

    container.innerHTML = `
      <div class="demo-stage" id="demoStage">
        <video class="demo-video" id="demoVideo"
          src="${CONST.VIDEO_URL}" preload="auto"
          autoplay playsinline muted loop></video>

        <div class="demo-empty" id="demoEmpty">
          <span>⬡ 加载检测数据中…</span>
          <small>请稍候</small>
        </div>

        <!-- Corner accents -->
        <div class="corner corner-tl"></div>
        <div class="corner corner-tr"></div>
        <div class="corner corner-bl"></div>
        <div class="corner corner-br"></div>

        <!-- Overlay: top-left -->
        <div class="overlay ov-tl" id="ovTL">
          <div class="ov-sys">
            <span>⬡ 交通流检测系统</span>
            <span class="live-dot"></span>
            <span style="color:#22c55e;font-size:10px;font-weight:500">LIVE</span>
          </div>
          <div class="ov-ts" id="ovTime">00:00 / 00:00</div>
        </div>

        <!-- Overlay: top-right -->
        <div class="overlay ov-tr" id="ovTR">
          <div class="ov-label">累计车辆</div>
          <div class="ov-val" id="ovVehicles">0</div>
          <div class="ov-sub">活跃 <span id="ovActive">0</span> 个目标</div>
        </div>

        <!-- Overlay: bottom-left -->
        <div class="overlay ov-bl" id="ovBL">
          <div class="ov-label">实时均速</div>
          <div class="ov-row">
            <span class="ov-val" id="ovSpeed">0.0</span>
            <span class="ov-unit">km/h</span>
          </div>
        </div>

        <!-- Overlay: bottom-right -->
        <div class="overlay ov-br" id="ovBR">
          <div class="ov-label">过车事件</div>
          <div class="ov-val" id="ovEvents">0</div>
          <div class="ov-sub">累计断面过车</div>
        </div>

        <!-- Bottom progress bar -->
        <div class="ov-progress">
          <div class="ov-fill" id="ovProgressFill" style="width:0%"></div>
        </div>

        <!-- Events strip -->
        <div class="ov-events-strip" id="ovEventsStrip"></div>
      </div>
    `;

    _video = document.getElementById('demoVideo');
    _video.addEventListener('loadedmetadata', onVideoReady);
    _video.addEventListener('timeupdate', onTimeUpdate);
    _video.addEventListener('playing', () => {
      document.getElementById('demoEmpty').style.display = 'none';
    });
    _video.addEventListener('waiting', () => {});

    loadAndSync();
  }

  async function loadAndSync() {
    try {
      const { meta, timeline } = await API.loadDemoData();
      Store.set('meta', meta);
      Store.set('timeline', timeline);
      Store.set('demoLoaded', true);

      // 更新顶栏
      updateTopBar(meta);

      // 更新总时长显示
      const duration = meta.video.duration_s;
      const timeEl = document.getElementById('ovTime');
      if (timeEl) timeEl.textContent = '00:00 / ' + FMT.time(duration);

      // 隐藏 loading
      const emptyEl = document.getElementById('demoEmpty');
      if (emptyEl) emptyEl.style.display = 'none';

      // 如果视频已可播放，手动触发首次同步
      if (_video && _video.readyState >= 2) {
        syncSnapshot();
      }
    } catch (err) {
      console.error('Demo data load failed:', err);
      const emptyEl = document.getElementById('demoEmpty');
      if (emptyEl) emptyEl.innerHTML = '<span>⚠ 数据加载失败</span><small>' + err.message + '</small>';
    }
  }

  function onVideoReady() {
    const emptyEl = document.getElementById('demoEmpty');
    if (emptyEl && Store.get('demoLoaded')) emptyEl.style.display = 'none';
    syncSnapshot();
  }

  function onTimeUpdate() {
    if (!Store.get('demoLoaded')) return;
    syncSnapshot();
  }

  function syncSnapshot() {
    if (!_video) return;
    const timeline = Store.get('timeline');
    if (!timeline || !timeline.snapshots) return;

    const t = Math.floor(_video.currentTime);
    const snaps = timeline.snapshots;
    const idx = Math.min(t, snaps.length - 1);
    if (idx < 0 || idx === _lastSnapIdx) return;
    _lastSnapIdx = idx;

    const snap = snaps[idx];
    Store.set('demoCurrentSec', t);

    // 更新顶栏时间
    const timeEl = document.getElementById('topbarTime');
    if (timeEl) timeEl.textContent = FMT.time(t);

    // 更新四角浮层
    const vehiclesEl = document.getElementById('ovVehicles');
    if (vehiclesEl) Animate.interpolate(vehiclesEl, snap.cumulative_vehicles, { digits: 0 });

    const activeEl = document.getElementById('ovActive');
    if (activeEl) activeEl.textContent = FMT.number(snap.active_tracks);

    const speedEl = document.getElementById('ovSpeed');
    if (speedEl) Animate.interpolate(speedEl, snap.avg_speed_kmh, { digits: 1 });

    const eventsEl = document.getElementById('ovEvents');
    if (eventsEl) Animate.interpolate(eventsEl, snap.cumulative_events, { digits: 0 });

    // 更新主时间
    const ovTimeEl = document.getElementById('ovTime');
    if (ovTimeEl) {
      const meta = Store.get('meta');
      const dur = meta ? meta.video.duration_s : 0;
      ovTimeEl.textContent = FMT.time(t) + ' / ' + FMT.time(dur);
    }

    // 更新底部进度条
    const progressFill = document.getElementById('ovProgressFill');
    if (progressFill && timeline.total_seconds > 0) {
      const pct = Math.min(100, (t / timeline.total_seconds) * 100);
      progressFill.style.width = pct + '%';
    }

    // 更新事件滚动条
    updateEventsStrip(snap);
  }

  function updateEventsStrip(snap) {
    const strip = document.getElementById('ovEventsStrip');
    if (!strip) return;
    const events = snap.recent_events || [];
    if (!events.length) { strip.innerHTML = ''; return; }

    strip.innerHTML = events.slice(0, 4).map(ev => {
      const cls = ev.class_name || '';
      const dir = ev.direction || '';
      const spd = FMT.speed(ev.speed_kmh) + 'km/h';
      return `<span class="ov-event-pill">${cls} ${dir} ${spd}</span>`;
    }).join('');
  }

  function updateTopBar(meta) {
    const infoEl = document.getElementById('topbarInfo');
    if (!infoEl) return;
    const name = meta && meta.video ? meta.video.name : '';
    infoEl.innerHTML = '<span class="topbar-time" id="topbarTime">00:00</span>';
  }

  function destroy() {
    if (_animFrame) cancelAnimationFrame(_animFrame);
    if (_video) {
      _video.removeEventListener('loadedmetadata', onVideoReady);
      _video.removeEventListener('timeupdate', onTimeUpdate);
      _video.pause();
      _video = null;
    }
    _lastSnapIdx = -1;
    ChartFactory.disposeAll();
    TrajectoryCanvas.destroy();
  }

  return { render, destroy };

})();

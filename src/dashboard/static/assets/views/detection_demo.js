/* 10.1 实时检测演示：视频（北进口）+ 右侧栏 */
window.DetectionDemoView = (() => {

  let _video = null, _lastSnapIdx = -1;
  const _scope = 'north';

  function render() {
    const container = document.getElementById('viewContainer');
    if (!container) return;
    container.innerHTML = `
      <div class="dd-stage">
        <div class="dd-video-wrap">
          <video class="dd-video" id="ddVideo"
            src="${CONST.VIDEO_URLS.north}" preload="auto"
            autoplay playsinline muted="muted" loop></video>
          <div class="dd-empty" id="ddEmpty" onclick="var v=document.getElementById('ddVideo');if(v)v.play();this.style.display='none';" style="cursor:pointer;">
            ⬡ 加载检测数据中…<br><small>如视频未自动播放，请点击此处</small></div>
        </div>
        <div class="dd-sidebar">
          <div class="dd-panel">
            <h4>实时检测统计</h4>
            <div class="dd-stat-row"><span class="dd-label">活跃目标</span><span class="dd-value" id="ddActive">0</span></div>
            <div class="dd-stat-row"><span class="dd-label">当前帧</span><span class="dd-value" id="ddFrame">0</span></div>
            <div class="dd-stat-row"><span class="dd-label">当前时间</span><span class="dd-value" id="ddTime">00:00</span></div>
          </div>
          <div class="dd-panel">
            <h4>车型分布（当前帧）</h4>
            <div id="ddClassBars"></div>
          </div>
          <div class="dd-panel">
            <h4>累计统计</h4>
            <div class="dd-stat-row"><span class="dd-label">累计车辆</span><span class="dd-value" id="ddVehicles">--</span></div>
            <div class="dd-stat-row"><span class="dd-label">过车事件</span><span class="dd-value" id="ddEvents">--</span></div>
            <div class="dd-stat-row"><span class="dd-label">平均速度</span><span class="dd-value" id="ddSpeed">--</span></div>
          </div>
          <div class="dd-panel dd-confidence">
            <h4>检测性能</h4>
            <div class="dd-stat-row"><span class="dd-label">数据总时长</span><span class="dd-value" id="ddDuration">--</span></div>
            <div class="dd-stat-row"><span class="dd-label">轨迹总数</span><span class="dd-value" id="ddTracks">--</span></div>
          </div>
        </div>
      </div>`;

    _video = document.getElementById('ddVideo');
    _video.addEventListener('loadedmetadata', () => {
      document.getElementById('ddEmpty').style.display = 'none';
      syncSnapshot();
    });
    _video.addEventListener('timeupdate', onTimeUpdate);
    _video.addEventListener('playing', () => {
      document.getElementById('ddEmpty').style.display = 'none';
    });
    loadData();
  }

  async function loadData() {
    try {
      const [meta, timeline, overview] = await Promise.all([
        API.loadMeta(_scope), API.loadTimeline(_scope), API.loadOverview(_scope)
      ]);
      Store.set('meta', meta); Store.set('timeline', timeline); Store.set('overview', overview);
      CMap.init(meta);
      document.getElementById('ddDuration').textContent = FMT.time(overview.active_duration_s);
      document.getElementById('ddTracks').textContent = FMT.number(overview.unique_tracks);
      document.getElementById('ddEmpty').style.display = 'none';
      if (_video && _video.readyState >= 2) syncSnapshot();
    } catch (e) {
      console.error(e);
      document.getElementById('ddEmpty').innerHTML = '<span>⚠ 数据加载失败</span>';
    }
  }

  function onTimeUpdate() { if (Store.get('timeline')) syncSnapshot(); }

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
    document.getElementById('ddActive').textContent = FMT.number(snap.active_tracks);
    document.getElementById('ddFrame').textContent = FMT.number(snap.frame_id);
    document.getElementById('ddTime').textContent = FMT.time(t);
    document.getElementById('ddVehicles').textContent = FMT.number(snap.cumulative_vehicles);
    document.getElementById('ddEvents').textContent = FMT.number(snap.cumulative_events);
    document.getElementById('ddSpeed').textContent = FMT.speed(snap.avg_speed_kmh) + ' km/h';
    renderClassBars(snap.class_counts_active || {});
  }

  function renderClassBars(counts) {
    const container = document.getElementById('ddClassBars');
    if (!container) return;
    const max = Math.max(1, ...Object.values(counts));
    container.innerHTML = Object.entries(counts).map(([name, v]) => {
      const pct = Math.round((v / max) * 100);
      const color = CMap.forClass(name);
      return `<div class="dd-class-bar">
        <span class="dd-class-name">${name}</span>
        <span class="dd-class-track"><span class="dd-class-fill" style="width:${pct}%;background:${color};"></span></span>
        <span class="dd-class-val">${v}</span>
      </div>`;
    }).join('');
  }

  function destroy() {
    if (_video) {
      _video.removeEventListener('timeupdate', onTimeUpdate);
      _video.pause(); _video = null;
    }
    _lastSnapIdx = -1;
  }

  return { render, destroy };
})();

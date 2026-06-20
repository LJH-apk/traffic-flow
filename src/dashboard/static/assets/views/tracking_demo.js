/* 10.2 多目标跟踪演示：视频（北进口）+ 跟踪时长面板 */
window.TrackingDemoView = (() => {

  let _video = null, _chartInst = null, _lastSnapIdx = -1;
  let _trackIndex = new Map();
  let _focusTracks = [];
  let _overview = null;
  let _trackStats = null;
  const _scope = 'north';

  function render() {
    const container = document.getElementById('viewContainer');
    if (!container) return;
    container.innerHTML = `
      <div class="td-stage">
        <div class="td-video-wrap">
          <video class="td-video" id="tdVideo"
            src="${CONST.VIDEO_URLS.north}" preload="auto"
            autoplay playsinline muted="muted" loop></video>
          <div class="td-hud td-hud-top">
            <div class="td-hud-pill">
              <span class="td-hud-label">当前时间</span>
              <span class="td-hud-value" id="tdCurrentTime">00:00</span>
            </div>
            <div class="td-hud-pill">
              <span class="td-hud-label">当前帧</span>
              <span class="td-hud-value" id="tdCurrentFrame">0</span>
            </div>
            <div class="td-hud-pill">
              <span class="td-hud-label">活跃 ID</span>
              <span class="td-hud-value" id="tdActiveTracks">0</span>
            </div>
          </div>
          <div class="td-hud td-hud-bottom">
            <div class="td-hud-card">
              <div class="td-hud-title">轨迹连续性观察要点</div>
              <div class="td-hud-text">同一车辆跨帧保持同一 ID，转弯和交汇场景下轨迹尽量不断裂。</div>
            </div>
            <div class="td-hud-card td-hud-card-accent">
              <div class="td-hud-title">当前平均速度</div>
              <div class="td-hud-metric"><span id="tdLiveSpeed">0.0</span><small>km/h</small></div>
            </div>
          </div>
          <div class="td-empty" id="tdEmpty" onclick="var v=document.getElementById('tdVideo');if(v)v.play();this.style.display='none';" style="cursor:pointer;">
            ⬡ 加载跟踪数据中…<br><small>如视频未自动播放，请点击此处</small></div>
        </div>
        <div class="td-sidebar">
          <div class="td-highlight-box" id="tdHighlight">
            <h4>多目标跟踪摘要</h4>
            <div class="td-highlight-grid">
              <div class="td-highlight-metric">
                <span class="td-highlight-label">最长连续跟踪</span>
                <span class="td-highlight-val" id="tdBestId">--</span>
              </div>
              <div class="td-highlight-metric">
                <span class="td-highlight-label">总轨迹数</span>
                <span class="td-highlight-num" id="tdTrackTotal">--</span>
              </div>
              <div class="td-highlight-metric">
                <span class="td-highlight-label">累计唯一 ID</span>
                <span class="td-highlight-num" id="tdUniqueTracks">--</span>
              </div>
              <div class="td-highlight-metric">
                <span class="td-highlight-label">超过 60s 轨迹</span>
                <span class="td-highlight-num" id="tdLongTracks">--</span>
              </div>
            </div>
            <div class="td-highlight-sub" id="tdBestInfo">加载中…</div>
          </div>
          <div class="td-panel">
            <h4>典型跟踪目标</h4>
            <div class="td-focus-list" id="tdFocusList"></div>
          </div>
          <div class="td-panel">
            <h4>ID 连续性说明</h4>
            <div class="td-note-list">
              <div class="td-note-item">车辆在连续帧中应尽量保持同一编号，避免频繁切换。</div>
              <div class="td-note-item">车辆遮挡、并行跟驰和转弯交汇时，重点观察轨迹是否平滑延续。</div>
              <div class="td-note-item">长时轨迹越多，说明系统对稳定通行目标的身份维护越可靠。</div>
            </div>
          </div>
          <div class="td-panel">
            <h4>跟踪时长排名</h4>
            <div id="tdTrackList" style="max-height:280px;overflow-y:auto;"></div>
          </div>
          <div class="td-panel">
            <h4>跟踪时长分布</h4>
            <div class="td-chart-wrap" id="tdDurChart"></div>
          </div>
          <div class="td-panel">
            <h4>代表性轨迹示意</h4>
            <div class="td-mini-map-wrap">
              <canvas id="tdMiniMap" class="td-mini-map"></canvas>
            </div>
            <div class="td-mini-caption">选取持续时间最长的若干轨迹进行路径示意，便于报告截图展示。</div>
          </div>
        </div>
      </div>`;

    _video = document.getElementById('tdVideo');
    _video.addEventListener('loadedmetadata', () => {
      document.getElementById('tdEmpty').style.display = 'none';
      syncSnapshot();
    });
    _video.addEventListener('timeupdate', onTimeUpdate);
    _video.addEventListener('playing', () => {
      document.getElementById('tdEmpty').style.display = 'none';
    });
    loadData();
  }

  async function loadData() {
    try {
      const [meta, trackStats, overview, timeline, trajectories] = await Promise.all([
        API.loadMeta(_scope), API.loadTrackStats(_scope), API.loadOverview(_scope),
        API.loadTimeline(_scope), API.loadTrajectories(_scope)
      ]);
      CMap.init(meta);
      Store.set('meta', meta);
      Store.set('timeline', timeline);
      _overview = overview;
      _trackStats = trackStats;
      buildTrackIndex(trajectories);

      const best = trackStats.tracks?.[0];
      if (best) {
        document.getElementById('tdBestId').textContent = '#' + best.track_id;
        document.getElementById('tdBestInfo').textContent =
          best.class_name + ' · 持续 ' + best.duration_s.toFixed(1) + 's · 均速 ' + best.avg_speed_kmh.toFixed(1) + ' km/h';
      }
      document.getElementById('tdTrackTotal').textContent = FMT.number(trackStats.total_tracks || 0);
      document.getElementById('tdUniqueTracks').textContent = FMT.number(overview.unique_tracks || 0);
      document.getElementById('tdLongTracks').textContent = FMT.number(trackStats.stats?.tracks_over_60s || 0);

      renderTrackList(trackStats.tracks || []);
      renderFocusList(trackStats.tracks || []);
      renderDurationChart(trackStats);
      renderMiniMap();
      document.getElementById('tdEmpty').style.display = 'none';
      if (_video && _video.readyState >= 2) syncSnapshot();

    } catch (e) {
      console.error(e);
      document.getElementById('tdEmpty').innerHTML = '<span>⚠ 数据加载失败</span>';
    }
  }

  function onTimeUpdate() {
    if (Store.get('timeline')) syncSnapshot();
  }

  function syncSnapshot() {
    if (!_video) return;
    const timeline = Store.get('timeline');
    if (!timeline || !timeline.snapshots) return;
    const tRaw = _video.currentTime || 0;
    const t = Math.floor(tRaw);
    const snaps = timeline.snapshots;
    const idx = Math.min(t, snaps.length - 1);
    if (idx < 0 || idx === _lastSnapIdx) return;
    _lastSnapIdx = idx;
    const snap = snaps[idx];
    const setText = (id, value) => {
      const el = document.getElementById(id);
      if (el) el.textContent = value;
    };
    setText('tdCurrentTime', FMT.time(t));
    setText('tdCurrentFrame', FMT.number(snap.frame_id));
    setText('tdActiveTracks', FMT.number(snap.active_tracks));
    setText('tdLiveSpeed', FMT.speed(snap.avg_speed_kmh));
    updateDynamicPanels(tRaw);
  }

  function renderTrackList(tracks) {
    const container = document.getElementById('tdTrackList');
    if (!container) return;
    container.innerHTML = (tracks || []).slice(0, 15).map(t => {
      const color = CMap.forClass(t.class_name);
      const activeLabel = isTrackActiveAtTime(t, _video?.currentTime || 0) ? '活跃' : '历史';
      return `<div class="td-track-item">
        <span class="td-track-id" style="color:${color}">#${t.track_id}</span>
        <span class="td-track-class">${t.class_name} · ${t.point_count}点 · ${t.avg_speed_kmh.toFixed(1)}km/h · ${activeLabel}</span>
        <span class="td-track-dur">${t.duration_s.toFixed(1)}s</span>
      </div>`;
    }).join('') || '<div class="td-track-empty">当前时刻没有持续轨迹</div>';
  }

  function renderFocusList(tracks) {
    const container = document.getElementById('tdFocusList');
    if (!container) return;
    const focusTracks = pickFocusTracks(tracks);
    _focusTracks = focusTracks;
    container.innerHTML = focusTracks.map((t, idx) => {
      const color = CMap.forClass(t.class_name);
      const path = _trackIndex.get(t.track_id);
      const pointCount = path?.point_count || t.point_count || 0;
      const continuity = t.duration_s >= 120 ? '高' : (t.duration_s >= 45 ? '中' : '基础');
      const activeState = isTrackActiveAtTime(t, _video?.currentTime || 0) ? '当前在场' : '历史代表';
      return `<div class="td-focus-item">
        <div class="td-focus-top">
          <span class="td-focus-badge" style="color:${color};border-color:${color}33;background:${color}14;">目标 ${idx + 1}</span>
          <span class="td-focus-id" style="color:${color}">#${t.track_id}</span>
        </div>
        <div class="td-focus-main">${t.class_name} · 持续 ${t.duration_s.toFixed(1)}s · 均速 ${t.avg_speed_kmh.toFixed(1)}km/h</div>
        <div class="td-focus-sub">采样点 ${pointCount} 个 · 连续性观察等级 ${continuity} · ${activeState}</div>
      </div>`;
    }).join('');
  }

  function pickFocusTracks(tracks) {
    const byClass = new Set();
    const focus = [];
    (tracks || []).forEach(track => {
      if (!track || !track.track_id) return;
      if (focus.length >= 4) return;
      if (track.duration_s < 20) return;
      if (byClass.has(track.class_name) && focus.length < 2) return;
      focus.push(track);
      byClass.add(track.class_name);
    });
    if (focus.length < 4) {
      (tracks || []).forEach(track => {
        if (focus.length >= 4) return;
        if (focus.some(item => item.track_id === track.track_id)) return;
        focus.push(track);
      });
    }
    return focus;
  }

  function isTrackActiveAtTime(track, timeS) {
    if (!track) return false;
    if (typeof track.start_s !== 'number' || typeof track.end_s !== 'number') return false;
    return track.start_s <= timeS && timeS <= track.end_s;
  }

  function activeTracksAtTime(timeS) {
    const tracks = _trackStats?.tracks || [];
    const active = tracks.filter(track => isTrackActiveAtTime(track, timeS));
    if (active.length) {
      return active.sort((a, b) => b.duration_s - a.duration_s);
    }
    return tracks.slice(0, 8);
  }

  function updateDynamicPanels(timeS) {
    if (!_trackStats) return;
    const activeTracks = activeTracksAtTime(timeS);
    const best = activeTracks[0];
    const bestId = document.getElementById('tdBestId');
    const bestInfo = document.getElementById('tdBestInfo');
    if (best && bestId && bestInfo) {
      bestId.textContent = '#' + best.track_id;
      const state = isTrackActiveAtTime(best, timeS) ? '当前活跃' : '全局最长';
      bestInfo.textContent =
        state + ' · ' + best.class_name + ' · 持续 ' + best.duration_s.toFixed(1) +
        's · 时间窗 ' + FMT.time(best.start_s) + '-' + FMT.time(best.end_s);
    }

    renderTrackList(activeTracks);
    renderFocusList(activeTracks.slice(0, 4));
    renderMiniMap();
  }

  function buildTrackIndex(trajectories) {
    _trackIndex = new Map();
    const tracks = trajectories?.tracks || [];
    tracks.forEach(track => {
      _trackIndex.set(track.track_id, track);
    });
  }

  function renderMiniMap() {
    const canvas = document.getElementById('tdMiniMap');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const wrap = canvas.parentElement.getBoundingClientRect();
    const width = Math.max(280, Math.floor(wrap.width));
    const height = Math.max(150, Math.floor(wrap.height));
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    canvas.style.width = width + 'px';
    canvas.style.height = height + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, width, height);
    ctx.strokeStyle = 'rgba(148,163,184,.12)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
      const y = (height / 4) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    const srcW = 3840;
    const srcH = 2160;
    const scale = Math.min(width / srcW, height / srcH);
    const offsetX = (width - srcW * scale) / 2;
    const offsetY = (height - srcH * scale) / 2;

    _focusTracks.forEach(track => {
      const path = _trackIndex.get(track.track_id);
      const pts = path?.points || [];
      if (pts.length < 2) return;
      const color = CMap.forClass(track.class_name);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.6;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.moveTo(offsetX + pts[0][0] * scale, offsetY + pts[0][1] * scale);
      for (let i = 1; i < pts.length; i++) {
        ctx.lineTo(offsetX + pts[i][0] * scale, offsetY + pts[i][1] * scale);
      }
      ctx.stroke();

      const end = pts[pts.length - 1];
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(offsetX + end[0] * scale, offsetY + end[1] * scale, 2.8, 0, Math.PI * 2);
      ctx.fill();
    });

    ctx.globalAlpha = 1;
  }

  function renderDurationChart(data) {
    if (_chartInst) _chartInst.dispose();
    const dom = document.getElementById('tdDurChart');
    if (!dom) return;
    _chartInst = echarts.init(dom);
    const bins = data.duration_histogram?.bins || [];
    const counts = data.duration_histogram?.counts || [];
    _chartInst.setOption({
      backgroundColor: 'transparent',
      textStyle: { color: '#8aa4ba', fontSize: 10 },
      tooltip: { trigger: 'axis' },
      grid: { left: 36, right: 8, top: 4, bottom: 18 },
      xAxis: { type: 'category', data: bins, axisLabel: { color: '#8aa4ba', fontSize: 9, rotate: 20 }, axisLine: { lineStyle: { color: '#334155' } } },
      yAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(148,163,184,.08)' } }, axisLabel: { color: '#8aa4ba', fontSize: 9 } },
      series: [{
        type: 'bar', data: counts,
        itemStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: '#22d3ee' }, { offset: 1, color: 'rgba(34,211,238,.2)' }] }, borderRadius: [2, 2, 0, 0] },
        barWidth: '60%',
      }],
    });
  }

  function destroy() {
    if (_video) {
      _video.removeEventListener('timeupdate', onTimeUpdate);
      _video.pause();
      _video = null;
    }
    if (_chartInst) { _chartInst.dispose(); _chartInst = null; }
    _lastSnapIdx = -1;
    _trackIndex = new Map();
    _focusTracks = [];
    _overview = null;
    _trackStats = null;
  }

  return { render, destroy };
})();

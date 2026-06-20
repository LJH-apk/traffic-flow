/* 10.4 可视化界面展示：全屏视频 + 四角浮层 + 拥堵告警 */
window.VisualizationView = (() => {

  let _video = null, _lastSnapIdx = -1, _miniChart = null;
  let _events = [];
  let _scope = 'north';   // 当前选中进口: north / east / south / merged

  function resolveScopeFromVideo() { return _scope; }

  function render() {
    const container = document.getElementById('viewContainer');
    if (!container) return;
    container.innerHTML = `
      <div class="vi-stage">
        <video class="vi-video" id="viVideo"
          src="${CONST.VIDEO_URLS[_scope]}" preload="auto"
          autoplay playsinline muted="muted" loop></video>
        <div class="vi-empty" id="viEmpty" onclick="var v=document.getElementById('viVideo');if(v)v.play();this.style.display='none';" style="cursor:pointer;">
          ⬡ 加载可视化数据中…<br><small>如视频未自动播放，请点击此处</small></div>

        <div class="vi-corner-tl">
          <div class="vi-scope-bar">
            <button class="vi-scope-btn active" data-scope="north">北进口</button>
            <button class="vi-scope-btn" data-scope="east">东进口</button>
            <button class="vi-scope-btn" data-scope="south">南进口</button>
            <button class="vi-scope-btn" data-scope="merged">全景</button>
          </div>
          <div class="vi-panel">
            <span class="vi-fps">FPS: <b id="viFps">25.0</b></span>
            <span class="vi-det">DET: <b id="viDet">0</b></span>
            <span class="vi-det">帧: <b id="viFrame">0</b></span>
            <span class="vi-det">时间: <b id="viTime">00:00 / 00:00</b></span>
          </div>
        </div>

        <div class="vi-corner-tr">
          <div class="vi-panel" style="border-color:rgba(245,158,11,.35);text-align:right;">
            <div class="vi-label">累计车辆</div>
            <div class="vi-val" style="color:var(--amber);font-size:32px;" id="viVehicles">--</div>
            <div class="vi-label" style="margin-top:4px;">过车事件</div>
            <div class="vi-val" style="color:var(--blue);font-size:32px;" id="viEvents">--</div>
            <div class="vi-label" style="margin-top:4px;">均速</div>
            <div class="vi-val" style="color:var(--emerald);font-size:24px;" id="viSpeed">--</div>
          </div>
        </div>

        <div class="vi-corner-bl">
          <div class="vi-panel" style="min-width:280px;">
            <div class="vi-label" style="margin-bottom:4px;">🚦 拥堵告警 (速度 &lt; 10 km/h)</div>
            <div id="viAlerts"></div>
          </div>
        </div>

        <div class="vi-corner-br">
          <div class="vi-panel">
            <div class="vi-label">断面流量分布</div>
            <div class="vi-mini-chart" id="viMiniChart"></div>
          </div>
        </div>

        <!-- Corner accents -->
        <div class="corner corner-tl"></div>
        <div class="corner corner-tr"></div>
        <div class="corner corner-bl"></div>
        <div class="corner corner-br"></div>
      </div>`;

    _video = document.getElementById('viVideo');
    _video.addEventListener('loadedmetadata', () => document.getElementById('viEmpty').style.display = 'none');
    _video.addEventListener('playing', () => document.getElementById('viEmpty').style.display = 'none');
    _video.addEventListener('timeupdate', onTimeUpdate);

    // scope 切换事件委托
    document.querySelector('.vi-scope-bar')?.addEventListener('click', function (e) {
      const btn = e.target.closest('.vi-scope-btn');
      if (!btn) return;
      const newScope = btn.dataset.scope;
      if (newScope && newScope !== _scope) {
        setScope(newScope);
      }
    });

    loadData();
  }

  async function loadData() {
    try {
      const [meta, timeline, overview, events] = await Promise.all([
        API.loadMeta(resolveScopeFromVideo()),
        API.loadTimeline(resolveScopeFromVideo()),
        API.loadOverview(resolveScopeFromVideo()),
        API.loadEvents(resolveScopeFromVideo())
      ]);
      Store.set('meta', meta); Store.set('timeline', timeline);
      Store.set('overview', overview); Store.set('events', events);
      CMap.init(meta);
      _events = events.items || [];

      // 初始拥堵告警
      renderCongestionAlertsForTime(0);

      // 断面流量迷你图
      setTimeout(() => renderMiniChart(timeline), 200);

      // 顶栏时间
      const dur = meta.video.duration_s;
      document.getElementById('viTime').textContent = '00:00 / ' + FMT.time(dur);
      document.getElementById('viEmpty').style.display = 'none';

      if (_video && _video.readyState >= 2) syncOverlay();

    } catch (e) {
      console.error(e);
      document.getElementById('viEmpty').innerHTML = '<span>⚠ 数据加载失败</span>';
    }
  }

  function setScope(newScope) {
    _scope = newScope;
    _lastSnapIdx = -1;
    _events = [];

    // 更新按钮 active 状态
    document.querySelectorAll('.vi-scope-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.scope === _scope);
    });

    // 切换视频源
    if (_video) {
      _video.src = CONST.VIDEO_URLS[_scope];
      _video.load();
      var empty = document.getElementById('viEmpty');
      if (empty) empty.style.display = 'flex';
    }

    // 清除 API 缓存，重新加载数据
    API.clearCache();
    loadData();
  }

  function filterEventsByScope(events, scope) {
    if (scope === 'merged') return events;
    var meta = Store.get('meta');
    var sections = meta?.sections || [];
    var scopeSections = sections.filter(function(s) { return s.entrance === scope; }).map(function(s) { return s.name; });
    if (scopeSections.length === 0) return events;
    return events.filter(function(e) {
      var sec = e.section || '';
      return scopeSections.some(function(s) { return sec.includes(s) || s.includes(sec); });
    });
  }

  function onTimeUpdate() { if (Store.get('timeline')) syncOverlay(); }

  function syncOverlay() {
    if (!_video) return;
    const timeline = Store.get('timeline');
    if (!timeline || !timeline.snapshots) return;
    const t = Math.floor(_video.currentTime);
    const snaps = timeline.snapshots;
    const idx = Math.min(t, snaps.length - 1);
    if (idx < 0 || idx === _lastSnapIdx) return;
    _lastSnapIdx = idx;
    const snap = snaps[idx];

    document.getElementById('viDet').textContent = FMT.number(snap.active_tracks);
    document.getElementById('viFrame').textContent = FMT.number(snap.frame_id);
    document.getElementById('viVehicles').textContent = FMT.number(snap.cumulative_vehicles);
    document.getElementById('viEvents').textContent = FMT.number(snap.cumulative_events);
    document.getElementById('viSpeed').textContent = FMT.speed(snap.avg_speed_kmh) + ' km/h';

    const meta = Store.get('meta');
    const dur = meta?.video?.duration_s || 0;
    document.getElementById('viTime').textContent = FMT.time(t) + ' / ' + FMT.time(dur);
    renderCongestionAlertsForTime(t);
  }

  function renderCongestionAlertsForTime(timeS) {
    var windowStart = Math.max(0, timeS - 6);
    var currentEvents = _events.filter(function(e) {
      return e.timestamp_s >= windowStart &&
        e.timestamp_s <= timeS &&
        e.speed_kmh > 0 &&
        e.speed_kmh < 10;
    });
    // 防御性：按当前进口过滤断面，防止 scope 数据出错时混入其他进口数据
    currentEvents = filterEventsByScope(currentEvents, _scope);
    renderCongestionAlerts(currentEvents, timeS);
  }

  function renderCongestionAlerts(events, timeS) {
    const container = document.getElementById('viAlerts');
    if (!container) return;
    if (!events.length) {
      container.innerHTML = '<div class="vi-no-alerts">✓ 当前时间窗内无拥堵告警</div>';
      return;
    }

    // 按断面分组，每个断面取最近一条，体现“当前时刻附近”的动态告警
    const bySection = {};
    for (const e of events) {
      const sec = e.section || '未知';
      if (!bySection[sec] || e.timestamp_s > bySection[sec].timestamp_s) {
        bySection[sec] = e;
      }
    }
    const entries = Object.entries(bySection)
      .sort((a, b) => b[1].timestamp_s - a[1].timestamp_s)
      .slice(0, 4);
    container.innerHTML = entries.map(([sec, e]) => {
      const cls = e.speed_kmh < 5 ? 'vi-alert-danger' : 'vi-alert-warn';
      const icon = e.speed_kmh < 5 ? '🔴' : '🟡';
      const ago = Math.max(0, Math.round(timeS - e.timestamp_s));
      return `<div class="vi-alert ${cls}">
        <span class="vi-alert-icon">${icon}</span>
        <span>${sec} · ${e.class_name} #${e.track_id} · ${e.speed_kmh.toFixed(1)} km/h · ${FMT.time(e.timestamp_s)} · ${ago}s前</span>
      </div>`;
    }).join('');
  }

  function renderMiniChart(timeline) {
    const dom = document.getElementById('viMiniChart');
    if (!dom) return;
    if (_miniChart) _miniChart.dispose();
    _miniChart = echarts.init(dom);

    // 从 timeline 提取每分钟断面流量趋势
    const snapshots = timeline.snapshots || [];
    const step = 60; // 每分钟采样
    const times = [];
    const values = [];
    for (let i = 0; i < snapshots.length; i += step) {
      const s = snapshots[i];
      times.push(FMT.time(s.t));
      let total = 0;
      const sc = s.section_counts_cumulative || {};
      for (const v of Object.values(sc)) total += v;
      values.push(total);
    }

    _miniChart.setOption({
      backgroundColor: 'transparent',
      textStyle: { color: '#8aa4ba', fontSize: 9 },
      grid: { left: 28, right: 4, top: 4, bottom: 14 },
      xAxis: { type: 'category', data: times, axisLabel: { color: '#8aa4ba', fontSize: 8, interval: Math.floor(times.length / 4) }, axisLine: { show: false }, axisTick: { show: false } },
      yAxis: { type: 'value', splitLine: { lineStyle: { color: 'rgba(148,163,184,.06)' } }, axisLabel: { color: '#8aa4ba', fontSize: 8 } },
      series: [{
        type: 'line', data: values,
        smooth: true, symbol: 'none', lineStyle: { color: '#22d3ee', width: 1.5 },
        areaStyle: { color: { type: 'linear', x: 0, y: 0, x2: 0, y2: 1, colorStops: [{ offset: 0, color: 'rgba(34,211,238,.15)' }, { offset: 1, color: 'rgba(34,211,238,0)' }] } },
      }],
    });
  }

  function destroy() {
    if (_video) { _video.removeEventListener('timeupdate', onTimeUpdate); _video.pause(); _video = null; }
    if (_miniChart) { _miniChart.dispose(); _miniChart = null; }
    _lastSnapIdx = -1;
    _events = [];
    _scope = 'north';
  }

  return { render, destroy };
})();

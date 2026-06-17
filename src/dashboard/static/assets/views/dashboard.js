/* ═══════════════════════════════════════════════════════════════════
   views/dashboard.js — 数据看板：KPI 实时轮询 + 静态图表 + 一键刷新
   ═══════════════════════════════════════════════════════════════════ */
window.DashboardView = (() => {

  let _pollTimer = null;
  let _prevKPI = {};
  let _overviewStatic = null; // 静态数据的初始快照

  function render() {
    const container = document.getElementById('viewContainer');
    if (!container) return;

    container.innerHTML = `
      <div class="dashboard-stage" id="dashStage">
        <!-- KPI Row -->
        <div class="kpi-row" id="kpiRow">
          <div class="kpi-card kpi-accent-cyan">
            <div class="kpi-label">累计车辆</div>
            <div class="kpi-val" id="kpiVehicles">--</div>
            <div class="kpi-sub">辆 · <span id="kpiDataAge">静态快照</span></div>
          </div>
          <div class="kpi-card kpi-accent-amber">
            <div class="kpi-label">过车事件</div>
            <div class="kpi-val" id="kpiEvents">--</div>
            <div class="kpi-sub">次</div>
          </div>
          <div class="kpi-card kpi-accent-emerald">
            <div class="kpi-label">实时均速</div>
            <div class="kpi-val" id="kpiSpeed">--</div>
            <div class="kpi-sub">km/h</div>
          </div>
          <div class="kpi-card kpi-accent-blue">
            <div class="kpi-label">活跃目标</div>
            <div class="kpi-val" id="kpiActive">--</div>
            <div class="kpi-sub">个</div>
          </div>
          <div class="kpi-card kpi-accent-rose">
            <div class="kpi-label">检测时长</div>
            <div class="kpi-val" id="kpiDuration">--</div>
            <div class="kpi-sub">秒</div>
          </div>
        </div>

        <!-- Chart Grid -->
        <div class="chart-grid">
          <div class="chart-panel">
            <h3>车型分布 <button class="db-refresh-btn" id="dbRefreshBtn" title="重建仪表盘数据">↻ 刷新数据</button></h3>
            <div class="chart-body" id="chartPie"></div>
          </div>
          <div class="chart-panel">
            <h3>速度分布</h3>
            <div class="chart-body" id="chartHist"></div>
          </div>
          <div class="chart-panel">
            <h3>断面流量对比</h3>
            <div class="chart-body" id="chartSection"></div>
          </div>
          <div class="chart-panel">
            <h3>精度评估</h3>
            <div class="chart-body" id="chartValidation"></div>
          </div>
        </div>

        <!-- Bottom Row -->
        <div class="db-bottom">
          <div class="chart-panel">
            <h3>轨迹路径图</h3>
            <div class="chart-body" style="position:relative;">
              <canvas id="canvasTraj" style="width:100%;height:100%;"></canvas>
            </div>
          </div>
          <div class="events-panel">
            <h3>最近过车事件</h3>
            <div class="ev-list" id="evList"></div>
          </div>
        </div>
      </div>
    `;

    // 刷新按钮
    document.getElementById('dbRefreshBtn').addEventListener('click', async function () {
      this.textContent = '⏳ 重建中…';
      this.disabled = true;
      try {
        await fetch('/api/detect/rebuild', { method: 'POST' }).catch(() => {});
      } catch(e) {}
      // 清缓存重新加载
      API.clearCache();
      await loadStaticData();
      this.textContent = '↻ 刷新数据';
      this.disabled = false;
    });

    loadStaticData();
    startPolling();
  }

  async function loadStaticData() {
    try {
      const { overview, charts, events, trajectories, validation } = await API.loadDashboardData();
      const meta = Store.get('meta') || await API.loadMeta();
      if (!Store.get('meta')) Store.set('meta', meta);

      Store.set('overview', overview);
      Store.set('charts', charts);
      Store.set('events', events);
      Store.set('trajectories', trajectories);
      Store.set('validation', validation);
      Store.set('dashLoaded', true);

      _overviewStatic = overview;

      if (meta) CMap.init(meta);

      // 初始 KPI 从静态数据填充
      fillKPI(overview);

      // 图表
      ChartFactory.classPie('chartPie', charts.class_pie, meta);
      ChartFactory.speedHistogram('chartHist', charts.speed_histogram);
      ChartFactory.sectionBar('chartSection', charts.section_bar);

      if (validation && validation.summary) {
        ChartFactory.validationBar('chartValidation', validation.summary);
      }

      // 轨迹图
      setTimeout(() => {
        TrajectoryCanvas.init('canvasTraj');
        TrajectoryCanvas.render(trajectories, meta);
      }, 100);

      // 事件列表
      EventList.render('evList', events ? events.items : [], { limit: 10 });

      window.addEventListener('resize', onResize);
      updateTopBar(overview);

    } catch (err) {
      console.error('Dashboard data load failed:', err);
      document.getElementById('viewContainer').innerHTML =
        '<div class="loading-placeholder" style="height:100vh;">⚠ 数据加载失败：' + err.message + '</div>';
    }
  }

  function startPolling() {
    _pollTimer = setInterval(pollLiveKPI, 1500);
    pollLiveKPI();
  }

  async function pollLiveKPI() {
    try {
      const [snapRes, statsRes] = await Promise.all([
        fetch('/api/live/status').then(r => r.json()).catch(() => null),
        fetch('/api/live/stats').then(r => r.json()).catch(() => null),
      ]);

      // 如果在检测中，用实时数据覆盖 KPI
      if (snapRes && snapRes.running) {
        const ageEl = document.getElementById('kpiDataAge');
        if (ageEl) ageEl.textContent = '实时 · ' + FMT.time(snapRes.progress?.timestamp_s || 0);

        if (statsRes) {
          animateKPI('kpiVehicles', statsRes.vehicles || 0, 0);
          animateKPI('kpiEvents', statsRes.events || 0, 0);
          animateKPI('kpiSpeed', statsRes.avg_speed || 0, 1);
          animateKPI('kpiActive', statsRes.active_tracks || 0, 0);

          // 时长从进度取
          const durEl = document.getElementById('kpiDuration');
          if (durEl) durEl.textContent = FMT.time(snapRes.progress?.timestamp_s || 0);
        }
      } else {
        // 没在运行，显示静态数据时间
        const ageEl = document.getElementById('kpiDataAge');
        if (ageEl && _overviewStatic) {
          ageEl.textContent = '快照 · ' + FMT.time(_overviewStatic.time_range?.end_s || 0);
        }
      }
    } catch (e) {
      // 静默
    }
  }

  function animateKPI(id, target, digits) {
    const el = document.getElementById(id);
    if (!el) return;
    const curText = el.textContent.replace(/,/g, '').replace(' km/h','');
    const from = parseFloat(curText) || 0;
    const to = Number(target) || 0;
    if (Math.abs(from - to) < 0.05) return;

    const prev = _prevKPI[id];
    if (prev === to) return;
    _prevKPI[id] = to;

    const start = performance.now();
    const dur = 350;
    function step(now) {
      const p = Math.min(1, (now - start) / dur);
      const e = 1 - Math.pow(1 - p, 3);
      const val = from + (to - from) * e;
      el.textContent = digits ? val.toFixed(digits) : Math.round(val).toLocaleString('zh-CN');
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  function fillKPI(overview) {
    if (!overview) return;
    const setter = (id, val, digits) => {
      const el = document.getElementById(id);
      if (!el) return;
      if (typeof val === 'number') {
        el.textContent = digits ? val.toFixed(digits) : FMT.number(val);
      } else {
        el.textContent = val || '--';
      }
    };
    setter('kpiVehicles', overview.total_vehicles, 0);
    setter('kpiEvents', overview.total_events, 0);
    setter('kpiSpeed', overview.avg_speed_kmh, 1);
    setter('kpiActive', overview.unique_tracks, 0);
    setter('kpiDuration', FMT.time(overview.active_duration_s));
  }

  function updateTopBar(overview) {
    const infoEl = document.getElementById('topbarInfo');
    if (!infoEl) return;
    if (overview && overview.time_range) {
      infoEl.innerHTML = '<span class="topbar-time">' + FMT.time(overview.time_range.start_s) + ' ~ ' + FMT.time(overview.time_range.end_s) + '</span>';
    }
  }

  function onResize() {
    ChartFactory.resizeAll();
    const trajectories = Store.get('trajectories');
    const meta = Store.get('meta');
    if (trajectories) TrajectoryCanvas.render(trajectories, meta);
  }

  function destroy() {
    if (_pollTimer) clearInterval(_pollTimer);
    _pollTimer = null;
    _prevKPI = {};
    _overviewStatic = null;
    window.removeEventListener('resize', onResize);
    ChartFactory.disposeAll();
    TrajectoryCanvas.destroy();
  }

  return { render, destroy };

})();

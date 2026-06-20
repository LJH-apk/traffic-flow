/* 10.3 交通流统计演示：KPI 卡片 + 2×2 ECharts 图表 */
window.TrafficStatsView = (() => {

  function render() {
    const container = document.getElementById('viewContainer');
    if (!container) return;
    container.innerHTML = `
      <div class="ts-stage">
        <div class="ts-kpi-row">
          <div class="ts-kpi-card ts-kpi-accent-cyan"><div class="ts-kpi-label">累计车辆</div><div class="ts-kpi-val" id="tsVehicles">--</div><div class="ts-kpi-sub">辆</div></div>
          <div class="ts-kpi-card ts-kpi-accent-amber"><div class="ts-kpi-label">过车事件</div><div class="ts-kpi-val" id="tsEvents">--</div><div class="ts-kpi-sub">次</div></div>
          <div class="ts-kpi-card ts-kpi-accent-emerald"><div class="ts-kpi-label">平均速度</div><div class="ts-kpi-val" id="tsSpeed">--</div><div class="ts-kpi-sub">km/h</div></div>
          <div class="ts-kpi-card ts-kpi-accent-blue"><div class="ts-kpi-label">最高速度</div><div class="ts-kpi-val" id="tsMaxSpeed">--</div><div class="ts-kpi-sub">km/h</div></div>
        </div>
        <div class="ts-chart-grid">
          <div class="ts-chart-panel"><h3>断面流量对比（分方向）</h3><div class="ts-chart-body" id="tsChartSection"></div></div>
          <div class="ts-chart-panel"><h3>车道流量</h3><div class="ts-chart-body" id="tsChartLane"></div></div>
          <div class="ts-chart-panel"><h3>速度分布</h3><div class="ts-chart-body" id="tsChartSpeed"></div></div>
          <div class="ts-chart-panel"><h3>车型分布</h3><div class="ts-chart-body" id="tsChartClass"></div></div>
        </div>
        <div class="ts-stats-grid">
          <div class="ts-stats-wrap"><h3>分方向统计</h3><div id="tsDirStats"></div></div>
          <div class="ts-stats-wrap"><h3>分进口统计</h3><div id="tsEntranceStats"></div></div>
        </div>
      </div>`;

    loadAndRender();
  }

  async function loadAndRender() {
    try {
      const [overview, charts, meta] = await Promise.all([
        API.loadOverview(), API.loadCharts(), API.loadMeta()
      ]);
      CMap.init(meta);

      document.getElementById('tsVehicles').textContent = FMT.number(overview.total_vehicles);
      document.getElementById('tsEvents').textContent = FMT.number(overview.total_events);
      document.getElementById('tsSpeed').textContent = FMT.speed(overview.avg_speed_kmh);
      document.getElementById('tsMaxSpeed').textContent = FMT.speed(overview.max_speed_kmh);

      ChartFactory.sectionBar('tsChartSection', charts.section_bar);
      ChartFactory.speedHistogram('tsChartSpeed', charts.speed_histogram);
      ChartFactory.classPie('tsChartClass', charts.class_pie, meta);

      // 车道流量柱状图
      renderLaneChart('tsChartLane', charts.lane_heatmap);

      // 方向统计
      const dirContainer = document.getElementById('tsDirStats');
      if (dirContainer) {
        const dirs = overview.direction_counts || {};
        dirContainer.innerHTML = Object.entries(dirs).map(([k, v]) =>
          `<div class="ts-stat-item"><span class="ts-stat-label">${k}</span><span class="ts-stat-val">${FMT.number(v)} 次</span></div>`
        ).join('');
      }

      // 进口统计（从断面统计聚合）
      const entranceContainer = document.getElementById('tsEntranceStats');
      if (entranceContainer) {
        const secs = overview.section_counts || {};
        const entranceMap = {};
        for (const [sec, cnt] of Object.entries(secs)) {
          const ent = sec.includes('北') ? '北进口' : sec.includes('南') ? '南进口' : sec.includes('东') ? '东进口' : '其他';
          entranceMap[ent] = (entranceMap[ent] || 0) + cnt;
        }
        entranceContainer.innerHTML = Object.entries(entranceMap).map(([k, v]) =>
          `<div class="ts-stat-item"><span class="ts-stat-label">${k}</span><span class="ts-stat-val">${FMT.number(v)} 次</span></div>`
        ).join('');
      }

      window.addEventListener('resize', ChartFactory.resizeAll);

    } catch (e) {
      console.error(e);
      document.getElementById('viewContainer').innerHTML = '<div class="ts-empty">⚠ 数据加载失败：' + e.message + '</div>';
    }
  }

  function renderLaneChart(domId, data) {
    const dom = document.getElementById(domId);
    if (!dom) return;
    const inst = echarts.init(dom);
    const lanes = data?.lanes || [];
    const counts = data?.counts || [];
    const colors = ['#22d3ee', '#6366f1', '#34d399', '#f59e0b', '#fb7185', '#64748b'];
    inst.setOption({
      backgroundColor: 'transparent',
      textStyle: { color: '#8aa4ba', fontSize: 11 },
      tooltip: { trigger: 'axis', backgroundColor: 'rgba(8,18,32,.92)', borderColor: 'rgba(34,211,238,.3)', textStyle: { color: '#e5edf8', fontSize: 12 } },
      grid: { left: 40, right: 16, top: 10, bottom: 24 },
      xAxis: { type: 'category', data: lanes, axisLabel: { color: '#8aa4ba', fontSize: 10 }, axisLine: { lineStyle: { color: '#334155' } } },
      yAxis: { type: 'value', name: '次', splitLine: { lineStyle: { color: 'rgba(148,163,184,.08)' } }, axisLabel: { color: '#8aa4ba', fontSize: 10 } },
      series: [{
        type: 'bar', data: counts,
        itemStyle: { color: function(p) { return colors[p.dataIndex % colors.length]; }, borderRadius: [3, 3, 0, 0] },
        barWidth: '55%',
      }],
    });
  }

  function destroy() {
    window.removeEventListener('resize', ChartFactory.resizeAll);
    ChartFactory.disposeAll();
  }

  return { render, destroy };
})();

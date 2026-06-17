/* ═══════════════════════════════════════════════════════════════════
   components/charts.js — ECharts option 工厂 + 实例管理
   ═══════════════════════════════════════════════════════════════════ */
window.ChartFactory = (() => {

  const _instances = {};

  /** 创建/更新 ECharts 实例 */
  function _bind(domId) {
    const dom = document.getElementById(domId);
    if (!dom) return null;
    let inst = _instances[domId];
    if (!inst || inst.isDisposed()) {
      inst = echarts.init(dom, null, { renderer: 'canvas' });
      _instances[domId] = inst;
    }
    return inst;
  }

  function _darkBase() {
    return {
      backgroundColor: 'transparent',
      textStyle: { color: '#8aa4ba' },
    };
  }

  /** 车型分布饼图 */
  function classPie(domId, data, meta) {
    const inst = _bind(domId);
    if (!inst) return;
    const colors = meta ? meta.class_colors : {};
    const pieData = (data || []).map(d => ({
      name: d.name,
      value: d.value,
      itemStyle: { color: colors[d.name] || '#64748b' },
    }));
    const option = {
      ..._darkBase(),
      tooltip: {
        trigger: 'item',
        formatter: '{b}: {c} 次 ({d}%)',
        backgroundColor: 'rgba(8,18,32,.92)',
        borderColor: 'rgba(34,211,238,.3)',
        textStyle: { color: '#e5edf8', fontSize: 12 },
      },
      series: [{
        type: 'pie',
        radius: ['45%', '72%'],
        center: ['50%', '52%'],
        label: { color: '#8aa4ba', fontSize: 11 },
        labelLine: { lineStyle: { color: '#475569' } },
        data: pieData,
        emphasis: {
          itemStyle: { shadowBlur: 14, shadowColor: 'rgba(34,211,238,.5)' },
        },
      }],
    };
    inst.setOption(option, true);
  }

  /** 速度分布直方图 */
  function speedHistogram(domId, data) {
    const inst = _bind(domId);
    if (!inst) return;
    const option = {
      ..._darkBase(),
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'rgba(8,18,32,.92)',
        borderColor: 'rgba(34,211,238,.3)',
        textStyle: { color: '#e5edf8', fontSize: 12 },
      },
      grid: { left: 40, right: 16, top: 16, bottom: 28 },
      xAxis: {
        type: 'category',
        data: data ? data.bins : [],
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#8aa4ba', fontSize: 10 },
      },
      yAxis: {
        type: 'value',
        name: '辆',
        axisLine: { show: false },
        splitLine: { lineStyle: { color: 'rgba(148,163,184,.08)' } },
        axisLabel: { color: '#8aa4ba', fontSize: 10 },
      },
      series: [{
        type: 'bar',
        data: data ? data.counts : [],
        itemStyle: {
          color: {
            type: 'linear', x: 0, y: 0, x2: 0, y2: 1,
            colorStops: [
              { offset: 0, color: '#22d3ee' },
              { offset: 1, color: 'rgba(34,211,238,.2)' },
            ],
          },
          borderRadius: [3, 3, 0, 0],
        },
        barWidth: '60%',
        emphasis: {
          itemStyle: { shadowBlur: 10, shadowColor: 'rgba(34,211,238,.4)' },
        },
      }],
    };
    inst.setOption(option, true);
  }

  /** 断面流量柱状图 */
  function sectionBar(domId, data) {
    const inst = _bind(domId);
    if (!inst) return;
    const DIR_COLORS = {
      '到达': '#22d3ee',
      '离去': '#6366f1',
      '右转': '#f59e0b',
      '直行': '#10b981',
      '掉头': '#ef4444',
    };
    const series = (data ? data.series : []).map(s => ({
      name: s.name,
      type: 'bar',
      data: s.data,
      itemStyle: { color: DIR_COLORS[s.name] || '#64748b', borderRadius: [2,2,0,0] },
      barGap: '20%',
    }));
    const option = {
      ..._darkBase(),
      tooltip: {
        trigger: 'axis',
        axisPointer: { type: 'shadow' },
        backgroundColor: 'rgba(8,18,32,.92)',
        borderColor: 'rgba(34,211,238,.3)',
        textStyle: { color: '#e5edf8', fontSize: 12 },
      },
      legend: {
        data: data ? data.series.map(s => s.name) : [],
        textStyle: { color: '#8aa4ba', fontSize: 11 },
        top: 0,
      },
      grid: { left: 40, right: 16, top: 30, bottom: 28 },
      xAxis: {
        type: 'category',
        data: data ? data.categories : [],
        axisLine: { lineStyle: { color: '#334155' } },
        axisLabel: { color: '#8aa4ba', fontSize: 9, rotate: 15 },
      },
      yAxis: {
        type: 'value',
        name: '次',
        axisLine: { show: false },
        splitLine: { lineStyle: { color: 'rgba(148,163,184,.08)' } },
        axisLabel: { color: '#8aa4ba', fontSize: 10 },
      },
      series: series,
    };
    inst.setOption(option, true);
  }

  /** 精度对比横向柱状图 */
  function validationBar(domId, summary) {
    const inst = _bind(domId);
    if (!inst) return;
    const labels = window.CONST.METRIC_LABELS;
    const metrics = ['event_precision','event_recall','lane_accuracy','direction_accuracy','class_accuracy','spacing_consistency_pass_rate'];
    const names = metrics.map(m => labels[m] || m);
    const values = metrics.map(m => summary[m] !== undefined ? +(summary[m] * 100).toFixed(1) : 0);
    const THRESH = window.CONST.METRIC_THRESHOLDS;

    const option = {
      ..._darkBase(),
      tooltip: { trigger: 'axis', axisPointer: { type: 'shadow' } },
      grid: { left: 100, right: 30, top: 10, bottom: 10 },
      xAxis: {
        type: 'value', max: 100,
        axisLabel: { color: '#8aa4ba', fontSize: 10, formatter: '{value}%' },
        splitLine: { lineStyle: { color: 'rgba(148,163,184,.08)' } },
      },
      yAxis: {
        type: 'category', data: names,
        axisLabel: { color: '#8aa4ba', fontSize: 10 },
        axisLine: { show: false },
      },
      series: [{
        type: 'bar', data: values,
        itemStyle: {
          color: function(params) {
            const v = values[params.dataIndex] / 100;
            if (v >= THRESH.good) return '#34d399';
            if (v >= THRESH.warn) return '#f59e0b';
            return '#fb7185';
          },
          borderRadius: [0, 3, 3, 0],
        },
        barWidth: '50%',
        label: { show: true, position: 'right', color: '#8aa4ba', fontSize: 10, formatter: '{c}%' },
      }],
    };
    inst.setOption(option, true);
  }

  /** resize 全部实例 */
  function resizeAll() {
    Object.values(_instances).forEach(inst => {
      if (inst && !inst.isDisposed()) inst.resize();
    });
  }

  /** dispose 全部实例 */
  function disposeAll() {
    Object.values(_instances).forEach(inst => {
      if (inst && !inst.isDisposed()) inst.dispose();
    });
    Object.keys(_instances).forEach(k => delete _instances[k]);
  }

  return { classPie, speedHistogram, sectionBar, validationBar, resizeAll, disposeAll };

})();

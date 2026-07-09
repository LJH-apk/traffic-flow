/* ═══════════════════════════════════════════════════════════════════
   constants.js — 前端常量与兜底值
   ═══════════════════════════════════════════════════════════════════ */
window.CONST = {

  API_BASE: '/api/dashboard',

  // 视频路径
  VIDEO_URL: '/outputs/trajectory.mp4',

  // 三进口独立视频
  VIDEO_URLS: {
    north:  '/outputs/trajectory_north_full.mp4',
    east:   '/outputs/trajectory_east_full.mp4',
    south:  '/outputs/trajectory_south_full.mp4',
    merged: '/outputs/trajectory.mp4',
  },

  VIDEO_SCOPE_MAP: {
    '/outputs/trajectory_north_full.mp4': 'north',
    '/outputs/trajectory_east_full.mp4': 'east',
    '/outputs/trajectory_south_full.mp4': 'south',
    '/outputs/trajectory.mp4': 'merged',
  },

  // 默认值
  FALLBACKS: {
    lane: 'UNKNOWN',
    direction: '未知',
    className: 'unknown',
    color: '未知',
    speed: '--',
    time: '--:--',
  },

  // ECharts 暗色主题配置
  CHART_DARK: {
    backgroundColor: 'transparent',
    textStyle: { color: '#8aa4ba' },
    legend: { textStyle: { color: '#8aa4ba' } },
    tooltip: { backgroundColor: 'rgba(8,18,32,.92)', borderColor: 'rgba(34,211,238,.3)', textStyle: { color: '#e5edf8' } },
  },

  // 关键指标中文标签
  METRIC_LABELS: {
    event_precision: '事件精确率',
    event_recall: '事件召回率',
    event_f1: '事件 F1',
    lane_accuracy: '车道准确率',
    direction_accuracy: '方向准确率',
    class_accuracy: '分类准确率',
    crossing_time_mae_s: '过车时间 MAE(s)',
    headway_mae_s: '车头时距 MAE(s)',
    spacing_consistency_pass_rate: '间距一致性',
    total_anomaly_count: '异常事件数',
  },

  // 指标阈值（好/警告/差）
  METRIC_THRESHOLDS: {
    good: 0.80,
    warn: 0.60,
  },

};

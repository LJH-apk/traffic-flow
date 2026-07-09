/* ═══════════════════════════════════════════════════════════════════
   main.js — 应用入口：初始化路由、顶栏导航、全局错误处理
   ═══════════════════════════════════════════════════════════════════ */
(function () {

  // ── 注册路由：答辩演示页(1-4) + 原有页面(5-7) ──
  Router.register('detection_demo', DetectionDemoView);
  Router.register('tracking_demo', TrackingDemoView);
  Router.register('traffic_stats', TrafficStatsView);
  Router.register('visualization', VisualizationView);
  Router.register('demo', DemoView);
  Router.register('live', LiveView);
  Router.register('dashboard', DashboardView);

  // ── 顶栏导航按钮 ──
  document.getElementById('topbarNav').addEventListener('click', function (e) {
    const btn = e.target.closest('.nav-btn');
    if (!btn) return;
    const route = btn.dataset.route;
    if (route) Router.navigate(route);
  });

  // ── 键盘快捷键 ──
  document.addEventListener('keydown', function (e) {
    if (e.ctrlKey || e.metaKey || document.activeElement !== document.body) return;
    var map = {
      '1': 'detection_demo',
      '2': 'tracking_demo',
      '3': 'traffic_stats',
      '4': 'visualization',
      '5': 'demo',
      '6': 'live',
      '7': 'dashboard',
    };
    if (map[e.key]) Router.navigate(map[e.key]);
  });

  // ── 全局错误监听 ──
  window.addEventListener('error', function (e) {
    console.error('Global error:', e.error || e.message);
  });

  window.addEventListener('unhandledrejection', function (e) {
    console.error('Unhandled promise rejection:', e.reason);
  });

  // ── 启动 ──
  console.log('交通流检测平台 v2 启动');
  Router.start();

})();

/* ═══════════════════════════════════════════════════════════════════
   main.js — 应用入口：初始化路由、顶栏导航、全局错误处理
   ═══════════════════════════════════════════════════════════════════ */
(function () {

  // ── 注册路由 ──
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
    // 数字键 1 → 演示模式，2 → 实时检测，3 → 数据看板
    if (e.key === '1' && !e.ctrlKey && !e.metaKey && document.activeElement === document.body) {
      Router.navigate('demo');
    } else if (e.key === '2' && !e.ctrlKey && !e.metaKey && document.activeElement === document.body) {
      Router.navigate('live');
    } else if (e.key === '3' && !e.ctrlKey && !e.metaKey && document.activeElement === document.body) {
      Router.navigate('dashboard');
    }
  });

  // ── 全局错误监听 ──
  window.addEventListener('error', function (e) {
    console.error('Global error:', e.error || e.message);
    // 出错时不打断用户，仅在控制台记录
  });

  window.addEventListener('unhandledrejection', function (e) {
    console.error('Unhandled promise rejection:', e.reason);
  });

  // ── 启动 ──
  console.log('交通流检测平台 v2 启动');
  Router.start();

})();

/* ═══════════════════════════════════════════════════════════════════
   router.js — Hash 路由切换
   ═══════════════════════════════════════════════════════════════════ */
window.Router = (() => {

  const routes = {};
  let currentView = null;

  function register(name, handler) {
    routes[name] = handler;
  }

  function navigate(name) {
    if (window.location.hash !== '#' + name) {
      window.location.hash = '#' + name;
    } else {
      // 手动触发
      _activate(name);
    }
  }

  function _activate(name) {
    if (currentView === name && name !== '') return;
    const handler = routes[name];
    if (!handler) {
      // fallback to demo
      if (routes['demo']) { _activate('demo'); return; }
      console.warn('Unknown route:', name);
      return;
    }

    // 清理旧视图
    if (currentView && routes[currentView] && routes[currentView].destroy) {
      routes[currentView].destroy();
    }
    currentView = name;
    Store.set('route', name);

    // 更新导航按钮
    document.querySelectorAll('.nav-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.route === name);
    });

    // 渲染新视图
    handler.render();
  }

  function onHashChange() {
    const hash = window.location.hash.replace('#', '') || 'demo';
    _activate(hash);
  }

  function start() {
    window.addEventListener('hashchange', onHashChange);
    // 初始路由
    const initial = window.location.hash.replace('#', '') || 'demo';
    _activate(initial);
  }

  function current() { return currentView; }

  return { register, navigate, start, current };

})();

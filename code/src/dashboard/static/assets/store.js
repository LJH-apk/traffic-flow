/* ═══════════════════════════════════════════════════════════════════
   store.js — 全局状态管理
   ═══════════════════════════════════════════════════════════════════ */
window.Store = (() => {

  const state = {
    route: '',
    // 数据缓存
    meta: null,
    timeline: null,
    events: null,
    overview: null,
    charts: null,
    trajectories: null,
    validation: null,
    // 演示模式
    demoLoaded: false,
    demoCurrentSec: 0,
    // 看板模式
    dashLoaded: false,
    // 错误状态
    error: null,
  };

  const listeners = {};

  function get(key) { return state[key]; }
  function set(key, value) {
    const prev = state[key];
    state[key] = value;
    if (listeners[key]) {
      listeners[key].forEach(fn => fn(value, prev));
    }
  }

  function on(key, fn) {
    if (!listeners[key]) listeners[key] = [];
    listeners[key].push(fn);
    return () => {
      listeners[key] = listeners[key].filter(f => f !== fn);
    };
  }

  function snapshot() { return { ...state }; }

  return { get, set, on, snapshot };

})();

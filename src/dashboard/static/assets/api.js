/* ═══════════════════════════════════════════════════════════════════
   api.js — 前端数据获取层
   ═══════════════════════════════════════════════════════════════════ */
window.API = (() => {

  const BASE = window.CONST.API_BASE;

  /** 缓存已加载的 JSON，避免重复请求 */
  const _cache = {};

  async function _fetchJSON(url) {
    if (_cache[url]) return _cache[url];
    const res = await fetch(url);
    if (!res.ok) throw new Error(url + ' ' + res.status);
    const data = await res.json();
    _cache[url] = data;
    return data;
  }

  function _endpoint(name) { return BASE + '/' + name; }

  async function loadMeta()        { return _fetchJSON(_endpoint('meta')); }
  async function loadTimeline()    { return _fetchJSON(_endpoint('timeline')); }
  async function loadEvents()      { return _fetchJSON(_endpoint('events')); }
  async function loadOverview()    { return _fetchJSON(_endpoint('overview')); }
  async function loadCharts()      { return _fetchJSON(_endpoint('charts')); }
  async function loadTrajectories(){ return _fetchJSON(_endpoint('trajectories')); }
  async function loadValidation()  { return _fetchJSON(_endpoint('validation')); }

  /** 并行加载演示模式所需的全部数据 */
  async function loadDemoData() {
    const [meta, timeline] = await Promise.all([
      loadMeta(), loadTimeline()
    ]);
    return { meta, timeline };
  }

  /** 并行加载看板模式所需的全部数据 */
  async function loadDashboardData() {
    const [overview, charts, events, trajectories, validation] = await Promise.all([
      loadOverview(), loadCharts(), loadEvents(), loadTrajectories(), loadValidation()
    ]);
    return { overview, charts, events, trajectories, validation };
  }

  function clearCache() { Object.keys(_cache).forEach(k => delete _cache[k]); }

  return {
    loadMeta, loadTimeline, loadEvents, loadOverview, loadCharts,
    loadTrajectories, loadValidation,
    loadDemoData, loadDashboardData,
    clearCache,
  };

})();

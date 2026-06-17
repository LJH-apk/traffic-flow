/* ═══════════════════════════════════════════════════════════════════
   components/event_list.js — 最近过车事件列表
   ═══════════════════════════════════════════════════════════════════ */
window.EventList = (() => {

  function render(containerId, events, opts) {
    opts = opts || {};
    const limit = opts.limit || 8;
    const container = document.getElementById(containerId);
    if (!container) return;

    const items = (events || []).slice(0, limit);
    if (!items.length) {
      container.innerHTML = '<div class="loading-placeholder">暂无过车事件</div>';
      return;
    }

    container.innerHTML = items.map((ev, idx) => {
      const title = (ev.section || '未知断面') + ' · ' + (ev.direction || '未知');
      const speed = FMT.speed(ev.speed_kmh) + ' km/h';
      const meta = FMT.time(ev.timestamp_s) + ' | ' + (ev.class_name || '') + (ev.lane_id ? ' · 车道' + ev.lane_id : '') + (ev.plate ? ' · ' + ev.plate : '');
      const freshClass = idx === 0 && opts.highlightFirst ? ' ev-item--fresh' : '';
      return `<div class="ev-item${freshClass}">
        <div>
          <div class="ev-title">${title}</div>
          <div class="ev-meta">${meta}</div>
        </div>
        <div class="ev-speed">${speed}</div>
      </div>`;
    }).join('');
  }

  return { render };

})();

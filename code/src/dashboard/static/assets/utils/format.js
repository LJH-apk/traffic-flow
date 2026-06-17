/* ═══════════════════════════════════════════════════════════════════
   utils/format.js — 数字 / 时间 / 速度格式化
   ═══════════════════════════════════════════════════════════════════ */
window.FMT = (() => {

  function time(seconds) {
    const s = Number(seconds) || 0;
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
  }

  function timeFull(seconds) {
    const s = Number(seconds) || 0;
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = Math.floor(s % 60);
    if (h > 0) return h + ':' + String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
    return String(m).padStart(2, '0') + ':' + String(sec).padStart(2, '0');
  }

  function number(num) {
    return Number(num || 0).toLocaleString('zh-CN');
  }

  function speed(kmh, digits) {
    if (digits === undefined) digits = 1;
    const v = Number(kmh) || 0;
    return v.toFixed(digits);
  }

  function percent(v, digits) {
    if (digits === undefined) digits = 1;
    return (Number(v || 0) * 100).toFixed(digits) + '%';
  }

  function duration(beginS, endS) {
    const d = Number(endS || 0) - Number(beginS || 0);
    if (d < 60) return Math.round(d) + 's';
    if (d < 3600) return (d / 60).toFixed(1) + 'min';
    return (d / 3600).toFixed(1) + 'h';
  }

  return { time, timeFull, number, speed, percent, duration };

})();

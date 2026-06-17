/* ═══════════════════════════════════════════════════════════════════
   utils/animate.js — 数字缓动 / DOM 过渡
   ═══════════════════════════════════════════════════════════════════ */
window.Animate = (() => {

  /** cubic ease-out */
  function easeOut(t) {
    return 1 - Math.pow(1 - t, 3);
  }

  /** 数字缓动过渡：从当前值渐变到目标值，duration 毫秒 */
  function interpolate(el, target, opts) {
    opts = opts || {};
    const digits = opts.digits !== undefined ? opts.digits : Number.isInteger(target) ? 0 : 1;
    const duration = opts.duration || 320;
    const suffix = opts.suffix || '';
    const prefix = opts.prefix || '';
    const useLocale = opts.locale !== false;

    const curText = el.textContent.replace(prefix, '').replace(suffix, '').replace(/,/g, '');
    const from = parseFloat(curText) || 0;
    const to = Number(target) || 0;

    if (from === to) return;

    const start = performance.now();
    function step(now) {
      const p = Math.min(1, (now - start) / duration);
      const val = from + (to - from) * easeOut(p);
      let text;
      if (useLocale) {
        text = digits > 0 ? val.toFixed(digits) : Math.round(val).toLocaleString('zh-CN');
      } else {
        text = digits > 0 ? val.toFixed(digits) : String(Math.round(val));
      }
      el.textContent = prefix + text + suffix;
      if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  /** 淡入元素 */
  function fadeIn(el, duration) {
    duration = duration || 300;
    el.style.opacity = 0;
    el.style.transition = 'opacity ' + duration + 'ms ease-out';
    requestAnimationFrame(() => { el.style.opacity = 1; });
  }

  /** 闪动效果（给元素加 class 再移除） */
  function pulse(el, className, duration) {
    className = className || 'kpi-pulse';
    duration = duration || 600;
    el.classList.remove(className);
    void el.offsetWidth;
    el.classList.add(className);
    setTimeout(() => el.classList.remove(className), duration);
  }

  return { easeOut, interpolate, fadeIn, pulse };

})();

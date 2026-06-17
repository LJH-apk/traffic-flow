/* ═══════════════════════════════════════════════════════════════════
   utils/dom.js — DOM 操作工具箱
   ═══════════════════════════════════════════════════════════════════ */
window.DOM = (() => {

  function $(sel, parent) { return (parent || document).querySelector(sel); }
  function $$(sel, parent) { return Array.from((parent || document).querySelectorAll(sel)); }

  function el(tag, attrs, children) {
    const e = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(k => {
        if (k === 'class') { e.className = attrs[k]; }
        else if (k === 'style' && typeof attrs[k] === 'object') { Object.assign(e.style, attrs[k]); }
        else if (k.startsWith('on')) { e.addEventListener(k.slice(2).toLowerCase(), attrs[k]); }
        else { e.setAttribute(k, attrs[k]); }
      });
    }
    if (children) {
      if (typeof children === 'string') { e.textContent = children; }
      else if (Array.isArray(children)) { children.forEach(c => e.appendChild(c)); }
      else { e.appendChild(children); }
    }
    return e;
  }

  function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); }
  function show(el) { el.classList.remove('hidden'); }
  function hide(el) { el.classList.add('hidden'); }
  function toggle(el, cond) { el.classList.toggle('hidden', !cond); }

  return { $, $$, el, clear, show, hide, toggle };

})();

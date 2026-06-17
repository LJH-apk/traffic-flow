/* ═══════════════════════════════════════════════════════════════════
   utils/color.js — 类别 / 方向颜色映射
   ═══════════════════════════════════════════════════════════════════ */
window.CMap = (() => {

  const DEFAULT_CLASS = '#64748b';
  const DEFAULT_DIR = '#22d3ee';

  const _class = {};
  const _direction = {};

  function init(meta) {
    // 从 meta.json 读入颜色映射
    if (meta && meta.class_colors) {
      Object.assign(_class, meta.class_colors);
    }
    if (meta && meta.direction_colors) {
      Object.assign(_direction, meta.direction_colors);
    }
  }

  function forClass(name) {
    return _class[name] || DEFAULT_CLASS;
  }

  function forDirection(name) {
    return _direction[name] || DEFAULT_DIR;
  }

  function allClasses() { return Object.keys(_class); }
  function allDirections() { return Object.keys(_direction); }

  return { init, forClass, forDirection, allClasses, allDirections };

})();

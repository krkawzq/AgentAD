/** Pure state and numeric helpers shared by the AgentAD WebUI. */

export const MIN_SPAN = 4;
export const TRACK_HEIGHT = 96;
export const CHART_VERTICAL_PADDING = 36;
export const DEFAULT_VIEW_OPTIONS = Object.freeze({
  normalization: "none",
  scope: "feature",
  labelIndex: null,
  mode: "pan",
  layout: "stacked",
  transform: Object.freeze({
    rotation: 0,
    scaleX: 1,
    scaleY: 1,
    flipX: false,
    flipY: false,
  }),
});
export const DEFAULT_TAB_PREFERENCES = Object.freeze({
  ...DEFAULT_VIEW_OPTIONS,
  labelName: null,
  inspectorOpen: false,
  inspectorWidth: 320,
});

function clampFinite(value, minimum, maximum, fallback) {
  const number = Number(value);
  if (!Number.isFinite(number)) return fallback;
  return Math.max(minimum, Math.min(number, maximum));
}

export function parseTabPreferences(value) {
  let source = value;
  if (typeof source === "string") {
    try {
      source = JSON.parse(source);
    } catch {
      source = null;
    }
  }
  if (!source || typeof source !== "object" || Array.isArray(source)) source = {};
  const transform =
    source.transform && typeof source.transform === "object" ? source.transform : {};
  return {
    normalization:
      typeof source.normalization === "string" && source.normalization
        ? source.normalization
        : DEFAULT_TAB_PREFERENCES.normalization,
    scope: source.scope === "global" ? "global" : DEFAULT_TAB_PREFERENCES.scope,
    labelIndex: null,
    labelName:
      typeof source.labelName === "string" && source.labelName ? source.labelName : null,
    mode: source.mode === "select" ? "select" : DEFAULT_TAB_PREFERENCES.mode,
    layout: source.layout === "overlay" ? "overlay" : DEFAULT_TAB_PREFERENCES.layout,
    transform: {
      rotation: clampFinite(transform.rotation, -180, 180, 0),
      scaleX: clampFinite(transform.scaleX, 0.25, 2, 1),
      scaleY: clampFinite(transform.scaleY, 0.25, 4, 1),
      flipX: transform.flipX === true,
      flipY: transform.flipY === true,
    },
    inspectorOpen: source.inspectorOpen === true,
    inspectorWidth: clampFinite(source.inspectorWidth, 260, 560, 320),
  };
}

export function serializeTabPreferences(preferences) {
  return JSON.stringify({ version: 1, ...parseTabPreferences(preferences) });
}

export function normalizeWheelDelta(deltaY, deltaX, deltaMode = 0) {
  const vertical = Number(deltaY) || 0;
  const horizontal = Number(deltaX) || 0;
  const dominant = Math.abs(vertical) >= Math.abs(horizontal) ? vertical : horizontal;
  const unit = deltaMode === 1 ? 16 : deltaMode === 2 ? 120 : 1;
  return Math.max(-240, Math.min(240, dominant * unit));
}

export function zoomScale(scale, wheelDelta, minimum = 0.25, maximum = 4) {
  const current = clampFinite(scale, minimum, maximum, 1);
  const delta = Number(wheelDelta) || 0;
  const next = current * Math.exp(-delta * 0.0015);
  return Number(Math.max(minimum, Math.min(next, maximum)).toFixed(4));
}

export function positionTooltip(pointer, viewport, tooltip, padding = 8, gap = 14) {
  const width = Math.max(0, Number(viewport?.width) || 0);
  const height = Math.max(0, Number(viewport?.height) || 0);
  const tooltipWidth = Math.max(0, Number(tooltip?.width) || 0);
  const tooltipHeight = Math.max(0, Number(tooltip?.height) || 0);
  const x = Math.max(0, Math.min(Number(pointer?.x) || 0, width));
  const y = Math.max(0, Math.min(Number(pointer?.y) || 0, height));

  let horizontal = "right";
  let left = x + gap;
  if (left + tooltipWidth > width - padding) {
    horizontal = "left";
    left = x - gap - tooltipWidth;
  }
  left = Math.max(padding, Math.min(left, Math.max(padding, width - tooltipWidth - padding)));

  let vertical = "center";
  let top = y - tooltipHeight / 2;
  if (y < height / 3) {
    vertical = "below";
    top = y + gap;
  } else if (y > (height * 2) / 3) {
    vertical = "above";
    top = y - gap - tooltipHeight;
  }
  top = Math.max(padding, Math.min(top, Math.max(padding, height - tooltipHeight - padding)));
  return { x: left, y: top, horizontal, vertical };
}

export function chartCanvasHeight(featureCount, layout) {
  const count = Math.max(1, Math.floor(Number(featureCount) || 0));
  const lanes = layout === "overlay" ? 1 : count;
  return lanes * TRACK_HEIGHT + CHART_VERTICAL_PADDING;
}

export function replaceTreeChildren(nodes, key, children) {
  for (let index = 0; index < nodes.length; index += 1) {
    const node = nodes[index];
    let nextNode = null;
    if (node.key === key) {
      nextNode = { ...node, children, isLeaf: children.length === 0 };
    } else if (Array.isArray(node.children)) {
      const nextChildren = replaceTreeChildren(node.children, key, children);
      if (nextChildren !== node.children) nextNode = { ...node, children: nextChildren };
    }
    if (nextNode) {
      const next = nodes.slice();
      next[index] = nextNode;
      return next;
    }
  }
  return nodes;
}

export function clampPanelWidth(width, minimum = 240, maximum = 520) {
  const min = Math.max(0, Number(minimum) || 0);
  const max = Math.max(min, Number(maximum) || min);
  const value = Number(width);
  if (!Number.isFinite(value)) return min;
  return Math.round(Math.max(min, Math.min(value, max)));
}

export function clampWindow(start, stop, total) {
  const size = Math.max(0, Math.floor(Number(total) || 0));
  if (size === 0) return [0, 0];

  const minSpan = Math.min(MIN_SPAN, size);
  let s = Number(start);
  let e = Number(stop);
  if (!Number.isFinite(s)) s = 0;
  if (!Number.isFinite(e)) e = size;
  s = Math.floor(s);
  e = Math.floor(e);
  s = Math.max(0, Math.min(s, size - minSpan));
  e = Math.max(s + minSpan, Math.min(e, size));
  return [s, e];
}

export function formatNumber(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  if (value === 0) return "0";
  const abs = Math.abs(value);
  if (abs >= 1e6 || abs < 1e-3) return value.toExponential(2);
  return String(Number(value.toPrecision(4)));
}

export function nearestSampleIndex(indices, target) {
  let lo = 0;
  let hi = indices.length - 1;
  if (hi < 0) return -1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (indices[mid] < target) lo = mid + 1;
    else hi = mid;
  }
  if (
    lo > 0 &&
    Math.abs(indices[lo - 1] - target) < Math.abs(indices[lo] - target)
  ) {
    return lo - 1;
  }
  return lo;
}

export function panWindow(window, delta, total) {
  const [start, stop] = clampWindow(window[0], window[1], total);
  const span = stop - start;
  if (span >= total) return [0, Math.max(0, total)];
  const shift = Math.round(Number(delta) || 0);
  let nextStart = start + shift;
  nextStart = Math.max(0, Math.min(nextStart, total - span));
  return [nextStart, nextStart + span];
}

export function zoomWindow(window, factor, anchor, total) {
  const [start, stop] = clampWindow(window[0], window[1], total);
  if (total <= 0) return [0, 0];
  const span = stop - start;
  const safeFactor = Number.isFinite(factor) && factor > 0 ? factor : 1;
  const nextSpan = Math.max(
    Math.min(MIN_SPAN, total),
    Math.min(total, Math.round(span * safeFactor)),
  );
  const safeAnchor = Number.isFinite(anchor) ? anchor : (start + stop) / 2;
  const fraction = span > 0 ? (safeAnchor - start) / span : 0.5;
  const nextStart = Math.round(safeAnchor - fraction * nextSpan);
  return clampWindow(nextStart, nextStart + nextSpan, total);
}

export function transformPoint2D(x, y, transform, centerX, centerY) {
  const angle = ((Number(transform.rotation) || 0) * Math.PI) / 180;
  const scaleX = (Number(transform.scaleX) || 1) * (transform.flipX ? -1 : 1);
  const scaleY = (Number(transform.scaleY) || 1) * (transform.flipY ? -1 : 1);
  const dx = (x - centerX) * scaleX;
  const dy = (y - centerY) * scaleY;
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [
    centerX + cosine * dx - sine * dy,
    centerY + sine * dx + cosine * dy,
  ];
}

export function buildPolyline2D(
  indices,
  values,
  mapX,
  mapY,
  transform,
  centerX,
  centerY,
) {
  const points = new Float64Array(indices.length * 2);
  for (let position = 0; position < indices.length; position += 1) {
    const value = values[position];
    if (value === null || value === undefined || !Number.isFinite(value)) {
      points[position * 2] = Number.NaN;
      points[position * 2 + 1] = Number.NaN;
      continue;
    }
    const x = mapX(indices[position]);
    const y = mapY(value);
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      points[position * 2] = Number.NaN;
      points[position * 2 + 1] = Number.NaN;
      continue;
    }
    const transformed = transformPoint2D(
      x,
      y,
      transform,
      centerX,
      centerY,
    );
    points[position * 2] = transformed[0];
    points[position * 2 + 1] = transformed[1];
  }
  return points;
}

export function parseHashState(hash) {
  const raw = typeof hash === "string" ? hash.replace(/^#/, "") : "";
  const params = new URLSearchParams(raw);
  const out = {
    s: null,
    f: null,
    n: "none",
    sc: "feature",
    l: null,
    w: null,
  };
  const uint = (name) => {
    const value = params.get(name);
    if (value === null || value === "") return null;
    const parsed = Number(value);
    return Number.isInteger(parsed) && parsed >= 0 ? parsed : null;
  };

  out.s = uint("s");
  const features = params.get("f");
  if (features) {
    const list = features
      .split(",")
      .map((part) => Number(part))
      .filter((value) => Number.isInteger(value) && value >= 0);
    if (list.length) out.f = [...new Set(list)];
  }
  const normalization = params.get("n");
  if (normalization) out.n = normalization;
  const scope = params.get("sc");
  if (scope === "global") out.sc = "global";
  out.l = uint("l");

  const window = params.get("w");
  if (window) {
    const parts = window.split("-").map((part) => Number(part));
    if (
      parts.length === 2 &&
      parts.every((value) => Number.isInteger(value) && value >= 0)
    ) {
      out.w = [parts[0], parts[1]];
    }
  }
  return out;
}

export function serializeHashState(fields) {
  const params = new URLSearchParams();
  if (fields.s != null) params.set("s", String(fields.s));
  if (fields.f != null && fields.f.length) {
    params.set("f", fields.f.join(","));
  }
  if (fields.n && fields.n !== "none") params.set("n", fields.n);
  if (fields.sc && fields.sc !== "feature") params.set("sc", fields.sc);
  if (fields.l != null) params.set("l", String(fields.l));
  if (fields.w) params.set("w", `${fields.w[0]}-${fields.w[1]}`);
  return params.toString();
}

export function valueDomain(values) {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (typeof value !== "number" || !Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (!Number.isFinite(min)) return null;

  const scale = Math.max(Math.abs(min), Math.abs(max)) || 1;
  const scaledMin = min / scale;
  const scaledMax = max / scale;
  const range = scaledMax - scaledMin;
  const padding = range > 0 ? Math.max(range * 0.08, Number.EPSILON * 8) : 0.08;
  return {
    min,
    max,
    scale,
    low: scaledMin - padding,
    high: scaledMax + padding,
  };
}

export function valueToY(value, domain, top, bottom) {
  if (!domain || typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  const scaled = value / domain.scale;
  return bottom - ((scaled - domain.low) / (domain.high - domain.low)) * (bottom - top);
}

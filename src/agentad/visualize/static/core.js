/** Pure state and numeric helpers shared by the AgentAD WebUI. */

export const MIN_SPAN = 4;

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

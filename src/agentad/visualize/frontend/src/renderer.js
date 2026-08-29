import {
  buildPolyline2D,
  formatNumber,
  nearestSampleIndex,
  valueDomain,
  valueToY,
} from "../../static/core.js";

export const TRACK_COLORS = [
  "#1769aa",
  "#00897b",
  "#7b5ea7",
  "#c05a24",
  "#3f7d20",
  "#a43d65",
  "#2f6f89",
  "#8b6914",
];

const INK = "#223047";
const MUTED = "#607089";
const GRID = "#e7ebf1";
const GRID_STRONG = "#d8dee8";
const BAND = "rgba(187, 50, 50, 0.08)";
const BAND_EDGE = "rgba(187, 50, 50, 0.28)";

export function chartGeometry(width, height, featureCount, layout) {
  const left = 148;
  const right = Math.max(left + 40, width - 20);
  const top = 18;
  const bottom = Math.max(top + 30, height - 34);
  const lanes = layout === "overlay" ? 1 : Math.max(1, featureCount);
  const laneHeight = (bottom - top) / lanes;
  return { left, right, top, bottom, lanes, laneHeight };
}

function textEllipsis(context, text, width) {
  if (context.measureText(text).width <= width) return text;
  let lower = 0;
  let upper = text.length;
  while (lower < upper) {
    const middle = Math.ceil((lower + upper) / 2);
    if (context.measureText(`${text.slice(0, middle)}…`).width <= width) {
      lower = middle;
    } else {
      upper = middle - 1;
    }
  }
  return `${text.slice(0, lower)}…`;
}

function combinedDomain(features) {
  let minimum = Infinity;
  let maximum = -Infinity;
  for (const feature of features) {
    const domain = valueDomain(feature.values);
    if (!domain) continue;
    minimum = Math.min(minimum, domain.min);
    maximum = Math.max(maximum, domain.max);
  }
  return Number.isFinite(minimum) ? valueDomain([minimum, maximum]) : null;
}

function drawGrid(context, geometry, laneTop, laneBottom) {
  context.strokeStyle = GRID;
  context.lineWidth = 1;
  for (const fraction of [0.25, 0.5, 0.75]) {
    const y = Math.round(laneTop + (laneBottom - laneTop) * fraction) + 0.5;
    context.beginPath();
    context.moveTo(geometry.left, y);
    context.lineTo(geometry.right, y);
    context.stroke();
  }
  context.strokeStyle = GRID_STRONG;
  context.beginPath();
  context.moveTo(geometry.left, laneBottom + 0.5);
  context.lineTo(geometry.right, laneBottom + 0.5);
  context.stroke();
}

function drawPolyline(context, points, color) {
  context.strokeStyle = color;
  context.lineWidth = 1.55;
  context.lineJoin = "round";
  context.lineCap = "round";
  context.beginPath();
  let drawing = false;
  for (let index = 0; index < points.length; index += 2) {
    const x = points[index];
    const y = points[index + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      drawing = false;
      continue;
    }
    if (drawing) context.lineTo(x, y);
    else context.moveTo(x, y);
    drawing = true;
  }
  context.stroke();
}

function drawTimeAxis(context, data, viewport, geometry) {
  if (!data.indices.length) return;
  const span = Math.max(1, viewport[1] - viewport[0]);
  const plotWidth = geometry.right - geometry.left;
  context.fillStyle = MUTED;
  context.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textAlign = "center";
  context.textBaseline = "top";
  const tickCount = plotWidth < 500 ? 4 : 6;
  for (let tick = 0; tick < tickCount; tick += 1) {
    const fraction = tick / (tickCount - 1);
    const target = viewport[0] + span * fraction;
    const sample = nearestSampleIndex(data.indices, target);
    if (sample < 0) continue;
    const x = geometry.left + ((data.indices[sample] - viewport[0]) / span) * plotWidth;
    if (x < geometry.left - 1 || x > geometry.right + 1) continue;
    const label = data.timestamp_labels[sample] ?? String(data.indices[sample]);
    const maxWidth = Math.max(42, plotWidth / tickCount - 10);
    context.fillText(
      textEllipsis(context, String(label), maxWidth),
      x,
      geometry.bottom + 10,
    );
  }
}

export function renderChart(canvas, options) {
  const {
    data,
    viewport,
    layout,
    transform,
    selection,
    hoverIndex,
    width,
    height,
    dpr,
  } = options;
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  if (!data || !data.features.length) return;

  const geometry = chartGeometry(width, height, data.features.length, layout);
  const plotWidth = geometry.right - geometry.left;
  const span = Math.max(1, viewport[1] - viewport[0]);
  const mapX = (index) => geometry.left + ((index - viewport[0]) / span) * plotWidth;

  context.save();
  context.beginPath();
  context.rect(
    geometry.left,
    geometry.top,
    plotWidth,
    geometry.bottom - geometry.top,
  );
  context.clip();
  for (const run of data.label_runs) {
    const start = Math.max(geometry.left, mapX(run.start_index - 0.5));
    const stop = Math.min(geometry.right, mapX(run.stop_index + 0.5));
    if (stop <= start) continue;
    context.fillStyle = BAND;
    context.fillRect(start, geometry.top, stop - start, geometry.bottom - geometry.top);
    context.fillStyle = BAND_EDGE;
    context.fillRect(start, geometry.top, 1, geometry.bottom - geometry.top);
    context.fillRect(stop - 1, geometry.top, 1, geometry.bottom - geometry.top);
  }
  context.restore();

  const overlayDomain = layout === "overlay" ? combinedDomain(data.features) : null;
  for (let featureIndex = 0; featureIndex < data.features.length; featureIndex += 1) {
    const feature = data.features[featureIndex];
    const lane = layout === "overlay" ? 0 : featureIndex;
    const laneTop = geometry.top + lane * geometry.laneHeight;
    const laneBottom = laneTop + geometry.laneHeight;
    if (layout !== "overlay" || featureIndex === 0) {
      drawGrid(context, geometry, laneTop, laneBottom);
    }
    const domain = overlayDomain ?? valueDomain(feature.values);
    if (!domain) continue;
    const innerTop = laneTop + 12;
    const innerBottom = laneBottom - 12;
    const mapY = (value) => valueToY(value, domain, innerTop, innerBottom);
    const centerX = (geometry.left + geometry.right) / 2;
    const centerY = (laneTop + laneBottom) / 2;
    const points = buildPolyline2D(
      data.indices,
      feature.values,
      mapX,
      mapY,
      transform,
      centerX,
      centerY,
    );
    context.save();
    context.beginPath();
    context.rect(geometry.left, laneTop, plotWidth, geometry.laneHeight);
    context.clip();
    drawPolyline(
      context,
      points,
      TRACK_COLORS[feature.index % TRACK_COLORS.length],
    );
    context.restore();

    if (layout === "stacked") {
      context.save();
      context.beginPath();
      context.rect(0, laneTop, geometry.left - 5, geometry.laneHeight);
      context.clip();
      const trackColor = TRACK_COLORS[feature.index % TRACK_COLORS.length];
      context.fillStyle = trackColor;
      context.fillRect(8, laneTop + 8, 3, 30);
      context.fillStyle = INK;
      context.font = "600 11px ui-sans-serif, system-ui, sans-serif";
      context.textAlign = "left";
      context.textBaseline = "top";
      context.fillText(feature.name, 17, laneTop + 8);
      context.fillStyle = MUTED;
      context.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
      context.fillText(
        `${formatNumber(domain.min)} – ${formatNumber(domain.max)}`,
        17,
        laneTop + 27,
      );
      context.restore();
    }
  }

  if (layout === "overlay") {
    context.fillStyle = INK;
    context.font = "600 11px ui-sans-serif, system-ui, sans-serif";
    context.textAlign = "left";
    context.textBaseline = "top";
    context.fillText(`${data.features.length} overlaid tracks`, 8, geometry.top + 8);
  }

  if (hoverIndex != null && data.indices[hoverIndex] !== undefined) {
    const x = mapX(data.indices[hoverIndex]);
    if (x >= geometry.left && x <= geometry.right) {
      context.strokeStyle = "#334155";
      context.lineWidth = 1;
      context.setLineDash([4, 4]);
      context.beginPath();
      context.moveTo(x + 0.5, geometry.top);
      context.lineTo(x + 0.5, geometry.bottom);
      context.stroke();
      context.setLineDash([]);
    }
  }

  if (selection) {
    const left = Math.min(selection.startX, selection.endX);
    const top = Math.min(selection.startY, selection.endY);
    const selectionWidth = Math.abs(selection.endX - selection.startX);
    const selectionHeight = Math.abs(selection.endY - selection.startY);
    context.fillStyle = "rgba(23, 105, 170, 0.10)";
    context.strokeStyle = "#1769aa";
    context.lineWidth = 1;
    context.setLineDash([5, 4]);
    context.fillRect(left, top, selectionWidth, selectionHeight);
    context.strokeRect(left + 0.5, top + 0.5, selectionWidth, selectionHeight);
    context.setLineDash([]);
  }

  drawTimeAxis(context, data, viewport, geometry);
}

export function renderInteractionOverlay(canvas, options) {
  const { data, viewport, layout, selection, hoverIndex, width, height, dpr } = options;
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!data) return;
  const geometry = chartGeometry(width, height, data.features.length, layout);
  const span = Math.max(1, viewport[1] - viewport[0]);
  const mapX = (index) =>
    geometry.left +
    ((index - viewport[0]) / span) * (geometry.right - geometry.left);

  if (hoverIndex != null && data.indices[hoverIndex] !== undefined) {
    const x = mapX(data.indices[hoverIndex]);
    if (x >= geometry.left && x <= geometry.right) {
      context.strokeStyle = "#334155";
      context.lineWidth = 1;
      context.setLineDash([4, 4]);
      context.beginPath();
      context.moveTo(x + 0.5, geometry.top);
      context.lineTo(x + 0.5, geometry.bottom);
      context.stroke();
      context.setLineDash([]);
    }
  }

  if (selection) {
    const left = Math.min(selection.startX, selection.endX);
    const top = Math.min(selection.startY, selection.endY);
    const selectionWidth = Math.abs(selection.endX - selection.startX);
    const selectionHeight = Math.abs(selection.endY - selection.startY);
    context.fillStyle = "rgba(23, 105, 170, 0.10)";
    context.strokeStyle = "#1769aa";
    context.lineWidth = 1;
    context.setLineDash([5, 4]);
    context.fillRect(left, top, selectionWidth, selectionHeight);
    context.strokeRect(left + 0.5, top + 0.5, selectionWidth, selectionHeight);
    context.setLineDash([]);
  }
}

// AgentAD Visualize — dependency-free browser application.

import {
  MIN_SPAN,
  clampWindow,
  formatNumber,
  nearestSampleIndex,
  parseHashState,
  serializeHashState,
  valueDomain,
  valueToY,
} from "/core.js";

const COLORS = {
  line: "#38bdf8",
  band: "rgba(248, 113, 113, 0.13)",
  bandEdge: "rgba(248, 113, 113, 0.38)",
  grid: "rgba(148, 163, 184, 0.09)",
  separator: "rgba(51, 65, 85, 0.7)",
  axisText: "#94a3b8",
  nameBg: "rgba(15, 23, 42, 0.78)",
  nameText: "#cbd5e1",
  crosshair: "#e2e8f0",
};

const X_AXIS_H = 26;
const ROW_MIN_H = 28;
const ROW_MAX_H = 170;
const MAX_CANVAS_PIXELS = 24_000_000;

class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.status = status;
  }
}

async function api(path, params = {}, signal) {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== null && value !== "") {
      qs.set(key, String(value));
    }
  }
  const suffix = qs.toString() ? `?${qs}` : "";
  const response = await fetch(path + suffix, { signal });
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    throw new ApiError(body?.error || `HTTP ${response.status}`, response.status);
  }
  return body;
}

const state = {
  overview: null,
  seriesIndex: null,
  points: 0,
  pendingStart: 0,
  pendingStop: null,
  features: new Set(),
  normalization: "none",
  scope: "feature",
  labelIndex: null,
  maxPoints: 4000,
  hardMaxPoints: 50000,
  maxFeatures: 128,
  items: [],
  itemsTotal: 0,
  itemsOffset: 0,
  query: "",
  data: null,
  colStats: [],
  hover: null,
  pendingWindowApply: null,
  loadedWin: null,
  treeSelected: null,
};

const el = {};
let ctx = null;
let dpr = 1;
let rowH = 100;
let renderQueued = false;
let lastHash = "";
let noticeTimer = 0;
const fetchState = { timer: 0, controller: null, seq: 0, requested: null };
const itemFetchState = { controller: null, seq: 0 };
const openFetchState = { seq: 0, busy: false };

// ---------- rendering ----------

function requestRender() {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    draw();
  });
}

function resizeCanvas() {
  if (!ctx) return;
  const wrap = el.chartWrap;
  const cssW = Math.max(1, wrap.clientWidth);
  const availH = Math.max(40, wrap.clientHeight - X_AXIS_H);
  const n = state.data ? state.data.features.length : 1;
  rowH = Math.max(ROW_MIN_H, Math.min(ROW_MAX_H, availH / n));
  const rowsH = state.data ? n * rowH : availH;
  const cssH = Math.max(wrap.clientHeight, rowsH + X_AXIS_H);
  const idealDpr = Math.min(window.devicePixelRatio || 1, 2);
  dpr = Math.min(
    idealDpr,
    Math.sqrt(MAX_CANVAS_PIXELS / Math.max(1, cssW * cssH)),
  );
  el.chart.width = Math.round(cssW * dpr);
  el.chart.height = Math.round(cssH * dpr);
  el.chart.style.width = `${cssW}px`;
  el.chart.style.height = `${cssH}px`;
}

function xAt(index, span, cssW) {
  return ((index - state.pendingStart) / span) * cssW;
}

function fitCanvasText(value, maxWidth) {
  const text = String(value);
  if (maxWidth <= 0 || ctx.measureText(text).width <= maxWidth) return text;
  let lower = 0;
  let upper = text.length;
  while (lower < upper) {
    const middle = Math.ceil((lower + upper) / 2);
    if (ctx.measureText(`${text.slice(0, middle)}…`).width <= maxWidth) {
      lower = middle;
    } else {
      upper = middle - 1;
    }
  }
  return `${text.slice(0, lower)}…`;
}

function draw() {
  if (!ctx) return;
  const cssW = el.chart.width / dpr;
  const cssH = el.chart.height / dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  const data = state.data;
  if (!data) return;

  const feats = data.features;
  const n = feats.length;
  const indices = data.indices;
  const span = Math.max(1, (state.pendingStop ?? state.points) - state.pendingStart);
  const rowsH = n * rowH;
  const mapX = (index) => xAt(index, span, cssW);

  for (const run of data.label_runs) {
    const x0 = Math.max(0, mapX(run.start_index - 0.5));
    const x1 = Math.min(cssW, mapX(run.stop_index + 0.5));
    if (x1 <= 0 || x0 >= cssW) continue;
    ctx.fillStyle = COLORS.band;
    ctx.fillRect(x0, 0, x1 - x0, rowsH);
    ctx.fillStyle = COLORS.bandEdge;
    ctx.fillRect(x0, 0, 1, rowsH);
    ctx.fillRect(x1 - 1, 0, 1, rowsH);
  }

  for (let r = 0; r < n; r++) {
    const y0 = r * rowH;
    const y1 = y0 + rowH;
    const domain = state.colStats[r];
    const plotTop = y0 + 6;
    const plotBottom = y1 - 6;
    const mapY = (value) => valueToY(value, domain, plotTop, plotBottom);

    ctx.strokeStyle = COLORS.grid;
    ctx.lineWidth = 1;
    for (const frac of [0.25, 0.5, 0.75]) {
      const y = Math.round(plotTop + frac * (plotBottom - plotTop)) + 0.5;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(cssW, y);
      ctx.stroke();
    }

    const values = feats[r].values;
    ctx.strokeStyle = COLORS.line;
    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    let pen = false;
    for (let k = 0; k < indices.length; k++) {
      const x = mapX(indices[k]);
      // The loaded window is wider than the viewport (overscan); skip and
      // stop early outside a small margin so panning stays cheap.
      if (x < -2) continue;
      if (x > cssW + 2) break;
      const value = values[k];
      if (value === null || value === undefined) {
        pen = false;
        continue;
      }
      const y = mapY(value);
      if (y === null || !Number.isFinite(y)) {
        pen = false;
        continue;
      }
      if (pen) ctx.lineTo(x, y);
      else ctx.moveTo(x, y);
      pen = true;
    }
    ctx.stroke();

    ctx.font = "11px " + getComputedStyle(document.body).getPropertyValue("--font-mono");
    const rangeText = domain
      ? `${formatNumber(domain.min)} ~ ${formatNumber(domain.max)}`
      : "";
    const rangeW = rangeText ? ctx.measureText(rangeText).width : 0;
    const name = fitCanvasText(feats[r].name, Math.max(24, cssW - rangeW - 38));
    const nameW = ctx.measureText(name).width;
    ctx.fillStyle = COLORS.nameBg;
    ctx.fillRect(4, y0 + 4, nameW + 12, 17);
    ctx.fillStyle = COLORS.nameText;
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    ctx.fillText(name, 10, y0 + 13);

    if (domain) {
      ctx.fillStyle = COLORS.nameBg;
      ctx.fillRect(cssW - rangeW - 14, y0 + 4, rangeW + 10, 17);
      ctx.fillStyle = COLORS.axisText;
      ctx.textAlign = "right";
      ctx.fillText(rangeText, cssW - 9, y0 + 13);
    }

    ctx.strokeStyle = COLORS.separator;
    ctx.beginPath();
    ctx.moveTo(0, Math.round(y1) - 0.5);
    ctx.lineTo(cssW, Math.round(y1) - 0.5);
    ctx.stroke();
  }

  ctx.strokeStyle = COLORS.separator;
  ctx.beginPath();
  ctx.moveTo(0, Math.round(rowsH) + 0.5);
  ctx.lineTo(cssW, Math.round(rowsH) + 0.5);
  ctx.stroke();

  ctx.fillStyle = COLORS.axisText;
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  if (indices.length) {
    const tickCount = Math.min(6, indices.length);
    for (let t = 0; t < tickCount; t++) {
      const frac = tickCount === 1 ? 0.5 : t / (tickCount - 1);
      const k = Math.round(frac * (indices.length - 1));
      const x = mapX(indices[k]);
      if (x < 30 || x > cssW - 30) continue;
      const label = fitCanvasText(
        data.timestamp_labels[k] ?? String(indices[k]),
        Math.max(40, cssW / tickCount - 10),
      );
      ctx.fillText(label, x, rowsH + 7);
    }
  }

  if (state.hover != null && indices[state.hover] !== undefined) {
    const hx = mapX(indices[state.hover]);
    ctx.strokeStyle = COLORS.crosshair;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(hx + 0.5, 0);
    ctx.lineTo(hx + 0.5, rowsH);
    ctx.stroke();
    ctx.setLineDash([]);
    for (let r = 0; r < n; r++) {
      const value = feats[r].values[state.hover];
      if (value === null || value === undefined) continue;
      const domain = state.colStats[r];
      const plotTop = r * rowH + 6;
      const plotBottom = (r + 1) * rowH - 6;
      const y = valueToY(value, domain, plotTop, plotBottom);
      if (y === null || !Number.isFinite(y)) continue;
      ctx.beginPath();
      ctx.arc(hx, y, 3, 0, Math.PI * 2);
      ctx.fillStyle = "#f8fafc";
      ctx.fill();
      ctx.strokeStyle = COLORS.line;
      ctx.stroke();
    }
  }
}

// ---------- data fetching ----------

// Fetch a window wider than the viewport so panning renders instantly;
// refetch only when the viewport drifts near the prefetched edge or the
// visible span drops well below the loaded span (zoom-in density).
const OVERSCAN = 0.6;
const REFETCH_MARGIN = 0.25;
const REDENSITY = 0.55;

function selectedFeatureCsv() {
  return [...state.features].sort((a, b) => a - b).join(",");
}

function fetchWindowFor(start, stop) {
  const total = state.points;
  if (stop == null) {
    return { start: 0, stop: null, max_points: state.maxPoints };
  }
  const span = Math.max(1, stop - start);
  const pad = Math.max(1, Math.round(span * OVERSCAN));
  const reqStart = Math.max(0, start - pad);
  const reqStop = Math.min(total, stop + pad);
  const ratio = (reqStop - reqStart) / span;
  const maxPoints = Math.max(
    state.maxPoints,
    Math.min(state.hardMaxPoints, Math.round(state.maxPoints * ratio)),
  );
  return { start: reqStart, stop: reqStop, max_points: maxPoints };
}

function coversWindow(win, start, stop, pad) {
  if (!win) return false;
  const winStop = win.stop == null ? Infinity : win.stop;
  return start - pad >= win.start && stop + pad <= winStop;
}

function windowNeedsFetch() {
  if (!state.loadedWin) return true;
  const start = state.pendingStart;
  const stop = state.pendingStop ?? state.points;
  const span = stop - start;
  if (span <= 0) return true;
  const pad = span * REFETCH_MARGIN;
  for (const win of [state.loadedWin, fetchState.requested]) {
    if (!coversWindow(win, start, stop, pad)) continue;
    // Density is judged against the visible span at fetch time, so panning
    // never refetches while zooming in does once the span shrinks enough.
    const basis = win.visSpan == null ? span : win.visSpan;
    if (span >= basis * REDENSITY) return false;
  }
  return true;
}

function scheduleFetch(delay = 0) {
  clearTimeout(fetchState.timer);
  fetchState.timer = setTimeout(executeFetch, delay);
}

async function executeFetch() {
  if (state.seriesIndex == null) return;
  fetchState.controller?.abort();
  const controller = new AbortController();
  fetchState.controller = controller;
  const seq = ++fetchState.seq;
  const win = fetchWindowFor(state.pendingStart, state.pendingStop);
  fetchState.requested = {
    start: win.start,
    stop: win.stop,
    visSpan: (state.pendingStop ?? state.points) - state.pendingStart,
  };
  setLoading(true);
  try {
    const payload = await api(
      "/api/data",
      {
        series: state.seriesIndex,
        features: selectedFeatureCsv(),
        normalization: state.normalization,
        scope: state.scope,
        start: win.start,
        stop: win.stop,
        max_points: win.max_points,
        label: state.labelIndex,
      },
      controller.signal,
    );
    if (seq !== fetchState.seq) return;
    state.loadedWin = fetchState.requested;
    onData(payload);
  } catch (error) {
    if (seq !== fetchState.seq || error.name === "AbortError") return;
    showChartMsg("error", error.message);
  } finally {
    if (seq === fetchState.seq) setLoading(false);
  }
}

function onData(payload) {
  state.points = payload.series.points;
  state.data = payload;
  state.hover = null;
  hideTooltip();

  if (state.pendingStop == null) {
    state.pendingStart = payload.window.start;
    state.pendingStop = payload.window.stop;
  }
  if (state.pendingWindowApply) {
    const [s, e] = clampWindow(
      state.pendingWindowApply[0],
      state.pendingWindowApply[1],
      state.points,
    );
    state.pendingWindowApply = null;
    if (s !== state.pendingStart || e !== state.pendingStop) {
      state.pendingStart = s;
      state.pendingStop = e;
      syncHash();
      if (windowNeedsFetch()) scheduleFetch(0);
    }
  }

  state.colStats = payload.features.map((feature) => valueDomain(feature.values));

  if (payload.series.points === 0) {
    showChartMsg("empty", "该序列不包含数据点");
  } else if (payload.features.length === 0) {
    showChartMsg("empty", "该集合不包含可绘制的特征");
  } else {
    hideChartMsg();
  }
  resizeCanvas();
  updateWindowInfo();
  updateChartSummary();
  syncHash();
  requestRender();
}

function setWindow(start, stop) {
  const [s, e] = clampWindow(start, stop, state.points);
  if (s === state.pendingStart && e === state.pendingStop) return false;
  state.pendingStart = s;
  state.pendingStop = e;
  syncHash();
  updateWindowInfo();
  requestRender();
  if (windowNeedsFetch()) scheduleFetch(0);
  return true;
}

function zoomAt(anchorIndex, factor) {
  if (state.pendingStop == null) return;
  const span = state.pendingStop - state.pendingStart;
  const newSpan = Math.min(
    Math.max(Math.round(span * factor), Math.min(MIN_SPAN, state.points)),
    state.points,
  );
  const anchor = Math.min(
    Math.max(anchorIndex, state.pendingStart),
    state.pendingStop,
  );
  const frac = span ? (anchor - state.pendingStart) / span : 0;
  const newStart = Math.round(anchor - frac * newSpan);
  setWindow(newStart, newStart + newSpan);
}

function panBy(delta) {
  if (state.pendingStop == null) return;
  setWindow(state.pendingStart + delta, state.pendingStop + delta);
}

function setSeries(index, options = {}) {
  clearTimeout(fetchState.timer);
  fetchState.controller?.abort();
  fetchState.requested = null;
  state.loadedWin = null;
  state.seriesIndex = index;
  state.data = null;
  state.points = 0;
  state.hover = null;
  state.colStats = [];
  state.pendingStart = 0;
  state.pendingStop = null;
  state.pendingWindowApply = options.window ?? null;
  hideTooltip();
  highlightSeries();
  clearHoverInfo();
  updateWindowInfo();
  updateChartSummary();
  syncHash();
  showChartMsg("loading");
  requestRender();
  scheduleFetch(0);
}

// ---------- UI building ----------

function el9(id) {
  return document.getElementById(id);
}

function buildStaticUI() {
  const ov = state.overview;
  document.title = `${ov.title} · AgentAD Visualize`;
  el.appTitle.textContent = ov.title;
  el.stats.textContent = [
    `${ov.series_count.toLocaleString("zh-CN")} 系列`,
    `${ov.point_count.toLocaleString("zh-CN")} 数据点`,
    `${ov.feature_count} 特征`,
    `${ov.label_count} 标注列`,
  ].join(" · ");

  el.normSelect.textContent = "";
  for (const item of ov.normalizations) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.name;
    option.title = item.description ?? "";
    el.normSelect.appendChild(option);
  }

  el.labelSelect.textContent = "";
  const none = document.createElement("option");
  none.value = "";
  none.textContent = "无";
  el.labelSelect.appendChild(none);
  for (const label of ov.binary_labels) {
    const option = document.createElement("option");
    option.value = String(label.index);
    option.textContent = label.name;
    el.labelSelect.appendChild(option);
  }

  buildChips();
  buildManifest();
  el.tabSeries.disabled = false;
  setControlsEnabled(true);
}

function setControlsEnabled(enabled) {
  el.normSelect.disabled = !enabled;
  el.scopeSelect.disabled = !enabled || state.normalization === "none";
  el.labelSelect.disabled = !enabled || !state.overview?.binary_labels.length;
  el.zoomIn.disabled = !enabled;
  el.zoomOut.disabled = !enabled;
  el.zoomReset.disabled = !enabled;
}

function buildChips() {
  el.featureBar.textContent = "";
  for (const feature of state.overview.features) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip";
    chip.dataset.index = String(feature.index);
    chip.textContent = feature.name;
    const meta = Object.entries(feature.metadata ?? {})
      .map(([key, value]) => `${key}: ${value}`)
      .join("；");
    chip.title = meta ? `${feature.name}\n${meta}` : feature.name;
    chip.setAttribute("aria-pressed", state.features.has(feature.index) ? "true" : "false");
    chip.addEventListener("click", () => toggleFeature(feature.index));
    el.featureBar.appendChild(chip);
  }
}

function toggleFeature(index) {
  if (state.features.has(index)) {
    if (state.features.size <= 1) {
      notify("至少保留一个特征");
      return;
    }
    state.features.delete(index);
  } else {
    if (state.features.size >= state.maxFeatures) {
      notify(`最多选择 ${state.maxFeatures} 个特征`);
      return;
    }
    state.features.add(index);
  }
  for (const chip of el.featureBar.children) {
    chip.setAttribute(
      "aria-pressed",
      state.features.has(Number(chip.dataset.index)) ? "true" : "false",
    );
  }
  syncHash();
  scheduleFetch(0);
}

function buildManifest() {
  const body = el.manifestBody;
  body.textContent = "";
  const entries = Object.entries(state.overview.manifest ?? {});
  el.manifestBox.hidden = entries.length === 0;
  for (const [key, value] of entries) {
    const row = document.createElement("div");
    const keyNode = document.createElement("b");
    keyNode.textContent = `${key}: `;
    row.appendChild(keyNode);
    const text = typeof value === "string" ? value : JSON.stringify(value);
    row.appendChild(document.createTextNode(text.length > 140 ? `${text.slice(0, 140)}…` : text));
    row.title = text;
    body.appendChild(row);
  }
}

function buildSeriesItem(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "series-item";
  button.dataset.index = String(item.index);
  button.setAttribute("role", "option");
  const id = document.createElement("span");
  id.className = "series-id";
  id.textContent = item.id;
  id.title = item.id;
  const points = document.createElement("span");
  points.className = "series-points";
  points.textContent = item.points.toLocaleString("zh-CN");
  button.append(id, points);
  button.addEventListener("click", () => {
    setSeries(item.index);
    closeDrawer();
  });
  return button;
}

async function fetchItems(append) {
  itemFetchState.controller?.abort();
  const controller = new AbortController();
  itemFetchState.controller = controller;
  const seq = ++itemFetchState.seq;
  el.seriesList.setAttribute("aria-busy", "true");
  el.moreBtn.disabled = true;
  try {
    const payload = await api(
      "/api/items",
      {
        query: state.query,
        offset: append ? state.itemsOffset : 0,
        limit: 50,
      },
      controller.signal,
    );
    if (seq !== itemFetchState.seq) return;
    state.items = append ? [...state.items, ...payload.items] : payload.items;
    state.itemsTotal = payload.total;
    state.itemsOffset = payload.offset + payload.items.length;
    renderSeriesList();
  } catch (error) {
    if (seq !== itemFetchState.seq || error.name === "AbortError") return;
    notify(`系列列表加载失败：${error.message}`);
  } finally {
    if (seq === itemFetchState.seq) {
      el.seriesList.setAttribute("aria-busy", "false");
      el.moreBtn.disabled = false;
    }
  }
}

function renderSeriesList() {
  const list = el.seriesList;
  if (!state.items.length) {
    list.textContent = "";
    const empty = document.createElement("div");
    empty.className = "series-empty";
    empty.textContent = state.query ? "没有匹配的系列" : "该集合不包含序列";
    list.appendChild(empty);
  } else {
    const wanted = new Set(state.items.map((item) => item.index));
    for (const child of [...list.children]) {
      if (child.dataset?.index && !wanted.has(Number(child.dataset.index))) {
        child.remove();
      }
    }
    const existing = new Set(
      [...list.children].map((child) => Number(child.dataset.index)),
    );
    for (const item of state.items) {
      if (!existing.has(item.index)) {
        list.appendChild(buildSeriesItem(item));
      }
    }
  }
  el.moreBtn.hidden = state.itemsOffset >= state.itemsTotal;
  el.listCount.textContent = state.itemsTotal
    ? `显示 ${state.items.length} / ${state.itemsTotal.toLocaleString("zh-CN")}`
    : "";
  highlightSeries();
}

function highlightSeries() {
  for (const child of el.seriesList.children) {
    if (child.dataset?.index) {
      child.setAttribute(
        "aria-selected",
        Number(child.dataset.index) === state.seriesIndex ? "true" : "false",
      );
    }
  }
}

// ---------- file tree ----------

const TREE_ICONS = {
  dir: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2Z"/></svg>',
  package: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m12 3 8 4.5v9L12 21l-8-4.5v-9Z"/><path d="m4 7.5 8 4.5 8-4.5"/><path d="M12 12v9"/></svg>',
  file: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 3h8l4 4v14H6Z"/><path d="M14 3v4h4"/></svg>',
};

function formatSize(bytes) {
  if (typeof bytes !== "number" || !Number.isFinite(bytes) || bytes < 0) return "";
  let value = bytes;
  for (const unit of ["B", "KB", "MB", "GB", "TB", "PB"]) {
    if (value < 1024 || unit === "PB") {
      const text = unit === "B"
        ? String(value)
        : value >= 100
          ? String(Math.round(value))
          : value.toFixed(1);
      return `${text} ${unit}`;
    }
    value /= 1024;
  }
  return "";
}

function switchTab(which) {
  const files = which === "files";
  if (!files && el.tabSeries.disabled) return;
  el.tabFiles.setAttribute("aria-selected", String(files));
  el.tabSeries.setAttribute("aria-selected", String(!files));
  el.tabFiles.tabIndex = files ? 0 : -1;
  el.tabSeries.tabIndex = files ? -1 : 0;
  el.panelFiles.hidden = !files;
  el.panelSeries.hidden = files;
  el.sidebarFoot.hidden = files;
}

function setTreeRootLabel(root) {
  const span = document.createElement("span");
  span.textContent = root;
  el.treeRoot.replaceChildren(span);
  el.treeRoot.title = root;
}

function buildTreeItem(entry, depth) {
  const item = document.createElement("div");
  item.className = "tree-item";
  item.setAttribute("role", "none");
  const row = document.createElement("button");
  row.type = "button";
  row.className = "tree-row";
  row.dataset.path = entry.path;
  row.dataset.depth = String(depth);
  row.style.setProperty("--depth", String(depth));
  const toggle = document.createElement("span");
  toggle.className = "tree-toggle";
  toggle.textContent = entry.dir ? "▶" : "";
  row.setAttribute("role", "treeitem");
  row.setAttribute("aria-level", String(depth + 1));
  const icon = document.createElement("span");
  icon.className = "tree-icon";
  icon.innerHTML = TREE_ICONS[entry.loadable ? "package" : entry.dir ? "dir" : "file"];
  const name = document.createElement("span");
  name.className = "tree-name";
  name.textContent = entry.name;
  row.append(toggle, icon, name);
  if (entry.loadable) {
    row.dataset.loadable = "1";
    row.title = `${entry.path} — 点击加载数据包`;
    row.setAttribute("aria-selected", "false");
    row.addEventListener("click", () => openCollection(entry.path));
  } else if (entry.dir) {
    row.setAttribute("aria-expanded", "false");
    row.title = entry.path;
    row.addEventListener("click", () => {
      if (row.getAttribute("aria-expanded") === "true") collapseTreeDir(row);
      else expandTreeDir(row);
    });
  } else {
    row.disabled = true;
    row.title = entry.path;
  }
  if (!entry.dir) {
    const size = document.createElement("span");
    size.className = "tree-size";
    size.textContent = formatSize(entry.size);
    row.appendChild(size);
  }
  item.appendChild(row);
  if (entry.dir && !entry.loadable) {
    const children = document.createElement("div");
    children.className = "tree-item-children";
    children.setAttribute("role", "group");
    children.hidden = true;
    children.dataset.loaded = "false";
    item.appendChild(children);
  }
  return { item, row };
}

async function fetchTreeChildren(relPath, container, depth) {
  const payload = await api("/api/tree", relPath ? { path: relPath } : {});
  container.textContent = "";
  if (!payload.entries.length) {
    const empty = document.createElement("div");
    empty.className = "tree-empty";
    empty.textContent = "（空目录）";
    container.appendChild(empty);
    return;
  }
  for (const entry of payload.entries) {
    container.appendChild(buildTreeItem(entry, depth).item);
  }
  if (payload.truncated) {
    const truncated = document.createElement("div");
    truncated.className = "tree-empty";
    truncated.textContent = "目录条目过多，仅显示前 5,000 项";
    container.appendChild(truncated);
  }
}

function treeChildrenOf(row) {
  return row.parentElement?.querySelector(":scope > .tree-item-children") ?? null;
}

async function expandTreeDir(row) {
  const children = treeChildrenOf(row);
  if (!children) return;
  const toggle = row.querySelector(":scope > .tree-toggle");
  if (children.dataset.loaded !== "true") {
    if (toggle) {
      toggle.classList.add("spinner");
      toggle.textContent = "";
    }
    try {
      await fetchTreeChildren(row.dataset.path, children, Number(row.dataset.depth) + 1);
      children.dataset.loaded = "true";
    } catch (error) {
      notify(`目录读取失败：${error.message}`);
      return;
    } finally {
      if (toggle) {
        toggle.classList.remove("spinner");
        toggle.textContent = "▶";
      }
    }
  }
  children.hidden = false;
  row.setAttribute("aria-expanded", "true");
}

function collapseTreeDir(row) {
  const children = treeChildrenOf(row);
  if (children) children.hidden = true;
  row.setAttribute("aria-expanded", "false");
}

function markTreeSelection() {
  for (const row of el.tree.querySelectorAll(".tree-row[data-loadable]")) {
    row.setAttribute(
      "aria-selected",
      row.dataset.path === state.treeSelected ? "true" : "false",
    );
  }
}

async function revealTreePath(relPath) {
  const parts = relPath.split("/").filter(Boolean);
  let container = el.tree;
  for (let i = 0; i < parts.length; i++) {
    const partial = parts.slice(0, i + 1).join("/");
    const row = container.querySelector(
      `.tree-row[data-path="${CSS.escape(partial)}"]`,
    );
    if (!row) return;
    if (i === parts.length - 1) {
      if (row.dataset.loadable) {
        state.treeSelected = relPath;
        markTreeSelection();
      }
      return;
    }
    if (row.getAttribute("aria-expanded") !== "true") {
      await expandTreeDir(row);
      if (row.getAttribute("aria-expanded") !== "true") return;
    }
    container = treeChildrenOf(row) ?? el.tree;
  }
}

async function openCollection(relPath) {
  if (state.treeSelected === relPath) {
    switchTab("series");
    el.tabSeries.focus();
    return;
  }
  if (openFetchState.busy) {
    notify("正在打开数据包，请稍候");
    return;
  }
  const row = el.tree.querySelector(
    `.tree-row[data-path="${CSS.escape(relPath)}"]`,
  );
  const seq = ++openFetchState.seq;
  openFetchState.busy = true;
  row?.setAttribute("aria-busy", "true");
  try {
    const payload = await api("/api/open", { path: relPath });
    if (seq !== openFetchState.seq) return;
    state.treeSelected = relPath;
    adoptCollection(payload);
  } catch (error) {
    if (seq !== openFetchState.seq) return;
    notify(`打开数据包失败：${error.message}`);
  } finally {
    row?.removeAttribute("aria-busy");
    if (seq === openFetchState.seq) openFetchState.busy = false;
  }
}

function adoptCollection(payload) {
  clearTimeout(fetchState.timer);
  fetchState.controller?.abort();
  itemFetchState.controller?.abort();
  const previous = state.overview;
  const previousFeatures = previous
    ? previous.features.filter((feature) => state.features.has(feature.index))
    : [];
  const previousLabel =
    previous && state.labelIndex != null
      ? previous.binary_labels.find((label) => label.index === state.labelIndex)?.name ?? null
      : null;

  state.overview = payload;
  state.maxPoints = payload.limits?.default_max_points ?? 4000;
  state.hardMaxPoints = payload.limits?.hard_max_points ?? 50000;
  state.maxFeatures = payload.limits?.max_selected_features ?? 128;
  // Carry toolbar selections over to the new collection, re-mapped by name.
  const previousNames = new Set(previousFeatures.map((feature) => feature.name));
  const carried = payload.features
    .filter((feature) => previousNames.has(feature.name))
    .slice(0, state.maxFeatures)
    .map((feature) => feature.index);
  state.features = new Set(
    carried.length
      ? carried
      : payload.features
          .slice(0, Math.min(8, state.maxFeatures))
          .map((feature) => feature.index),
  );
  state.normalization = payload.normalizations.some(
    (item) => item.id === state.normalization,
  )
    ? state.normalization
    : "none";
  state.labelIndex =
    previousLabel == null
      ? null
      : payload.binary_labels.find((label) => label.name === previousLabel)?.index ?? null;

  state.query = "";
  el.search.value = "";
  state.items = [];
  state.itemsTotal = 0;
  state.itemsOffset = 0;
  history.replaceState(null, "", location.pathname);
  lastHash = "";
  buildStaticUI();
  el.normSelect.value = state.normalization;
  el.scopeSelect.value = state.scope;
  el.labelSelect.value = state.labelIndex == null ? "" : String(state.labelIndex);
  markTreeSelection();
  switchTab("series");
  el.tabSeries.focus();
  if (payload.series_count > 0) {
    setSeries(0);
    fetchItems(false);
  } else {
    state.seriesIndex = null;
    state.points = 0;
    state.data = null;
    state.pendingStart = 0;
    state.pendingStop = null;
    showChartMsg("empty", "该集合不包含任何序列");
    updateWindowInfo();
    updateChartSummary();
    requestRender();
  }
}

function enterBrowserMode() {
  state.overview = null;
  el.appTitle.textContent = "AgentAD Visualize";
  el.stats.textContent = "";
  el.tabSeries.disabled = true;
  setControlsEnabled(false);
  showChartMsg("empty", "从左侧「文件」面板选择一个数据包");
  switchTab("files");
}

// ---------- sidebar resizing ----------

const SIDEBAR_DEFAULT_W = 280;
const SIDEBAR_MIN_W = 200;
const SIDEBAR_MAX_W = 600;
const SIDEBAR_WIDTH_KEY = "agentad-visualize:sidebar-width";

function clampSidebarWidth(width) {
  const max = Math.max(SIDEBAR_MIN_W, Math.min(SIDEBAR_MAX_W, window.innerWidth - 320));
  return Math.round(Math.min(Math.max(width, SIDEBAR_MIN_W), max));
}

function applySidebarWidth(width) {
  const clamped = clampSidebarWidth(width);
  el.sidebar.style.width = `${clamped}px`;
  el.resizer.setAttribute("aria-valuenow", String(clamped));
  return clamped;
}

function restoreSidebarWidth() {
  let stored = NaN;
  try {
    stored = Number(window.localStorage.getItem(SIDEBAR_WIDTH_KEY));
  } catch {
    stored = NaN;
  }
  if (Number.isFinite(stored) && stored > 0) applySidebarWidth(stored);
}

function saveSidebarWidth() {
  try {
    window.localStorage.setItem(
      SIDEBAR_WIDTH_KEY,
      String(Math.round(el.sidebar.getBoundingClientRect().width)),
    );
  } catch {
    /* storage unavailable — width just won't persist */
  }
}

function bindResizer() {
  const resizer = el.resizer;
  let dragging = false;
  resizer.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    dragging = true;
    resizer.setPointerCapture(event.pointerId);
    document.body.classList.add("is-resizing");
  });
  resizer.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    const rect = el.sidebar.parentElement.getBoundingClientRect();
    applySidebarWidth(event.clientX - rect.left);
  });
  const stopDrag = (event) => {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove("is-resizing");
    if (resizer.hasPointerCapture(event.pointerId)) {
      resizer.releasePointerCapture(event.pointerId);
    }
    saveSidebarWidth();
  };
  resizer.addEventListener("pointerup", stopDrag);
  resizer.addEventListener("pointercancel", stopDrag);
  resizer.addEventListener("dblclick", () => {
    applySidebarWidth(SIDEBAR_DEFAULT_W);
    saveSidebarWidth();
  });
  resizer.addEventListener("keydown", (event) => {
    const current = el.sidebar.getBoundingClientRect().width;
    let next = null;
    if (event.key === "ArrowLeft") next = current - 16;
    else if (event.key === "ArrowRight") next = current + 16;
    else if (event.key === "Home") next = SIDEBAR_DEFAULT_W;
    if (next == null) return;
    event.preventDefault();
    applySidebarWidth(next);
    saveSidebarWidth();
  });
}

// ---------- messages, tooltip, status ----------

function setLoading(active) {
  el.loadingBar.hidden = !active;
  el.chartWrap.setAttribute("aria-busy", String(active));
  if (!active && !state.data && state.seriesIndex == null) hideChartMsg();
}

function showChartMsg(kind, detail) {
  const box = el.chartMsg;
  box.textContent = "";
  box.hidden = false;
  if (kind === "loading") {
    const text = document.createElement("div");
    text.textContent = "加载中…";
    box.appendChild(text);
  } else if (kind === "error") {
    const text = document.createElement("div");
    text.className = "error-text";
    text.textContent = `加载失败：${detail}`;
    const retry = document.createElement("button");
    retry.type = "button";
    retry.className = "btn";
    retry.textContent = "重试";
    retry.addEventListener("click", () => {
      showChartMsg("loading");
      scheduleFetch(0);
    });
    box.append(text, retry);
  } else if (kind === "empty") {
    const text = document.createElement("div");
    text.textContent = detail;
    box.appendChild(text);
  }
}

function hideChartMsg() {
  el.chartMsg.hidden = true;
  el.chartMsg.textContent = "";
}

function updateTooltip(sampleIdx, clientX, clientY) {
  const data = state.data;
  if (!data || sampleIdx < 0) {
    hideTooltip();
    return;
  }
  const tooltip = el.tooltip;
  tooltip.textContent = "";
  const time = document.createElement("div");
  time.className = "tt-time";
  time.textContent = data.timestamp_labels[sampleIdx] ?? `#${data.indices[sampleIdx]}`;
  tooltip.appendChild(time);
  const visibleFeatures = data.features.slice(0, 12);
  visibleFeatures.forEach((feature) => {
    const value = feature.values[sampleIdx];
    const row = document.createElement("div");
    row.className = "tt-row";
    const swatch = document.createElement("span");
    swatch.className = "tt-swatch";
    const name = document.createElement("span");
    name.textContent = feature.name;
    const num = document.createElement("b");
    num.textContent = value === null || value === undefined ? "—" : formatNumber(value);
    row.append(swatch, name, num);
    tooltip.appendChild(row);
  });
  if (data.features.length > visibleFeatures.length) {
    const more = document.createElement("div");
    more.className = "tt-more";
    more.textContent = `另有 ${data.features.length - visibleFeatures.length} 个特征`;
    tooltip.appendChild(more);
  }
  const absolute = data.indices[sampleIdx];
  const inRun = data.label_runs.some(
    (run) => absolute >= run.start_index && absolute <= run.stop_index,
  );
  if (data.label_runs.length) {
    const flag = document.createElement("div");
    flag.className = "tt-flag";
    flag.textContent = inRun ? "● 标注区间内" : "○ 标注区间外";
    tooltip.appendChild(flag);
  }
  tooltip.hidden = false;
  const wrapRect = el.chartWrap.getBoundingClientRect();
  const tw = tooltip.offsetWidth;
  const th = tooltip.offsetHeight;
  const visibleLeft = el.chartWrap.scrollLeft;
  const visibleTop = el.chartWrap.scrollTop;
  let left = visibleLeft + clientX - wrapRect.left + 14;
  let top = visibleTop + clientY - wrapRect.top + 14;
  if (left + tw > visibleLeft + el.chartWrap.clientWidth - 8) {
    left = visibleLeft + clientX - wrapRect.left - tw - 14;
  }
  if (top + th > visibleTop + el.chartWrap.clientHeight - 8) {
    top = visibleTop + clientY - wrapRect.top - th - 14;
  }
  tooltip.style.left = `${Math.max(visibleLeft + 4, left)}px`;
  tooltip.style.top = `${Math.max(visibleTop + 4, top)}px`;
}

function hideTooltip() {
  el.tooltip.hidden = true;
}

function clearHoverInfo() {
  el.hoverInfo.textContent = "";
}

function updateWindowInfo() {
  if (state.seriesIndex == null || !state.data) {
    el.windowInfo.textContent = "";
    return;
  }
  const sampled = state.data.sampled_points;
  el.windowInfo.textContent =
    `窗口 [${state.pendingStart.toLocaleString("zh-CN")}, ` +
    `${(state.pendingStop ?? state.points).toLocaleString("zh-CN")}) / ` +
    `${state.points.toLocaleString("zh-CN")} 点 · 采样 ${sampled.toLocaleString("zh-CN")} 点`;
}

function updateChartSummary() {
  if (!state.data) {
    el.chartSummary.textContent = "尚未加载时间序列";
    return;
  }
  const data = state.data;
  const featureNames = data.features.map((feature) => feature.name).join("、");
  const winStart = state.pendingStart;
  const winStop = state.pendingStop ?? data.series.points;
  const summary =
    `序列 ${data.series.id}，窗口 ${winStart} 到 ${winStop}，` +
    `${winStop - winStart} 个数据点，显示 ${data.features.length} 个特征`;
  el.chartSummary.textContent = featureNames ? `${summary}：${featureNames}` : summary;
  el.chart.setAttribute("aria-label", summary);
}

function notify(message) {
  el.notice.textContent = message;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => {
    el.notice.textContent = "";
  }, 4000);
}

// ---------- hash ----------

function currentHashFields() {
  const full = state.points > 0 && state.pendingStart === 0 && state.pendingStop === state.points;
  return {
    s: state.seriesIndex,
    f: [...state.features].sort((a, b) => a - b),
    n: state.normalization,
    sc: state.scope,
    l: state.labelIndex,
    w: state.pendingStop != null && !full
      ? [state.pendingStart, state.pendingStop]
      : null,
  };
}

function syncHash() {
  const hash = `#${serializeHashState(currentHashFields())}`;
  if (hash === `#${lastHash}` || hash === location.hash) {
    lastHash = location.hash.replace(/^#/, "");
    return;
  }
  lastHash = hash.slice(1);
  history.replaceState(null, "", hash);
}

function applyHashFields(fields) {
  const ov = state.overview;
  let changed = false;
  const requestedFeatures = (fields.f ?? []).filter(
    (index) => index < ov.feature_count,
  );
  const featureIndices = requestedFeatures.length
    ? requestedFeatures.slice(0, state.maxFeatures)
    : ov.features.slice(0, Math.min(8, state.maxFeatures)).map((item) => item.index);
  const nextFeatures = new Set(featureIndices);
  if (
    nextFeatures.size !== state.features.size ||
    [...nextFeatures].some((index) => !state.features.has(index))
  ) {
    state.features = nextFeatures;
    changed = true;
    buildChips();
  }

  const normalization = ov.normalizations.some((item) => item.id === fields.n)
    ? fields.n
    : "none";
  if (normalization !== state.normalization) {
    state.normalization = normalization;
    changed = true;
  }
  el.normSelect.value = normalization;

  if (fields.sc !== state.scope) {
    state.scope = fields.sc;
    changed = true;
  }
  el.scopeSelect.value = fields.sc;

  const labelIndex = ov.binary_labels.some((item) => item.index === fields.l)
    ? fields.l
    : null;
  if (labelIndex !== state.labelIndex) {
    state.labelIndex = labelIndex;
    changed = true;
  }
  el.labelSelect.value = labelIndex == null ? "" : String(labelIndex);
  setControlsEnabled(true);
  return changed;
}

// ---------- interactions ----------

function bindInteractions() {
  const chart = el.chart;

  chart.addEventListener("pointerdown", (event) => {
    if (event.button !== 0 || !state.data) return;
    chart.setPointerCapture(event.pointerId);
    chart.classList.add("is-panning");
    const startX = event.clientX;
    const origStart = state.pendingStart;
    const origStop = state.pendingStop;
    const span = origStop - origStart;
    const cssW = el.chart.width / dpr;
    const onMove = (moveEvent) => {
      const delta = ((startX - moveEvent.clientX) / cssW) * span;
      const [s, e] = clampWindow(
        Math.round(origStart + delta),
        Math.round(origStop + delta),
        state.points,
      );
      state.pendingStart = s;
      state.pendingStop = e;
      syncHash();
      updateWindowInfo();
      requestRender();
      if (windowNeedsFetch()) scheduleFetch(0);
    };
    const onUp = (upEvent) => {
      chart.removeEventListener("pointermove", onMove);
      chart.removeEventListener("pointerup", onUp);
      chart.removeEventListener("pointercancel", onUp);
      chart.classList.remove("is-panning");
      if (chart.hasPointerCapture(upEvent.pointerId)) {
        chart.releasePointerCapture(upEvent.pointerId);
      }
      if (windowNeedsFetch()) scheduleFetch(0);
    };
    chart.addEventListener("pointermove", onMove);
    chart.addEventListener("pointerup", onUp);
    chart.addEventListener("pointercancel", onUp);
    void event;
  });

  chart.addEventListener("pointermove", (event) => {
    if (chart.classList.contains("is-panning") || !state.data) return;
    const rect = chart.getBoundingClientRect();
    const cssW = rect.width;
    const span = Math.max(1, (state.pendingStop ?? state.points) - state.pendingStart);
    const target = state.pendingStart + (event.clientX - rect.left) / cssW * span;
    const idx = nearestSampleIndex(state.data.indices, Math.round(target));
    if (idx !== state.hover) {
      state.hover = idx;
      requestRender();
    }
    if (idx >= 0) {
      updateTooltip(idx, event.clientX, event.clientY);
      const absolute = state.data.indices[idx];
      el.hoverInfo.textContent = `#${absolute.toLocaleString("zh-CN")} · ${state.data.timestamp_labels[idx] ?? ""}`;
    }
  });

  chart.addEventListener("pointerleave", () => {
    state.hover = null;
    hideTooltip();
    clearHoverInfo();
    requestRender();
  });

  chart.addEventListener(
    "wheel",
    (event) => {
      if (!state.data) return;
      const verticallyScrollable = el.chart.scrollHeight > el.chartWrap.clientHeight;
      if (verticallyScrollable && !event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      const rect = chart.getBoundingClientRect();
      const frac = (event.clientX - rect.left) / rect.width;
      const span = (state.pendingStop ?? state.points) - state.pendingStart;
      const anchor = state.pendingStart + frac * span;
      const factor = Math.exp(event.deltaY * 0.0016);
      zoomAt(anchor, factor);
    },
    { passive: false },
  );

  chart.addEventListener("dblclick", () => {
    if (state.points > 0) setWindow(0, state.points);
  });

  chart.addEventListener("keydown", (event) => {
    if (state.pendingStop == null) return;
    const span = state.pendingStop - state.pendingStart;
    const center = (state.pendingStart + state.pendingStop) / 2;
    let handled = true;
    if (event.key === "ArrowLeft") panBy(-Math.round(span * 0.1));
    else if (event.key === "ArrowRight") panBy(Math.round(span * 0.1));
    else if (event.key === "+" || event.key === "=") zoomAt(center, 0.8);
    else if (event.key === "-" || event.key === "_") zoomAt(center, 1.25);
    else if (event.key === "0" || event.key === "Home") setWindow(0, state.points);
    else handled = false;
    if (handled) event.preventDefault();
  });

  el.zoomIn.addEventListener("click", () => {
    zoomAt((state.pendingStart + (state.pendingStop ?? state.points)) / 2, 0.8);
  });
  el.zoomOut.addEventListener("click", () => {
    zoomAt((state.pendingStart + (state.pendingStop ?? state.points)) / 2, 1.25);
  });
  el.zoomReset.addEventListener("click", () => {
    if (state.points > 0) setWindow(0, state.points);
  });

  el.normSelect.addEventListener("change", () => {
    state.normalization = el.normSelect.value;
    setControlsEnabled(true);
    syncHash();
    scheduleFetch(0);
  });
  el.scopeSelect.addEventListener("change", () => {
    state.scope = el.scopeSelect.value;
    syncHash();
    scheduleFetch(0);
  });
  el.labelSelect.addEventListener("change", () => {
    state.labelIndex = el.labelSelect.value === "" ? null : Number(el.labelSelect.value);
    syncHash();
    scheduleFetch(0);
  });

  let searchTimer = 0;
  el.search.addEventListener("input", () => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      state.query = el.search.value.trim();
      fetchItems(false);
    }, 250);
  });
  el.moreBtn.addEventListener("click", () => fetchItems(true));

  el.tabFiles.addEventListener("click", () => switchTab("files"));
  el.tabSeries.addEventListener("click", () => switchTab("series"));
  bindResizer();
  for (const tab of [el.tabFiles, el.tabSeries]) {
    tab.addEventListener("keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
      event.preventDefault();
      const next = tab === el.tabFiles ? el.tabSeries : el.tabFiles;
      if (next.disabled) return;
      switchTab(next === el.tabFiles ? "files" : "series");
      next.focus();
    });
  }

  el.menuBtn.addEventListener("click", () => {
    const open = !el.sidebar.classList.contains("is-open");
    el.sidebar.classList.toggle("is-open", open);
    el.backdrop.hidden = !open;
    el.menuBtn.setAttribute("aria-expanded", String(open));
  });
  el.backdrop.addEventListener("click", closeDrawer);
  el.chartWrap.addEventListener("scroll", hideTooltip, { passive: true });
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && el.sidebar.classList.contains("is-open")) {
      closeDrawer();
      el.menuBtn.focus();
    }
  });
}

function closeDrawer() {
  el.sidebar.classList.remove("is-open");
  el.backdrop.hidden = true;
  el.menuBtn.setAttribute("aria-expanded", "false");
}

// ---------- boot ----------

async function boot() {
  Object.assign(el, {
    menuBtn: el9("menu-btn"),
    appTitle: el9("app-title"),
    stats: el9("stats"),
    loadingBar: el9("loading-bar"),
    sidebar: el9("sidebar"),
    resizer: el9("sidebar-resizer"),
    tabFiles: el9("tab-files"),
    tabSeries: el9("tab-series"),
    panelFiles: el9("panel-files"),
    panelSeries: el9("panel-series"),
    sidebarFoot: el9("sidebar-foot"),
    tree: el9("tree"),
    treeRoot: el9("tree-root"),
    search: el9("search"),
    seriesList: el9("series-list"),
    moreBtn: el9("more-btn"),
    listCount: el9("list-count"),
    manifestBox: el9("manifest-box"),
    manifestBody: el9("manifest-body"),
    backdrop: el9("backdrop"),
    normSelect: el9("norm-select"),
    scopeSelect: el9("scope-select"),
    labelSelect: el9("label-select"),
    zoomIn: el9("zoom-in"),
    zoomOut: el9("zoom-out"),
    zoomReset: el9("zoom-reset"),
    featureBar: el9("feature-bar"),
    chartWrap: el9("chart-wrap"),
    chart: el9("chart"),
    tooltip: el9("tooltip"),
    chartMsg: el9("chart-msg"),
    chartSummary: el9("chart-summary"),
    windowInfo: el9("window-info"),
    hoverInfo: el9("hover-info"),
    notice: el9("notice"),
  });
  ctx = el.chart.getContext("2d");
  bindInteractions();
  restoreSidebarWidth();

  let loaded = null;
  try {
    loaded = await api("/api/overview");
  } catch (error) {
    if (error.status !== 409) {
      showChartMsg("error", `无法读取集合信息（${error.message}）`);
    }
  }

  if (loaded) {
    state.overview = loaded;
    state.maxPoints = state.overview.limits?.default_max_points ?? 4000;
    state.hardMaxPoints = state.overview.limits?.hard_max_points ?? 50000;
    state.maxFeatures = state.overview.limits?.max_selected_features ?? 128;
    state.features = new Set(
      state.overview.features
        .slice(0, Math.min(8, state.maxFeatures))
        .map((feature) => feature.index),
    );
    buildStaticUI();

    const hashFields = parseHashState(location.hash);
    lastHash = location.hash.replace(/^#/, "");
    applyHashFields(hashFields);

    switchTab("series");
    const total = state.overview.series_count;
    if (total === 0) {
      showChartMsg("empty", "该集合不包含任何序列");
      syncHash();
    } else {
      const startSeries =
        hashFields.s != null && hashFields.s < total ? hashFields.s : 0;
      setSeries(startSeries, hashFields.w ? { window: hashFields.w } : {});
      fetchItems(false);
    }
  } else {
    enterBrowserMode();
  }

  try {
    const rootListing = await api("/api/tree");
    setTreeRootLabel(rootListing.root);
    await fetchTreeChildren("", el.tree, 0);
  } catch (error) {
    const failure = document.createElement("div");
    failure.className = "tree-empty";
    failure.textContent = `目录树加载失败：${error.message}`;
    el.tree.appendChild(failure);
  }
  if (state.overview?.path) {
    await revealTreePath(state.overview.path);
  }

  const onResize = () => {
    resizeCanvas();
    requestRender();
  };
  if (typeof ResizeObserver === "function") {
    new ResizeObserver(onResize).observe(el.chartWrap);
  } else {
    window.addEventListener("resize", onResize);
  }
  resizeCanvas();

  window.addEventListener("hashchange", () => {
    if (!state.overview) return;
    const next = location.hash.replace(/^#/, "");
    if (next === lastHash) return;
    lastHash = next;
    const fields = parseHashState(location.hash);
    const selectionChanged = applyHashFields(fields);
    const totalNow = state.overview.series_count;
    const targetSeries = fields.s != null && fields.s < totalNow ? fields.s : 0;
    if (targetSeries !== state.seriesIndex) {
      setSeries(targetSeries, fields.w ? { window: fields.w } : {});
      return;
    }
    let windowChanged = false;
    if (state.points > 0) {
      windowChanged = fields.w
        ? setWindow(fields.w[0], fields.w[1])
        : setWindow(0, state.points);
    }
    if (selectionChanged && !windowChanged) {
      scheduleFetch(0);
    }
    syncHash();
  });
}

if (typeof document !== "undefined") {
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
}

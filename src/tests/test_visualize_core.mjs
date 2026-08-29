import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const sourceUrl = new URL("../agentad/visualize/static/core.js", import.meta.url);
const source = await readFile(sourceUrl, "utf8");
const core = await import(
  `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`
);

test("clampWindow handles empty and short series", () => {
  assert.deepEqual(core.clampWindow(0, 100, 0), [0, 0]);
  assert.deepEqual(core.clampWindow(0, 100, 1), [0, 1]);
  assert.deepEqual(core.clampWindow(-5, 2, 10), [0, 4]);
  assert.deepEqual(core.clampWindow(8, 20, 10), [6, 10]);
});

test("panel widths stay within accessible resize bounds", () => {
  assert.equal(core.clampPanelWidth(120), 240);
  assert.equal(core.clampPanelWidth(337.4), 337);
  assert.equal(core.clampPanelWidth(900), 520);
  assert.equal(core.clampPanelWidth(Number.NaN, 260, 480), 260);
});

test("track canvases keep a fixed lane height and stack only when needed", () => {
  assert.equal(core.chartCanvasHeight(1, "stacked"), 132);
  assert.equal(core.chartCanvasHeight(3, "stacked"), 324);
  assert.equal(core.chartCanvasHeight(20, "overlay"), 132);
});

test("tree updates preserve untouched branches", () => {
  const right = { key: "right", name: "right" };
  const tree = [
    { key: "left", children: [{ key: "target", children: undefined }] },
    right,
  ];
  const children = [{ key: "child" }];
  const next = core.replaceTreeChildren(tree, "target", children);
  assert.notEqual(next, tree);
  assert.notEqual(next[0], tree[0]);
  assert.equal(next[1], right);
  assert.deepEqual(next[0].children[0], { key: "target", children, isLeaf: false });
  assert.equal(core.replaceTreeChildren(next, "missing", []), next);
});

test("last-tab preferences are validated for new independent tabs", () => {
  const preferences = core.parseTabPreferences(JSON.stringify({
    normalization: "robust",
    scope: "global",
    labelName: "is_anomaly",
    mode: "select",
    layout: "overlay",
    transform: { rotation: 500, scaleX: 0, scaleY: 8, flipX: true },
    inspectorOpen: true,
    inspectorWidth: 900,
  }));
  assert.deepEqual(preferences, {
    normalization: "robust",
    scope: "global",
    labelIndex: null,
    labelName: "is_anomaly",
    mode: "select",
    layout: "overlay",
    transform: {
      rotation: 180,
      scaleX: 0.25,
      scaleY: 4,
      flipX: true,
      flipY: false,
    },
    inspectorOpen: true,
    inspectorWidth: 560,
  });
  assert.deepEqual(
    core.parseTabPreferences(core.serializeTabPreferences(preferences)),
    preferences,
  );
});

test("wheel deltas work for mice and macOS trackpads", () => {
  assert.equal(core.normalizeWheelDelta(3, 0, 1), 48);
  assert.equal(core.normalizeWheelDelta(0, -4, 1), -64);
  assert.equal(core.normalizeWheelDelta(1000, 0, 0), 240);
  assert.ok(core.zoomScale(1, -100) > 1);
  assert.ok(core.zoomScale(1, 100) < 1);
  assert.equal(core.zoomScale(4, -1000), 4);
  assert.equal(core.zoomScale(0.25, 1000), 0.25);
});

test("tooltips choose below, centered, and above placements", () => {
  const viewport = { width: 800, height: 600 };
  const tooltip = { width: 200, height: 120 };
  assert.deepEqual(core.positionTooltip({ x: 400, y: 20 }, viewport, tooltip), {
    x: 414,
    y: 34,
    horizontal: "right",
    vertical: "below",
  });
  assert.deepEqual(core.positionTooltip({ x: 400, y: 300 }, viewport, tooltip), {
    x: 414,
    y: 240,
    horizontal: "right",
    vertical: "center",
  });
  assert.deepEqual(core.positionTooltip({ x: 790, y: 580 }, viewport, tooltip), {
    x: 576,
    y: 446,
    horizontal: "left",
    vertical: "above",
  });
});

test("nearestSampleIndex uses ordered nearest-neighbor lookup", () => {
  assert.equal(core.nearestSampleIndex([], 5), -1);
  assert.equal(core.nearestSampleIndex([0, 10, 20], 6), 1);
  assert.equal(core.nearestSampleIndex([0, 10, 20], 14), 1);
});

test("hash parsing returns a complete validated default state", () => {
  assert.deepEqual(core.parseHashState(""), {
    s: null,
    f: null,
    n: "none",
    sc: "feature",
    l: null,
    w: null,
  });
  const parsed = core.parseHashState("#s=2&f=3,3,1&n=robust&sc=global&l=4&w=8-20");
  assert.deepEqual(parsed, {
    s: 2,
    f: [3, 1],
    n: "robust",
    sc: "global",
    l: 4,
    w: [8, 20],
  });
  assert.equal(
    core.serializeHashState(parsed),
    "s=2&f=3%2C1&n=robust&sc=global&l=4&w=8-20",
  );
});

test("value domains remain drawable for constants and extreme values", () => {
  const constant = core.valueDomain([1e308, 1e308]);
  assert.ok(constant.high > constant.low);
  assert.ok(Number.isFinite(core.valueToY(1e308, constant, 0, 100)));

  const extreme = core.valueDomain([-1e308, 1e308]);
  assert.ok(extreme.high > extreme.low);
  assert.ok(Number.isFinite(core.valueToY(-1e308, extreme, 0, 100)));
  assert.equal(core.valueDomain([null, undefined, Number.NaN]), null);
});

test("timeline pan and anchored zoom remain inside the series", () => {
  assert.deepEqual(core.panWindow([20, 40], -100, 100), [0, 20]);
  assert.deepEqual(core.panWindow([20, 40], 100, 100), [80, 100]);
  assert.deepEqual(core.zoomWindow([20, 60], 0.5, 30, 100), [25, 45]);
  assert.deepEqual(core.zoomWindow([0, 100], 2, 50, 100), [0, 100]);
});

test("2D point buffers preserve gaps and apply rotation", () => {
  const point = core.transformPoint2D(
    2,
    1,
    { rotation: 90, scaleX: 1, scaleY: 1, flipX: false, flipY: false },
    1,
    1,
  );
  assert.ok(Math.abs(point[0] - 1) < 1e-12);
  assert.ok(Math.abs(point[1] - 2) < 1e-12);

  const points = core.buildPolyline2D(
    [0, 1, 2],
    [2, null, 4],
    (value) => value,
    (value) => value * 2,
    { rotation: 0, scaleX: 1, scaleY: 1, flipX: false, flipY: false },
    0,
    0,
  );
  assert.deepEqual([...points.slice(0, 2)], [0, 4]);
  assert.ok(Number.isNaN(points[2]) && Number.isNaN(points[3]));
  assert.deepEqual([...points.slice(4, 6)], [2, 8]);
});

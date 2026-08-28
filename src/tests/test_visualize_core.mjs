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

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

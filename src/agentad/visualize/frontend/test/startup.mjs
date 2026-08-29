import assert from "node:assert/strict";
import test from "node:test";

import { Window } from "happy-dom";

test("production bundle mounts the application shell without runtime errors", async () => {
  const browser = new Window({ url: "http://127.0.0.1:8765/" });
  browser.document.body.innerHTML = '<div id="root"></div>';
  browser.fetch = async () =>
    new browser.Response(JSON.stringify({ error: "no collection loaded" }), {
      status: 409,
      headers: { "Content-Type": "application/json" },
    });

  for (const name of [
    "window",
    "document",
    "navigator",
    "HTMLElement",
    "HTMLDialogElement",
    "SVGElement",
    "Element",
    "Node",
    "MutationObserver",
    "ResizeObserver",
    "getComputedStyle",
    "requestAnimationFrame",
    "cancelAnimationFrame",
    "localStorage",
  ]) {
    Object.defineProperty(globalThis, name, {
      configurable: true,
      value: name === "window" ? browser : browser[name],
    });
  }
  globalThis.fetch = browser.fetch;

  const errors = [];
  browser.addEventListener("error", (event) => {
    errors.push(String(event.error ?? event.message));
  });

  await import(`../../static/app.js?startup=${Date.now()}`);
  await new Promise((resolve) => setTimeout(resolve, 300));

  assert.deepEqual(errors, []);
  assert.equal(
    browser.document.getElementById("root")?.firstElementChild?.className,
    "app-shell",
  );
  browser.close();
});

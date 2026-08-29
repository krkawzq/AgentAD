import { useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";

import {
  MIN_SPAN,
  chartCanvasHeight,
  clampWindow,
  formatNumber,
  nearestSampleIndex,
  panWindow,
  zoomWindow,
} from "../../static/core.js";
import { chartGeometry, renderChart, renderInteractionOverlay } from "./renderer.js";

const MAX_CANVAS_PIXELS = 24_000_000;
const MIN_SELECTION_PIXELS = 8;

function pointerPosition(event, canvas) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: event.clientX - bounds.left,
    y: event.clientY - bounds.top,
  };
}

function Tooltip({ data, index, position }) {
  if (index == null || data?.indices[index] === undefined) return null;
  const absoluteIndex = data.indices[index];
  const inLabel = data.label_runs.some(
    (run) => absoluteIndex >= run.start_index && absoluteIndex <= run.stop_index,
  );
  return (
    <div
      className="chart-tooltip"
      style={{ left: position.x + 14, top: position.y + 14 }}
      role="status"
    >
      <div className="tooltip-time">
        {data.timestamp_labels[index] ?? `Point ${absoluteIndex}`}
      </div>
      {data.features.slice(0, 10).map((feature) => (
        <div className="tooltip-row" key={feature.index}>
          <span
            className="track-swatch"
            style={{ "--track-color": `var(--track-${feature.index % 8})` }}
          />
          <span>{feature.name}</span>
          <strong>{formatNumber(feature.values[index])}</strong>
        </div>
      ))}
      {data.features.length > 10 && (
        <div className="tooltip-more">+{data.features.length - 10} more tracks</div>
      )}
      {data.label_runs.length > 0 && (
        <div className={`tooltip-label ${inLabel ? "is-active" : ""}`}>
          {inLabel ? "Inside selected label" : "Outside selected label"}
        </div>
      )}
    </div>
  );
}

export function CanvasEditor({
  data,
  viewport,
  totalPoints,
  layout,
  transform,
  mode,
  onModeChange,
  onTransformChange,
  onViewportChange,
}) {
  const scrollRef = useRef(null);
  const baseCanvasRef = useRef(null);
  const canvasRef = useRef(null);
  const interactionRef = useRef(null);
  const spaceRef = useRef(false);
  const [size, setSize] = useState({ width: 900, viewportHeight: 520 });
  const [selection, setSelection] = useState(null);
  const [hover, setHover] = useState({ index: null, position: { x: 0, y: 0 } });

  const canvasHeight = useMemo(
    () => chartCanvasHeight(data?.features.length ?? 0, layout),
    [data?.features.length, layout],
  );

  const dpr = useMemo(() => {
    const ideal = Math.min(window.devicePixelRatio || 1, 2);
    return Math.min(
      ideal,
      Math.sqrt(MAX_CANVAS_PIXELS / Math.max(1, size.width * canvasHeight)),
    );
  }, [canvasHeight, size.width]);

  useLayoutEffect(() => {
    const element = scrollRef.current;
    if (!element) return undefined;
    const update = () => {
      setSize({
        width: Math.max(320, element.clientWidth),
        viewportHeight: Math.max(280, element.clientHeight),
      });
    };
    update();
    if (typeof ResizeObserver !== "function") {
      window.addEventListener("resize", update);
      return () => window.removeEventListener("resize", update);
    }
    const observer = new ResizeObserver(update);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    const canvas = baseCanvasRef.current;
    if (!canvas) return undefined;
    canvas.width = Math.max(1, Math.round(size.width * dpr));
    canvas.height = Math.max(1, Math.round(canvasHeight * dpr));
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${canvasHeight}px`;
    const frame = requestAnimationFrame(() => {
      renderChart(canvas, {
        data,
        viewport,
        layout,
        transform,
        selection: null,
        hoverIndex: null,
        width: size.width,
        height: canvasHeight,
        dpr,
      });
    });
    return () => cancelAnimationFrame(frame);
  }, [canvasHeight, data, dpr, layout, size.width, transform, viewport]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return undefined;
    canvas.width = Math.max(1, Math.round(size.width * dpr));
    canvas.height = Math.max(1, Math.round(canvasHeight * dpr));
    canvas.style.width = `${size.width}px`;
    canvas.style.height = `${canvasHeight}px`;
    const frame = requestAnimationFrame(() => {
      renderInteractionOverlay(canvas, {
        data,
        viewport,
        layout,
        selection,
        hoverIndex: hover.index,
        width: size.width,
        height: canvasHeight,
        dpr,
      });
    });
    return () => cancelAnimationFrame(frame);
  }, [canvasHeight, data, dpr, hover.index, layout, selection, size.width, viewport]);

  useEffect(() => {
    const onKeyUp = (event) => {
      if (event.code === "Space") spaceRef.current = false;
    };
    window.addEventListener("keyup", onKeyUp);
    return () => window.removeEventListener("keyup", onKeyUp);
  }, []);

  const commitViewport = (next) => {
    const clamped = clampWindow(next[0], next[1], totalPoints);
    onViewportChange(clamped, true);
  };

  const handlePointerDown = (event) => {
    if (event.button !== 0 || !data || totalPoints <= 0) return;
    const canvas = canvasRef.current;
    const point = pointerPosition(event, canvas);
    canvas.focus();
    canvas.setPointerCapture(event.pointerId);
    const action = mode === "pan" || spaceRef.current ? "pan" : "select";
    interactionRef.current = {
      action,
      pointerId: event.pointerId,
      origin: point,
      originalViewport: viewport,
      lastViewport: viewport,
      lastPoint: point,
    };
    if (action === "select") {
      setSelection({ startX: point.x, startY: point.y, endX: point.x, endY: point.y });
    }
  };

  const handlePointerMove = (event) => {
    const canvas = canvasRef.current;
    if (!canvas || !data) return;
    const point = pointerPosition(event, canvas);
    const interaction = interactionRef.current;
    if (interaction) {
      interaction.lastPoint = point;
      if (interaction.action === "pan") {
        const geometry = chartGeometry(size.width, canvasHeight, data.features.length, layout);
        const span = interaction.originalViewport[1] - interaction.originalViewport[0];
        const delta =
          ((interaction.origin.x - point.x) / (geometry.right - geometry.left)) * span;
        const next = panWindow(interaction.originalViewport, delta, totalPoints);
        interaction.lastViewport = next;
        onViewportChange(next, false);
      } else {
        setSelection((current) => ({ ...current, endX: point.x, endY: point.y }));
      }
      return;
    }

    const geometry = chartGeometry(size.width, canvasHeight, data.features.length, layout);
    const fraction = Math.max(
      0,
      Math.min(1, (point.x - geometry.left) / (geometry.right - geometry.left)),
    );
    const target = viewport[0] + fraction * (viewport[1] - viewport[0]);
    const index = nearestSampleIndex(data.indices, target);
    setHover({ index, position: point });
  };

  const finishPointer = (event) => {
    const canvas = canvasRef.current;
    const interaction = interactionRef.current;
    if (!canvas || !interaction) return;
    if (interaction.action === "pan") {
      commitViewport(interaction.lastViewport);
    } else {
      const endPoint = interaction.lastPoint;
      const currentSelection = {
        startX: interaction.origin.x,
        startY: interaction.origin.y,
        endX: endPoint.x,
        endY: endPoint.y,
      };
      const geometry = chartGeometry(size.width, canvasHeight, data?.features.length ?? 0, layout);
      const left = Math.max(
        geometry.left,
        Math.min(currentSelection.startX, currentSelection.endX),
      );
      const right = Math.min(
        geometry.right,
        Math.max(currentSelection.startX, currentSelection.endX),
      );
      if (right - left >= MIN_SELECTION_PIXELS) {
        const span = viewport[1] - viewport[0];
        const plotWidth = geometry.right - geometry.left;
        commitViewport([
          Math.floor(viewport[0] + ((left - geometry.left) / plotWidth) * span),
          Math.ceil(viewport[0] + ((right - geometry.left) / plotWidth) * span),
        ]);
      }
      setSelection(null);
    }
    interactionRef.current = null;
    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
  };

  const handleWheel = (event) => {
    if (!data || totalPoints <= 0) return;
    const scroll = scrollRef.current;
    const tracksOverflow =
      layout === "stacked" && scroll && scroll.scrollHeight > scroll.clientHeight + 1;
    if (tracksOverflow && !event.ctrlKey && !event.metaKey && !event.shiftKey && !event.altKey) {
      return;
    }
    event.preventDefault();
    const span = viewport[1] - viewport[0];
    if (event.shiftKey || event.altKey) {
      const direction = event.deltaY || event.deltaX;
      commitViewport(panWindow(viewport, (direction / 500) * span, totalPoints));
      return;
    }
    const canvas = canvasRef.current;
    const point = pointerPosition(event, canvas);
    const geometry = chartGeometry(size.width, canvasHeight, data.features.length, layout);
    const fraction = Math.max(
      0,
      Math.min(1, (point.x - geometry.left) / (geometry.right - geometry.left)),
    );
    const anchor = viewport[0] + fraction * span;
    commitViewport(zoomWindow(viewport, Math.exp(event.deltaY * 0.0015), anchor, totalPoints));
  };

  const handleKeyDown = (event) => {
    if (event.code === "Space") {
      spaceRef.current = true;
      event.preventDefault();
      return;
    }
    if (!data || totalPoints <= 0) return;
    const span = viewport[1] - viewport[0];
    const center = (viewport[0] + viewport[1]) / 2;
    let next = null;
    if (event.key.toLowerCase() === "v") onModeChange("pan");
    else if (event.key.toLowerCase() === "z") onModeChange("select");
    else if (event.key === "ArrowLeft") {
      next = panWindow(viewport, -span * (event.shiftKey ? 0.3 : 0.1), totalPoints);
    } else if (event.key === "ArrowRight") {
      next = panWindow(viewport, span * (event.shiftKey ? 0.3 : 0.1), totalPoints);
    } else if (event.key === "+" || event.key === "=") {
      next = zoomWindow(viewport, 0.75, center, totalPoints);
    } else if (event.key === "-" || event.key === "_") {
      next = zoomWindow(viewport, 1.35, center, totalPoints);
    } else if (event.key === "0" || event.key === "Home") {
      next = [0, totalPoints];
    } else if (event.key === "[") {
      onTransformChange({ ...transform, rotation: transform.rotation - 15 });
    } else if (event.key === "]") {
      onTransformChange({ ...transform, rotation: transform.rotation + 15 });
    } else if (event.key.toLowerCase() === "h") {
      onTransformChange({ ...transform, flipX: !transform.flipX });
    } else {
      return;
    }
    event.preventDefault();
    if (next) commitViewport(next);
  };

  const tooltipPosition = {
    x: Math.min(hover.position.x, size.width - 290),
    y: Math.min(hover.position.y, size.viewportHeight - 220) + (scrollRef.current?.scrollTop ?? 0),
  };

  return (
    <div
      className="canvas-shell"
      style={{ "--canvas-content-height": `${canvasHeight}px` }}
    >
      <div className="canvas-scroll" ref={scrollRef}>
        <canvas
          ref={baseCanvasRef}
          className="series-canvas series-canvas-base"
          aria-hidden="true"
        />
        <canvas
          ref={canvasRef}
          className={`series-canvas series-canvas-overlay mode-${mode}`}
          tabIndex="0"
          aria-label="Time-series editor. Press Z for box zoom, V for pan, plus or minus to zoom, arrow keys to pan, and zero to fit."
          aria-describedby="chart-summary"
          onDoubleClick={() => commitViewport([0, totalPoints])}
          onKeyDown={handleKeyDown}
          onPointerCancel={finishPointer}
          onPointerDown={handlePointerDown}
          onPointerLeave={() => setHover({ index: null, position: { x: 0, y: 0 } })}
          onPointerMove={handlePointerMove}
          onPointerUp={finishPointer}
          onWheel={handleWheel}
        >
          This visualization requires a browser with Canvas support.
        </canvas>
        <Tooltip data={data} index={hover.index} position={tooltipPosition} />
      </div>
      <p id="chart-summary" className="sr-only" aria-live="polite">
        {data
          ? `Series ${data.series.id}, showing ${data.features.length} tracks from point ${viewport[0]} to ${viewport[1]}.`
          : "No time series loaded."}
      </p>
    </div>
  );
}

export function TimelineNavigator({ viewport, totalPoints, onViewportChange }) {
  const railRef = useRef(null);
  const dragRef = useRef(null);
  const total = Math.max(1, totalPoints);
  const left = (viewport[0] / total) * 100;
  const width = Math.max(0.6, ((viewport[1] - viewport[0]) / total) * 100);

  const begin = (event, action) => {
    if (totalPoints <= 0) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      action,
      pointerId: event.pointerId,
      startX: event.clientX,
      original: viewport,
      last: viewport,
    };
  };

  const move = (event) => {
    const drag = dragRef.current;
    const rail = railRef.current;
    if (!drag || !rail) return;
    const delta = ((event.clientX - drag.startX) / rail.clientWidth) * totalPoints;
    let next;
    if (drag.action === "move") {
      next = panWindow(drag.original, delta, totalPoints);
    } else if (drag.action === "start") {
      const minimumSpan = Math.min(MIN_SPAN, totalPoints);
      const start = Math.max(
        0,
        Math.min(
          Math.round(drag.original[0] + delta),
          drag.original[1] - minimumSpan,
        ),
      );
      next = [start, drag.original[1]];
    } else {
      const minimumSpan = Math.min(MIN_SPAN, totalPoints);
      const stop = Math.min(
        totalPoints,
        Math.max(
          Math.round(drag.original[1] + delta),
          drag.original[0] + minimumSpan,
        ),
      );
      next = [drag.original[0], stop];
    }
    drag.last = next;
    onViewportChange(next, false);
  };

  const finish = (event) => {
    const drag = dragRef.current;
    if (!drag) return;
    onViewportChange(drag.last, true);
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const jump = (event) => {
    if (event.target !== railRef.current || totalPoints <= 0) return;
    const bounds = railRef.current.getBoundingClientRect();
    const center = ((event.clientX - bounds.left) / bounds.width) * totalPoints;
    const span = viewport[1] - viewport[0];
    const next = clampWindow(center - span / 2, center + span / 2, totalPoints);
    onViewportChange(next, true);
  };

  return (
    <div className="timeline" aria-label="Timeline viewport">
      <div className="timeline-rail" ref={railRef} onPointerDown={jump}>
        <div className="timeline-window" style={{ left: `${left}%`, width: `${width}%` }}>
          <button
            type="button"
            className="timeline-handle is-start"
            aria-label="Resize viewport start"
            onPointerCancel={finish}
            onPointerDown={(event) => begin(event, "start")}
            onPointerMove={move}
            onPointerUp={finish}
          />
          <button
            type="button"
            className="timeline-thumb"
            aria-label="Move timeline viewport"
            onPointerCancel={finish}
            onPointerDown={(event) => begin(event, "move")}
            onPointerMove={move}
            onPointerUp={finish}
          />
          <button
            type="button"
            className="timeline-handle is-stop"
            aria-label="Resize viewport end"
            onPointerCancel={finish}
            onPointerDown={(event) => begin(event, "stop")}
            onPointerMove={move}
            onPointerUp={finish}
          />
        </div>
      </div>
    </div>
  );
}

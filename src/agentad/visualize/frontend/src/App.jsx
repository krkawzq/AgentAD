import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  BarChart3,
  CircleHelp,
  Hand,
  LoaderCircle,
  Maximize2,
  Menu,
  MousePointer2,
  PanelRight,
  Rows3,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

import { clampPanelWidth, zoomWindow } from "../../static/core.js";
import { api, ApiError } from "./api.js";
import { CanvasEditor, TimelineNavigator } from "./CanvasEditor.jsx";
import { Inspector } from "./Inspector.jsx";
import { Sidebar } from "./Sidebar.jsx";

const DEFAULT_TRANSFORM = Object.freeze({
  rotation: 0,
  scaleX: 1,
  scaleY: 1,
  flipX: false,
  flipY: false,
});
const SIDEBAR_DEFAULT_WIDTH = 300;
const SIDEBAR_MIN_WIDTH = 240;
const SIDEBAR_MAX_WIDTH = 520;
const SIDEBAR_WIDTH_KEY = "agentad-visualizer:sidebar-width";

function Header({ overview, activePath, onMenu, onInspector, onHelp }) {
  return (
    <header className="app-header">
      <div className="header-brand">
        <button type="button" className="mobile-icon-button" aria-label="Open navigation" onClick={onMenu}>
          <Menu size={18} />
        </button>
        <div>
          <h1>{overview?.title ?? "AgentAD Visualizer"}</h1>
          <p>Scientific time-series workspace</p>
        </div>
      </div>
      <div className="header-context" aria-label="Dataset summary">
        {overview ? (
          <>
            <span className="source-path" title={activePath || "In-memory collection"}>
              {activePath || "In-memory collection"}
            </span>
            <span><strong>{overview.series_count.toLocaleString()}</strong> series</span>
            <span><strong>{overview.point_count.toLocaleString()}</strong> points</span>
            <span><strong>{overview.feature_count.toLocaleString()}</strong> features</span>
          </>
        ) : (
          <span className="source-path">No data source open</span>
        )}
      </div>
      <div className="header-actions">
        <button type="button" aria-label="Show keyboard reference" title="Keyboard reference" onClick={onHelp}>
          <CircleHelp size={17} />
        </button>
        <button type="button" aria-label="Open inspector" title="Open inspector" onClick={onInspector}>
          <PanelRight size={17} />
        </button>
      </div>
    </header>
  );
}

function SidebarResizer({ width, workspaceRef, onResize }) {
  const drag = useRef(null);

  const preview = (nextWidth, element) => {
    const next = clampPanelWidth(nextWidth, SIDEBAR_MIN_WIDTH, SIDEBAR_MAX_WIDTH);
    if (drag.current) drag.current.lastWidth = next;
    workspaceRef.current?.style.setProperty("--sidebar-width", `${next}px`);
    element.setAttribute("aria-valuenow", String(next));
  };

  const finish = (event) => {
    if (!drag.current) return;
    const next = drag.current.lastWidth;
    drag.current = null;
    document.body.classList.remove("is-resizing-sidebar");
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    onResize(next);
  };

  return (
    <div
      className="sidebar-resizer"
      role="separator"
      tabIndex="0"
      aria-label="Resize data sidebar"
      aria-controls="data-sidebar"
      aria-orientation="vertical"
      aria-valuemin={SIDEBAR_MIN_WIDTH}
      aria-valuemax={SIDEBAR_MAX_WIDTH}
      aria-valuenow={width}
      title="Drag to resize; double-click to reset"
      onDoubleClick={(event) => {
        preview(SIDEBAR_DEFAULT_WIDTH, event.currentTarget);
        onResize(SIDEBAR_DEFAULT_WIDTH);
      }}
      onKeyDown={(event) => {
        let next = null;
        const step = event.shiftKey ? 48 : 16;
        if (event.key === "ArrowLeft") next = width - step;
        else if (event.key === "ArrowRight") next = width + step;
        else if (event.key === "Home") next = SIDEBAR_DEFAULT_WIDTH;
        if (next === null) return;
        event.preventDefault();
        onResize(clampPanelWidth(next, SIDEBAR_MIN_WIDTH, SIDEBAR_MAX_WIDTH));
      }}
      onPointerCancel={finish}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        drag.current = { lastWidth: width };
        event.currentTarget.setPointerCapture(event.pointerId);
        document.body.classList.add("is-resizing-sidebar");
      }}
      onPointerMove={(event) => {
        if (!drag.current || !workspaceRef.current) return;
        const bounds = workspaceRef.current.getBoundingClientRect();
        preview(event.clientX - bounds.left, event.currentTarget);
      }}
      onPointerUp={finish}
    >
      <span aria-hidden="true" />
    </div>
  );
}

function Toolbar({
  overview,
  normalization,
  scope,
  labelIndex,
  mode,
  layout,
  disabled,
  onNormalization,
  onScope,
  onLabel,
  onMode,
  onLayout,
  onZoom,
  onFit,
}) {
  return (
    <div className="editor-toolbar" aria-label="Visualization controls">
      <div className="toolbar-group">
        <span className="toolbar-caption">TOOL</span>
        <div className="segmented">
          <button
            type="button"
            className={mode === "select" ? "is-active" : ""}
            aria-pressed={mode === "select"}
            title="Box zoom (Z)"
            onClick={() => onMode("select")}
          >
            <MousePointer2 size={14} /> Select
          </button>
          <button
            type="button"
            className={mode === "pan" ? "is-active" : ""}
            aria-pressed={mode === "pan"}
            title="Pan (V or Space)"
            onClick={() => onMode("pan")}
          >
            <Hand size={14} /> Pan
          </button>
        </div>
      </div>
      <div className="toolbar-group">
        <span className="toolbar-caption">VIEW</span>
        <div className="icon-cluster">
          <button type="button" aria-label="Zoom out" title="Zoom out (-)" disabled={disabled} onClick={() => onZoom(1.35)}>
            <ZoomOut size={15} />
          </button>
          <button type="button" aria-label="Zoom in" title="Zoom in (+)" disabled={disabled} onClick={() => onZoom(0.75)}>
            <ZoomIn size={15} />
          </button>
          <button type="button" aria-label="Fit all points" title="Fit all (0)" disabled={disabled} onClick={onFit}>
            <Maximize2 size={15} />
          </button>
        </div>
      </div>
      <div className="toolbar-divider" />
      <label className="toolbar-field">
        <span>Normalization</span>
        <select value={normalization} disabled={disabled} onChange={(event) => onNormalization(event.target.value)}>
          {(overview?.normalizations ?? []).map((option) => (
            <option value={option.id} key={option.id}>{option.name}</option>
          ))}
        </select>
      </label>
      <label className="toolbar-field">
        <span>Scope</span>
        <select value={scope} disabled={disabled} onChange={(event) => onScope(event.target.value)}>
          <option value="feature">Per feature</option>
          <option value="global">Global</option>
        </select>
      </label>
      <label className="toolbar-field">
        <span>Label overlay</span>
        <select value={labelIndex ?? ""} disabled={disabled} onChange={(event) => onLabel(event.target.value === "" ? null : Number(event.target.value))}>
          <option value="">None</option>
          {(overview?.binary_labels ?? []).map((label) => (
            <option value={label.index} key={label.index}>{label.name}</option>
          ))}
        </select>
      </label>
      <div className="toolbar-spacer" />
      <div className="toolbar-group">
        <span className="toolbar-caption">TRACKS</span>
        <div className="segmented">
          <button
            type="button"
            className={layout === "stacked" ? "is-active" : ""}
            aria-pressed={layout === "stacked"}
            onClick={() => onLayout("stacked")}
          >
            <Rows3 size={14} /> Stacked
          </button>
          <button
            type="button"
            className={layout === "overlay" ? "is-active" : ""}
            aria-pressed={layout === "overlay"}
            onClick={() => onLayout("overlay")}
          >
            <BarChart3 size={14} /> Overlay
          </button>
        </div>
      </div>
    </div>
  );
}

function EmptyState({ overview, error, loading, onBrowse }) {
  if (loading) {
    return (
      <div className="empty-state" role="status">
        <LoaderCircle className="spin" size={24} />
        <h2>Preparing the visualization</h2>
        <p>Reading the selected window and generating bounded chart geometry.</p>
      </div>
    );
  }
  if (error) {
    return (
      <div className="empty-state is-error" role="alert">
        <h2>The chart could not be loaded</h2>
        <p>{error}</p>
      </div>
    );
  }
  if (!overview) {
    return (
      <div className="empty-state">
        <div className="empty-figure" aria-hidden="true">
          <span /><span /><span /><span />
        </div>
        <span className="eyebrow">START A WORKSPACE</span>
        <h2>Open a SeriesData package or contract-based CSV</h2>
        <p>Choose a local source, then inspect features as editable timeline tracks.</p>
        <button type="button" className="primary-button" onClick={onBrowse}>Browse data sources</button>
      </div>
    );
  }
  return (
    <div className="empty-state">
      <h2>No series available</h2>
      <p>The selected collection does not contain a drawable time series.</p>
    </div>
  );
}

function StatusBar({ data, viewport, totalPoints, loading, notice }) {
  return (
    <footer className="status-bar">
      <span className="status-indicator"><i className={loading ? "is-busy" : ""} />{loading ? "Updating view" : "Ready"}</span>
      {data && (
        <>
          <span>Window [{viewport[0].toLocaleString()}, {viewport[1].toLocaleString()}) of {totalPoints.toLocaleString()}</span>
          <span>{data.sampled_points.toLocaleString()} sampled points</span>
          <span>{data.features.length} visible tracks</span>
        </>
      )}
      <span className="status-notice" role="status" aria-live="polite">{notice}</span>
    </footer>
  );
}

export function App() {
  const [overview, setOverview] = useState(null);
  const [activePath, setActivePath] = useState(null);
  const [openingPath, setOpeningPath] = useState(null);
  const [activeTab, setActiveTab] = useState("sources");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [items, setItems] = useState([]);
  const [itemsTotal, setItemsTotal] = useState(0);
  const [itemsLoading, setItemsLoading] = useState(false);
  const [seriesQuery, setSeriesQuery] = useState("");
  const [debouncedQuery, setDebouncedQuery] = useState("");
  const [seriesIndex, setSeriesIndex] = useState(null);
  const [seriesPoints, setSeriesPoints] = useState(0);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [normalization, setNormalization] = useState("none");
  const [scope, setScope] = useState("feature");
  const [labelIndex, setLabelIndex] = useState(null);
  const [viewport, setViewport] = useState([0, 0]);
  const [committedViewport, setCommittedViewport] = useState([0, 0]);
  const [data, setData] = useState(null);
  const [dataLoading, setDataLoading] = useState(false);
  const [dataError, setDataError] = useState("");
  const [booting, setBooting] = useState(true);
  const [mode, setMode] = useState("select");
  const [layout, setLayout] = useState("stacked");
  const [transform, setTransform] = useState({ ...DEFAULT_TRANSFORM });
  const [notice, setNotice] = useState("");
  const [sidebarWidth, setSidebarWidth] = useState(() => {
    try {
      const stored = Number(window.localStorage.getItem(SIDEBAR_WIDTH_KEY));
      if (!Number.isFinite(stored) || stored <= 0) return SIDEBAR_DEFAULT_WIDTH;
      return clampPanelWidth(
        stored,
        SIDEBAR_MIN_WIDTH,
        SIDEBAR_MAX_WIDTH,
      );
    } catch {
      return SIDEBAR_DEFAULT_WIDTH;
    }
  });
  const noticeTimer = useRef(null);
  const itemsRequest = useRef(null);
  const workspaceRef = useRef(null);

  useEffect(() => {
    try {
      window.localStorage.setItem(SIDEBAR_WIDTH_KEY, String(sidebarWidth));
    } catch {
      // Storage can be unavailable in privacy-restricted browser contexts.
    }
  }, [sidebarWidth]);

  const notify = useCallback((message) => {
    setNotice(message);
    window.clearTimeout(noticeTimer.current);
    noticeTimer.current = window.setTimeout(() => setNotice(""), 4000);
  }, []);

  const installOverview = useCallback((next) => {
    setOverview(next);
    setActivePath(next.path ?? null);
    setSeriesQuery("");
    setDebouncedQuery("");
    setItems([]);
    setItemsTotal(0);
    setSeriesIndex(null);
    setSeriesPoints(0);
    setViewport([0, 0]);
    setCommittedViewport([0, 0]);
    setData(null);
    setDataError("");
    setSelectedFeatures(
      next.features.slice(0, Math.min(6, next.limits?.max_selected_features ?? 128)).map((feature) => feature.index),
    );
    setNormalization("none");
    setScope("feature");
    setLabelIndex(null);
    setTransform({ ...DEFAULT_TRANSFORM });
    setActiveTab("series");
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    api("/api/overview", {}, controller.signal)
      .then(installOverview)
      .catch((error) => {
        if (!(error instanceof ApiError && error.status === 409) && error.name !== "AbortError") {
          setDataError(error.message);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setBooting(false);
      });
    return () => controller.abort();
  }, [installOverview]);

  useEffect(() => {
    document.title = `${overview?.title ?? "AgentAD Visualizer"} · Scientific Workspace`;
  }, [overview]);

  useEffect(() => {
    const timer = window.setTimeout(() => setDebouncedQuery(seriesQuery.trim()), 220);
    return () => window.clearTimeout(timer);
  }, [seriesQuery]);

  const fetchItems = useCallback((append = false) => {
    if (!overview) return;
    itemsRequest.current?.abort();
    const controller = new AbortController();
    itemsRequest.current = controller;
    const offset = append ? items.length : 0;
    setItemsLoading(true);
    api("/api/items", { query: debouncedQuery, offset, limit: 80 }, controller.signal)
      .then((payload) => {
        setItems((current) => (append ? [...current, ...payload.items] : payload.items));
        setItemsTotal(payload.total);
      })
      .catch((error) => {
        if (error.name !== "AbortError") notify(`Series list: ${error.message}`);
      })
      .finally(() => {
        if (!controller.signal.aborted) setItemsLoading(false);
      });
  }, [debouncedQuery, items.length, notify, overview]);

  useEffect(() => {
    fetchItems(false);
  }, [debouncedQuery, overview]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectSeries = useCallback((item) => {
    setSeriesIndex(item.index);
    setSeriesPoints(item.points);
    const next = [0, item.points];
    setViewport(next);
    setCommittedViewport(next);
    setData(null);
    setDataError("");
    setSidebarOpen(false);
  }, []);

  useEffect(() => {
    if (seriesIndex == null && items.length > 0 && !debouncedQuery) {
      selectSeries(items[0]);
    }
  }, [debouncedQuery, items, selectSeries, seriesIndex]);

  useEffect(() => {
    if (!overview || seriesIndex == null || !selectedFeatures.length) return undefined;
    const controller = new AbortController();
    const timer = window.setTimeout(() => {
      const visibleSpan = Math.max(1, committedViewport[1] - committedViewport[0]);
      const padding = Math.max(1, Math.round(visibleSpan * 0.45));
      const requestStart = Math.max(0, committedViewport[0] - padding);
      const requestStop = Math.min(seriesPoints, committedViewport[1] + padding);
      const ratio = Math.max(1, (requestStop - requestStart) / visibleSpan);
      const limits = overview.limits ?? {};
      const valueBudget = Math.floor(
        (limits.max_response_values ?? 1_000_000) / selectedFeatures.length,
      );
      const maxPoints = Math.max(
        2,
        Math.min(
          limits.hard_max_points ?? 50_000,
          valueBudget,
          Math.round((limits.default_max_points ?? 4_000) * ratio),
        ),
      );
      setDataLoading(true);
      setDataError("");
      api(
        "/api/data",
        {
          series: seriesIndex,
          features: selectedFeatures.join(","),
          normalization,
          scope,
          label: labelIndex,
          start: requestStart,
          stop: requestStop,
          max_points: maxPoints,
        },
        controller.signal,
      )
        .then(setData)
        .catch((error) => {
          if (error.name !== "AbortError") setDataError(error.message);
        })
        .finally(() => {
          if (!controller.signal.aborted) setDataLoading(false);
        });
    }, 90);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [committedViewport, labelIndex, normalization, overview, scope, selectedFeatures, seriesIndex, seriesPoints]);

  const openSource = useCallback((path) => {
    setOpeningPath(path);
    setDataError("");
    api("/api/open", { path })
      .then((payload) => {
        installOverview(payload);
        notify(`Opened ${path}`);
        setSidebarOpen(false);
      })
      .catch((error) => notify(`Could not open source: ${error.message}`))
      .finally(() => setOpeningPath(null));
  }, [installOverview, notify]);

  const handleViewport = useCallback((next, commit) => {
    setViewport(next);
    if (commit) setCommittedViewport(next);
  }, []);

  const toggleFeature = useCallback((featureIndex) => {
    setSelectedFeatures((current) => {
      if (current.includes(featureIndex)) {
        if (current.length <= 1) {
          notify("Keep at least one feature track visible.");
          return current;
        }
        return current.filter((index) => index !== featureIndex);
      }
      const maximum = overview?.limits?.max_selected_features ?? 128;
      if (current.length >= maximum) {
        notify(`At most ${maximum} feature tracks can be selected.`);
        return current;
      }
      return [...current, featureIndex];
    });
  }, [notify, overview]);

  const selectedFeatureRecords = useMemo(() => {
    const records = new Map((overview?.features ?? []).map((feature) => [feature.index, feature]));
    return selectedFeatures.map((index) => records.get(index)).filter(Boolean);
  }, [overview, selectedFeatures]);

  const moveTrack = useCallback((position, direction) => {
    setSelectedFeatures((current) => {
      const nextPosition = position + direction;
      if (nextPosition < 0 || nextPosition >= current.length) return current;
      const next = [...current];
      [next[position], next[nextPosition]] = [next[nextPosition], next[position]];
      return next;
    });
  }, []);

  const zoom = (factor) => {
    const center = (viewport[0] + viewport[1]) / 2;
    handleViewport(zoomWindow(viewport, factor, center, seriesPoints), true);
  };

  return (
    <div className="app-shell">
      <Header
        overview={overview}
        activePath={activePath}
        onMenu={() => setSidebarOpen(true)}
        onInspector={() => setInspectorOpen(true)}
        onHelp={() => {
          setInspectorOpen(true);
          notify("Keyboard shortcuts are listed in the inspector.");
        }}
      />
      <div
        className="workspace"
        ref={workspaceRef}
        style={{ "--sidebar-width": `${sidebarWidth}px` }}
      >
        <Sidebar
          activeTab={activeTab}
          open={sidebarOpen}
          onTabChange={setActiveTab}
          onClose={() => setSidebarOpen(false)}
          activePath={activePath}
          openingPath={openingPath}
          onOpenSource={openSource}
          overview={overview}
          items={items}
          itemsTotal={itemsTotal}
          itemsLoading={itemsLoading}
          seriesQuery={seriesQuery}
          selectedSeries={seriesIndex}
          onSeriesQueryChange={setSeriesQuery}
          onSelectSeries={selectSeries}
          onLoadMore={() => fetchItems(true)}
          selectedFeatures={selectedFeatures}
          onToggleFeature={toggleFeature}
          onSelectDefaultFeatures={() => {
            setSelectedFeatures((overview?.features ?? []).slice(0, 6).map((feature) => feature.index));
          }}
        />
        <SidebarResizer
          width={sidebarWidth}
          workspaceRef={workspaceRef}
          onResize={setSidebarWidth}
        />
        <main className="editor" id="editor-main" tabIndex="-1">
          <Toolbar
            overview={overview}
            normalization={normalization}
            scope={scope}
            labelIndex={labelIndex}
            mode={mode}
            layout={layout}
            disabled={!data}
            onNormalization={setNormalization}
            onScope={setScope}
            onLabel={setLabelIndex}
            onMode={setMode}
            onLayout={setLayout}
            onZoom={zoom}
            onFit={() => handleViewport([0, seriesPoints], true)}
          />
          <section className="editor-surface" aria-label="Time-series tracks">
            {data ? (
              <>
                <CanvasEditor
                  data={data}
                  viewport={viewport}
                  totalPoints={seriesPoints}
                  layout={layout}
                  transform={transform}
                  mode={mode}
                  onModeChange={setMode}
                  onTransformChange={setTransform}
                  onViewportChange={handleViewport}
                />
                <TimelineNavigator viewport={viewport} totalPoints={seriesPoints} onViewportChange={handleViewport} />
              </>
            ) : (
              <EmptyState
                overview={overview}
                error={dataError}
                loading={booting || dataLoading || openingPath !== null}
                onBrowse={() => {
                  setActiveTab("sources");
                  setSidebarOpen(true);
                }}
              />
            )}
            {dataLoading && data && (
              <div className="surface-loading" role="status"><LoaderCircle className="spin" size={15} /> Updating</div>
            )}
            {dataError && data && <div className="surface-error" role="alert">{dataError}</div>}
          </section>
          <StatusBar data={data} viewport={viewport} totalPoints={seriesPoints} loading={dataLoading} notice={notice} />
        </main>
        <Inspector
          open={inspectorOpen}
          features={selectedFeatureRecords}
          transform={transform}
          overview={overview}
          data={data}
          onClose={() => setInspectorOpen(false)}
          onTransformChange={setTransform}
          onTransformReset={() => setTransform({ ...DEFAULT_TRANSFORM })}
          onMoveTrack={moveTrack}
          onRemoveTrack={toggleFeature}
          onSoloTrack={(featureIndex) => setSelectedFeatures([featureIndex])}
        />
      </div>
    </div>
  );
}

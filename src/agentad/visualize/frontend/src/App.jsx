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
  Plus,
  Rows3,
  X,
  ZoomIn,
  ZoomOut,
} from "lucide-react";

import {
  DEFAULT_VIEW_OPTIONS,
  clampPanelWidth,
  parseTabPreferences,
  serializeTabPreferences,
  zoomWindow,
} from "../../static/core.js";
import { api, ApiError } from "./api.js";
import { CanvasEditor, TimelineNavigator } from "./CanvasEditor.jsx";
import { Inspector } from "./Inspector.jsx";
import { Sidebar } from "./Sidebar.jsx";

const SIDEBAR_DEFAULT_WIDTH = 300;
const SIDEBAR_MIN_WIDTH = 240;
const SIDEBAR_MAX_WIDTH = 520;
const SIDEBAR_WIDTH_KEY = "agentad-visualizer:sidebar-width";
const INSPECTOR_DEFAULT_WIDTH = 320;
const INSPECTOR_MIN_WIDTH = 260;
const INSPECTOR_MAX_WIDTH = 560;
const LAST_TAB_PREFERENCES_KEY = "agentad-visualizer:last-tab-preferences";

const TAB_FIELDS = [
  "items",
  "itemsTotal",
  "itemsLoading",
  "seriesQuery",
  "debouncedQuery",
  "featureQuery",
  "seriesIndex",
  "seriesPoints",
  "selectedFeatures",
  "viewOptions",
  "viewport",
  "committedViewport",
  "data",
  "dataLoading",
  "dataError",
  "inspectorOpen",
  "inspectorWidth",
];

function readLastTabPreferences() {
  try {
    const stored = window.localStorage.getItem(LAST_TAB_PREFERENCES_KEY);
    const preferences = parseTabPreferences(stored);
    if (!stored) preferences.inspectorOpen = window.innerWidth > 1240;
    return preferences;
  } catch {
    const preferences = parseTabPreferences(null);
    preferences.inspectorOpen = window.innerWidth > 1240;
    return preferences;
  }
}

function createDatasetTab(overview, preferences) {
  const normalizations = overview.normalizations ?? [];
  const normalization = normalizations.some((item) => item.id === preferences.normalization)
    ? preferences.normalization
    : (normalizations[0]?.id ?? DEFAULT_VIEW_OPTIONS.normalization);
  const labels = overview.binary_labels ?? [];
  const label = preferences.labelName
    ? labels.find((item) => item.name === preferences.labelName)
    : null;
  const path = overview.path ?? null;
  const source = overview.source ?? path ?? "initial";
  const title = path?.split("/").at(-1) ?? overview.title ?? "In-memory collection";
  return {
    source,
    path,
    title,
    overview,
    items: [],
    itemsTotal: 0,
    itemsLoading: false,
    seriesQuery: "",
    debouncedQuery: "",
    featureQuery: "",
    seriesIndex: null,
    seriesPoints: 0,
    selectedFeatures: overview.features
      .slice(0, Math.min(6, overview.limits?.max_selected_features ?? 128))
      .map((feature) => feature.index),
    viewOptions: {
      normalization,
      scope: preferences.scope,
      labelIndex: label?.index ?? null,
      mode: preferences.mode,
      layout: preferences.layout,
      transform: { ...preferences.transform },
    },
    viewport: [0, 0],
    committedViewport: [0, 0],
    data: null,
    dataLoading: false,
    dataError: "",
    inspectorOpen: preferences.inspectorOpen,
    inspectorWidth: preferences.inspectorWidth,
  };
}

function preferencesFromTab(tab) {
  const labelName = (tab.overview.binary_labels ?? []).find(
    (label) => label.index === tab.viewOptions.labelIndex,
  )?.name ?? null;
  return {
    ...tab.viewOptions,
    labelName,
    inspectorOpen: tab.inspectorOpen,
    inspectorWidth: tab.inspectorWidth,
  };
}

function useDatasetTabs() {
  const [tabs, setTabs] = useState([]);
  const [activeSource, setActiveSource] = useState(null);
  const activeDataset = useMemo(
    () => tabs.find((tab) => tab.source === activeSource) ?? null,
    [activeSource, tabs],
  );
  const lastPreferencesRef = useRef(readLastTabPreferences());
  if (activeDataset) lastPreferencesRef.current = preferencesFromTab(activeDataset);

  const openTab = useCallback((overview) => {
    const source = overview.source ?? overview.path ?? "initial";
    setTabs((current) =>
      current.some((tab) => tab.source === source)
        ? current
        : [...current, createDatasetTab(overview, lastPreferencesRef.current)],
    );
    setActiveSource(source);
  }, []);

  const setters = useMemo(() => {
    const entries = TAB_FIELDS.map((field) => [
      `set${field[0].toUpperCase()}${field.slice(1)}`,
      (value) => {
        const source = activeSource;
        if (source == null) return;
        setTabs((current) =>
          current.map((tab) => {
            if (tab.source !== source) return tab;
            const nextValue = typeof value === "function" ? value(tab[field]) : value;
            return Object.is(nextValue, tab[field]) ? tab : { ...tab, [field]: nextValue };
          }),
        );
      },
    ]);
    return Object.fromEntries(entries);
  }, [activeSource]);

  const closeTab = useCallback((source) => {
    const index = tabs.findIndex((tab) => tab.source === source);
    if (index < 0) return;
    const remaining = tabs.filter((tab) => tab.source !== source);
    setTabs(remaining);
    if (activeSource === source) {
      setActiveSource(remaining[Math.min(index, remaining.length - 1)]?.source ?? null);
    }
  }, [activeSource, tabs]);

  useEffect(() => {
    if (!activeDataset) return undefined;
    const preferences = preferencesFromTab(activeDataset);
    const timer = window.setTimeout(() => {
      try {
        window.localStorage.setItem(
          LAST_TAB_PREFERENCES_KEY,
          serializeTabPreferences(preferences),
        );
      } catch {
        // Storage can be unavailable in privacy-restricted browser contexts.
      }
    }, 160);
    return () => window.clearTimeout(timer);
  }, [activeDataset]);

  useEffect(() => {
    const flush = () => {
      try {
        window.localStorage.setItem(
          LAST_TAB_PREFERENCES_KEY,
          serializeTabPreferences(lastPreferencesRef.current),
        );
      } catch {
        // Storage can be unavailable in privacy-restricted browser contexts.
      }
    };
    window.addEventListener("pagehide", flush);
    return () => window.removeEventListener("pagehide", flush);
  }, []);

  return {
    tabs,
    activeSource,
    activeDataset,
    setActiveSource,
    openTab,
    closeTab,
    ...setters,
  };
}

function DatasetTabs({ tabs, activeSource, onSelect, onClose, onNew }) {
  return (
    <div className="dataset-tabs" role="tablist" aria-label="Open datasets">
      <div className="dataset-tab-list">
        {tabs.map((tab) => (
          <div
            className={`dataset-tab ${tab.source === activeSource ? "is-active" : ""}`}
            role="tab"
            aria-selected={tab.source === activeSource}
            tabIndex={tab.source === activeSource ? 0 : -1}
            title={tab.path ?? tab.title}
            key={tab.source}
            onClick={() => onSelect(tab.source)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                onSelect(tab.source);
              }
            }}
          >
            <span>{tab.title}</span>
            <button
              type="button"
              aria-label={`Close ${tab.title}`}
              title="Close dataset"
              onClick={(event) => {
                event.stopPropagation();
                onClose(tab.source);
              }}
            >
              <X size={12} aria-hidden="true" />
            </button>
          </div>
        ))}
      </div>
      <button type="button" className="dataset-tab-new" onClick={onNew} title="Open dataset">
        <Plus size={14} aria-hidden="true" />
        <span>Open</span>
      </button>
    </div>
  );
}

function Header({ overview, activePath, inspectorOpen, onMenu, onInspector, onHelp }) {
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
        <button
          type="button"
          className={inspectorOpen ? "is-active" : ""}
          aria-label={inspectorOpen ? "Close inspector" : "Open inspector"}
          aria-pressed={inspectorOpen}
          title={inspectorOpen ? "Close inspector" : "Open inspector"}
          onClick={onInspector}
        >
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

function InspectorResizer({ width, workspaceRef, onResize }) {
  const drag = useRef(null);

  const preview = (nextWidth, element) => {
    const next = clampPanelWidth(nextWidth, INSPECTOR_MIN_WIDTH, INSPECTOR_MAX_WIDTH);
    if (drag.current) drag.current.lastWidth = next;
    workspaceRef.current?.style.setProperty("--inspector-width", `${next}px`);
    element.setAttribute("aria-valuenow", String(next));
  };

  const finish = (event) => {
    if (!drag.current) return;
    const next = drag.current.lastWidth;
    drag.current = null;
    document.body.classList.remove("is-resizing-inspector");
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    onResize(next);
  };

  return (
    <div
      className="inspector-resizer"
      role="separator"
      tabIndex="0"
      aria-label="Resize track inspector"
      aria-controls="track-inspector"
      aria-orientation="vertical"
      aria-valuemin={INSPECTOR_MIN_WIDTH}
      aria-valuemax={INSPECTOR_MAX_WIDTH}
      aria-valuenow={width}
      title="Drag to resize; double-click to reset"
      onDoubleClick={(event) => {
        preview(INSPECTOR_DEFAULT_WIDTH, event.currentTarget);
        onResize(INSPECTOR_DEFAULT_WIDTH);
      }}
      onKeyDown={(event) => {
        let next = null;
        const step = event.shiftKey ? 48 : 16;
        if (event.key === "ArrowLeft") next = width + step;
        else if (event.key === "ArrowRight") next = width - step;
        else if (event.key === "Home") next = INSPECTOR_DEFAULT_WIDTH;
        if (next === null) return;
        event.preventDefault();
        onResize(clampPanelWidth(next, INSPECTOR_MIN_WIDTH, INSPECTOR_MAX_WIDTH));
      }}
      onPointerCancel={finish}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        drag.current = { lastWidth: width };
        event.currentTarget.setPointerCapture(event.pointerId);
        document.body.classList.add("is-resizing-inspector");
      }}
      onPointerMove={(event) => {
        if (!drag.current || !workspaceRef.current) return;
        const bounds = workspaceRef.current.getBoundingClientRect();
        preview(bounds.right - event.clientX, event.currentTarget);
      }}
      onPointerUp={finish}
    >
      <span aria-hidden="true" />
    </div>
  );
}

function Toolbar({
  mode,
  layout,
  disabled,
  onMode,
  onLayout,
  onZoom,
  onFit,
}) {
  return (
    <div className="editor-toolbar" aria-label="Visualization controls">
      <div className="toolbar-primary">
        <div className="toolbar-group">
          <span className="toolbar-caption">Tool</span>
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
          <span className="toolbar-caption">View</span>
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
        <div className="toolbar-group">
          <span className="toolbar-caption">Layout</span>
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
  const tabState = useDatasetTabs();
  const {
    tabs,
    activeSource,
    activeDataset,
    setActiveSource,
    openTab,
    closeTab,
    setItems,
    setItemsTotal,
    setItemsLoading,
    setSeriesQuery,
    setDebouncedQuery,
    setSeriesIndex,
    setSeriesPoints,
    setSelectedFeatures,
    setViewOptions,
    setViewport,
    setCommittedViewport,
    setData,
    setDataLoading,
    setDataError,
    setInspectorOpen,
    setInspectorWidth,
    setFeatureQuery,
  } = tabState;
  const overview = activeDataset?.overview ?? null;
  const activePath = activeDataset?.path ?? null;
  const items = activeDataset?.items ?? [];
  const itemsTotal = activeDataset?.itemsTotal ?? 0;
  const itemsLoading = activeDataset?.itemsLoading ?? false;
  const seriesQuery = activeDataset?.seriesQuery ?? "";
  const debouncedQuery = activeDataset?.debouncedQuery ?? "";
  const featureQuery = activeDataset?.featureQuery ?? "";
  const seriesIndex = activeDataset?.seriesIndex ?? null;
  const seriesPoints = activeDataset?.seriesPoints ?? 0;
  const selectedFeatures = activeDataset?.selectedFeatures ?? [];
  const viewOptions = activeDataset?.viewOptions ?? DEFAULT_VIEW_OPTIONS;
  const { normalization, scope, labelIndex, mode, layout, transform } = viewOptions;
  const viewport = activeDataset?.viewport ?? [0, 0];
  const committedViewport = activeDataset?.committedViewport ?? [0, 0];
  const data = activeDataset?.data ?? null;
  const dataLoading = activeDataset?.dataLoading ?? false;
  const dataError = activeDataset?.dataError ?? "";
  const inspectorOpen = activeDataset?.inspectorOpen ?? false;
  const inspectorWidth = activeDataset?.inspectorWidth ?? INSPECTOR_DEFAULT_WIDTH;
  const [openingPath, setOpeningPath] = useState(null);
  const [activeTab, setActiveTab] = useState("sources");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [booting, setBooting] = useState(true);
  const [bootError, setBootError] = useState("");
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
  const workspaceRef = useRef(null);

  const setNormalization = useCallback((value) => {
    setViewOptions((current) => ({ ...current, normalization: value }));
  }, [setViewOptions]);
  const setScope = useCallback((value) => {
    setViewOptions((current) => ({ ...current, scope: value }));
  }, [setViewOptions]);
  const setMode = useCallback((value) => {
    setViewOptions((current) => ({ ...current, mode: value }));
  }, [setViewOptions]);
  const setLayout = useCallback((value) => {
    setViewOptions((current) => ({ ...current, layout: value }));
  }, [setViewOptions]);
  const setTransform = useCallback((value) => {
    setViewOptions((current) => ({
      ...current,
      transform: typeof value === "function" ? value(current.transform) : value,
    }));
  }, [setViewOptions]);
  const setLabelIndex = useCallback((value) => {
    const label = (overview?.binary_labels ?? []).find((item) => item.index === value);
    setViewOptions((current) => ({
      ...current,
      labelIndex: label?.index ?? null,
    }));
  }, [overview, setViewOptions]);

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
    setBootError("");
    openTab(next);
    setActiveTab("series");
  }, [openTab]);

  useEffect(() => {
    const controller = new AbortController();
    api("/api/overview", {}, controller.signal)
      .then(installOverview)
      .catch((error) => {
        if (!(error instanceof ApiError && error.status === 409) && error.name !== "AbortError") {
          setBootError(error.message);
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
  }, [seriesQuery, setDebouncedQuery]);

  const fetchItems = useCallback((append = false) => {
    if (!overview || !activeSource) return null;
    const controller = new AbortController();
    const offset = append ? items.length : 0;
    setItemsLoading(true);
    api(
      "/api/items",
      { source: activeSource, query: debouncedQuery, offset, limit: 80 },
      controller.signal,
    )
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
    return controller;
  }, [activeSource, debouncedQuery, items.length, notify, overview, setItems, setItemsLoading, setItemsTotal]);

  useEffect(() => {
    const controller = fetchItems(false);
    return () => {
      controller?.abort();
      setItemsLoading(false);
    };
  }, [activeSource, debouncedQuery]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectSeries = useCallback((item) => {
    setSeriesIndex(item.index);
    setSeriesPoints(item.points);
    const next = [0, item.points];
    setViewport(next);
    setCommittedViewport(next);
    setData(null);
    setDataError("");
    setSidebarOpen(false);
  }, [setCommittedViewport, setData, setDataError, setSeriesIndex, setSeriesPoints, setViewport]);

  useEffect(() => {
    if (seriesIndex == null && items.length > 0 && !debouncedQuery) {
      selectSeries(items[0]);
    }
  }, [debouncedQuery, items, selectSeries, seriesIndex]);

  useEffect(() => {
    if (!overview || !activeSource || seriesIndex == null || !selectedFeatures.length) {
      return undefined;
    }
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
          source: activeSource,
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
      setDataLoading(false);
    };
  }, [activeSource, committedViewport, labelIndex, normalization, overview, scope, selectedFeatures, seriesIndex, seriesPoints, setData, setDataError, setDataLoading]);

  const openSource = useCallback((path) => {
    setOpeningPath(path);
    api("/api/open", { path })
      .then((payload) => {
        installOverview(payload);
        notify(`Opened ${path}`);
        setSidebarOpen(false);
      })
      .catch((error) => notify(`Could not open source: ${error.message}`))
      .finally(() => setOpeningPath(null));
  }, [installOverview, notify]);

  const closeDataset = useCallback((source) => {
    closeTab(source);
    api("/api/close", { source }).catch((error) => {
      if (!(error instanceof ApiError && error.status === 409)) {
        notify(`Could not release dataset: ${error.message}`);
      }
    });
  }, [closeTab, notify]);

  const handleViewport = useCallback((next, commit) => {
    setViewport(next);
    if (commit) setCommittedViewport(next);
  }, [setCommittedViewport, setViewport]);

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
  }, [notify, overview, setSelectedFeatures]);

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
  }, [setSelectedFeatures]);

  const zoom = (factor) => {
    const center = (viewport[0] + viewport[1]) / 2;
    handleViewport(zoomWindow(viewport, factor, center, seriesPoints), true);
  };

  return (
    <div className="app-shell">
      <Header
        overview={overview}
        activePath={activePath}
        inspectorOpen={inspectorOpen}
        onMenu={() => setSidebarOpen(true)}
        onInspector={() => setInspectorOpen((current) => !current)}
        onHelp={() => {
          setInspectorOpen(true);
          notify("Interaction controls are listed in the inspector.");
        }}
      />
      <div
        className={`workspace ${inspectorOpen ? "has-inspector" : ""}`}
        ref={workspaceRef}
        style={{
          "--sidebar-width": `${sidebarWidth}px`,
          "--inspector-width": `${inspectorWidth}px`,
        }}
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
          featureQuery={featureQuery}
          onFeatureQueryChange={setFeatureQuery}
        />
        <SidebarResizer
          width={sidebarWidth}
          workspaceRef={workspaceRef}
          onResize={setSidebarWidth}
        />
        <main className="editor" id="editor-main" tabIndex="-1">
          <DatasetTabs
            tabs={tabs}
            activeSource={activeSource}
            onSelect={setActiveSource}
            onClose={closeDataset}
            onNew={() => {
              setActiveTab("sources");
              setSidebarOpen(true);
            }}
          />
          <Toolbar
            mode={mode}
            layout={layout}
            disabled={!data}
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
                error={dataError || bootError}
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
        {inspectorOpen && (
          <InspectorResizer
            width={inspectorWidth}
            workspaceRef={workspaceRef}
            onResize={setInspectorWidth}
          />
        )}
        <Inspector
          open={inspectorOpen}
          normalization={normalization}
          scope={scope}
          labelIndex={labelIndex}
          features={selectedFeatureRecords}
          transform={transform}
          overview={overview}
          data={data}
          controlsDisabled={!data}
          onClose={() => setInspectorOpen(false)}
          onNormalizationChange={setNormalization}
          onScopeChange={setScope}
          onLabelChange={setLabelIndex}
          onTransformChange={setTransform}
          onTransformReset={() => setTransform({ ...DEFAULT_VIEW_OPTIONS.transform })}
          onMoveTrack={moveTrack}
          onRemoveTrack={toggleFeature}
          onSoloTrack={(featureIndex) => setSelectedFeatures([featureIndex])}
        />
      </div>
    </div>
  );
}

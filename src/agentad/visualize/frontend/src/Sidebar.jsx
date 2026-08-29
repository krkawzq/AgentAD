import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Check,
  ChevronRight,
  Database,
  File,
  FileArchive,
  FileSpreadsheet,
  Folder,
  FolderOpen,
  Layers3,
  LoaderCircle,
  Search,
} from "lucide-react";
import Tree from "@rc-component/tree";

import { replaceTreeChildren } from "../../static/core.js";
import { api } from "./api.js";
import { TRACK_COLORS } from "./renderer.js";

const TREE_MOTION = Object.freeze({
  motionName: "source-tree-motion",
  motionDeadline: 280,
  onAppearStart: () => ({ height: 0, opacity: 0 }),
  onAppearActive: (element) => ({ height: element.scrollHeight, opacity: 1 }),
  onLeaveStart: (element) => ({ height: element.offsetHeight, opacity: 1 }),
  onLeaveActive: () => ({ height: 0, opacity: 0 }),
});

function useReducedMotion() {
  const [reduced, setReduced] = useState(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
  );
  useEffect(() => {
    const query = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!query) return undefined;
    const update = () => setReduced(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return reduced;
}

function SidebarTabs({ active, onChange }) {
  const tabs = [
    ["sources", "Sources", Database],
    ["series", "Series", Layers3],
    ["features", "Features", Check],
  ];
  return (
    <div className="sidebar-tabs" role="tablist" aria-label="Dataset navigation">
      {tabs.map(([id, label, Icon]) => (
        <button
          type="button"
          role="tab"
          aria-selected={active === id}
          className="sidebar-tab"
          key={id}
          onClick={() => onChange(id)}
        >
          <Icon size={15} aria-hidden="true" />
          {label}
        </button>
      ))}
    </div>
  );
}

function toTreeNode(entry) {
  return {
    ...entry,
    key: entry.path,
    title: entry.name,
    isLeaf: !entry.dir || entry.loadable,
    disabled: !entry.dir && !entry.loadable,
  };
}

const SourceTreeTitle = memo(function SourceTreeTitle({ node, opening }) {
  return (
    <span className="source-tree-title" title={node.path}>
      <span>{node.name}</span>
      {opening && <LoaderCircle className="spin" size={12} aria-label="Opening" />}
    </span>
  );
});

function treeIcon(props) {
  const node = props.data ?? props;
  if (node.dir) {
    const Icon = props.expanded ? FolderOpen : Folder;
    return <Icon size={15} aria-hidden="true" />;
  }
  const Icon = node.loadable
    ? node.format === "csv"
      ? FileSpreadsheet
      : FileArchive
    : File;
  return <Icon size={15} aria-hidden="true" />;
}

function switcherIcon(props) {
  if (props.loading) return <LoaderCircle className="spin" size={13} aria-hidden="true" />;
  if (props.isLeaf) return <span className="tree-switcher-spacer" />;
  return (
    <ChevronRight
      className={`tree-chevron ${props.expanded ? "is-expanded" : ""}`}
      size={13}
      aria-hidden="true"
    />
  );
}

function SourceBrowser({ activePath, openingPath, onOpen, hidden }) {
  const treeHost = useRef(null);
  const [treeData, setTreeData] = useState(null);
  const [treeHeight, setTreeHeight] = useState(320);
  const [expandedKeys, setExpandedKeys] = useState([]);
  const [root, setRoot] = useState("");
  const [truncatedPaths, setTruncatedPaths] = useState([]);
  const [error, setError] = useState("");
  const reducedMotion = useReducedMotion();

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    api("/api/tree", {}, controller.signal)
      .then((listing) => {
        setRoot(listing.root);
        setTreeData(listing.entries.map(toTreeNode));
        setTruncatedPaths(listing.truncated ? ["Root"] : []);
      })
      .catch((requestError) => {
        if (requestError.name !== "AbortError") setError(requestError.message);
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const element = treeHost.current;
    if (!element) return undefined;
    const update = () => {
      if (element.clientHeight <= 0) return;
      const next = Math.max(120, element.clientHeight);
      setTreeHeight((current) => (current === next ? current : next));
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

  const loadData = useCallback(async (node) => {
    if (node.isLeaf || Array.isArray(node.children)) return;
    setError("");
    try {
      const listing = await api("/api/tree", { path: node.key });
      setTreeData((current) =>
        replaceTreeChildren(current ?? [], node.key, listing.entries.map(toTreeNode)),
      );
      if (listing.truncated) {
        setTruncatedPaths((current) =>
          current.includes(node.key) ? current : [...current, node.key],
        );
      }
      setError("");
    } catch (requestError) {
      setError(requestError.message);
      throw requestError;
    }
  }, []);

  const renderTitle = useCallback(
    (node) => <SourceTreeTitle node={node} opening={node.key === openingPath} />,
    [openingPath],
  );
  const handleSelect = useCallback((_, info) => {
    if (info.node.loadable) onOpen(String(info.node.key));
  }, [onOpen]);

  return (
    <div className="sidebar-panel source-panel" role="tabpanel" hidden={hidden}>
      <div className="panel-heading">
        <div>
          <span className="eyebrow">LOCAL WORKSPACE</span>
          <h2>Data sources</h2>
        </div>
      </div>
      <div className="tree-root-path" title={root}>{root || "Workspace root"}</div>
      <div className="source-tree-host" ref={treeHost} aria-busy={!treeData && !error}>
        {error && <div className="panel-error" role="alert">{error}</div>}
        {!error && treeData === null && <div className="panel-empty">Reading directory…</div>}
        {treeData?.length === 0 && <div className="panel-empty">This directory is empty.</div>}
        {treeData?.length > 0 && (
          <Tree
            className="source-tree"
            treeData={treeData}
            height={treeHeight}
            itemHeight={36}
            virtual
            motion={reducedMotion ? null : TREE_MOTION}
            showIcon
            icon={treeIcon}
            switcherIcon={switcherIcon}
            expandAction="click"
            loadData={loadData}
            expandedKeys={expandedKeys}
            onExpand={setExpandedKeys}
            selectedKeys={activePath ? [activePath] : []}
            titleRender={renderTitle}
            onSelect={handleSelect}
          />
        )}
      </div>
      {truncatedPaths.length > 0 && (
        <div className="tree-limit-note" role="status">
          Entry limit reached in {truncatedPaths.length} director{truncatedPaths.length === 1 ? "y" : "ies"}.
        </div>
      )}
    </div>
  );
}

function SeriesBrowser({
  overview,
  items,
  itemsTotal,
  itemsLoading,
  query,
  selectedSeries,
  onQueryChange,
  onSelect,
  onLoadMore,
}) {
  return (
    <div className="sidebar-panel" role="tabpanel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">COLLECTION</span>
          <h2>Series</h2>
        </div>
        <span className="count-badge">{overview?.series_count?.toLocaleString() ?? 0}</span>
      </div>
      <label className="search-box">
        <Search size={15} aria-hidden="true" />
        <span className="sr-only">Search series IDs</span>
        <input
          type="search"
          value={query}
          onChange={(event) => onQueryChange(event.target.value)}
          placeholder="Search series ID"
          autoComplete="off"
          spellCheck="false"
        />
      </label>
      <div className="series-list" role="listbox" aria-busy={itemsLoading}>
        {items.map((item) => (
          <button
            type="button"
            role="option"
            aria-selected={selectedSeries === item.index}
            className={`series-row ${selectedSeries === item.index ? "is-active" : ""}`}
            key={item.index}
            onClick={() => onSelect(item)}
          >
            <span className="series-index">{String(item.index + 1).padStart(3, "0")}</span>
            <span>
              <strong>{item.id}</strong>
              <small>{item.points.toLocaleString()} points</small>
            </span>
          </button>
        ))}
        {!itemsLoading && !items.length && (
          <div className="panel-empty">
            {overview ? "No matching series." : "Open a data source first."}
          </div>
        )}
      </div>
      {items.length < itemsTotal && (
        <button type="button" className="load-more" onClick={onLoadMore} disabled={itemsLoading}>
          {itemsLoading ? "Loading…" : `Load more (${items.length}/${itemsTotal})`}
        </button>
      )}
    </div>
  );
}

function FeatureBrowser({ overview, selected, onToggle, onSelectDefaults }) {
  const [query, setQuery] = useState("");
  const matches = useMemo(() => {
    const needle = query.trim().toLocaleLowerCase();
    const features = overview?.features ?? [];
    if (!needle) return features;
    return features.filter((feature) =>
      `${feature.name} ${Object.values(feature.metadata).join(" ")}`
        .toLocaleLowerCase()
        .includes(needle),
    );
  }, [overview, query]);
  const visible = matches.slice(0, 300);

  return (
    <div className="sidebar-panel" role="tabpanel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">TRACK LIBRARY</span>
          <h2>Features</h2>
        </div>
        <span className="count-badge">{selected.length} selected</span>
      </div>
      <label className="search-box">
        <Search size={15} aria-hidden="true" />
        <span className="sr-only">Search features</span>
        <input
          type="search"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search feature or metadata"
          autoComplete="off"
          spellCheck="false"
        />
      </label>
      <div className="feature-actions">
        <button type="button" onClick={onSelectDefaults}>Select first tracks</button>
        <span>{matches.length.toLocaleString()} matches</span>
      </div>
      <div className="feature-list">
        {visible.map((feature) => {
          const active = selected.includes(feature.index);
          const metadata = Object.entries(feature.metadata)
            .map(([name, value]) => `${name}: ${value}`)
            .join(" · ");
          return (
            <label className={`feature-row ${active ? "is-active" : ""}`} key={feature.index}>
              <input
                type="checkbox"
                checked={active}
                onChange={() => onToggle(feature.index)}
              />
              <span
                className="track-swatch"
                style={{ "--track-color": TRACK_COLORS[feature.index % TRACK_COLORS.length] }}
              />
              <span>
                <strong>{feature.name}</strong>
                <small title={metadata}>{metadata || `Feature ${feature.index + 1}`}</small>
              </span>
            </label>
          );
        })}
        {!visible.length && <div className="panel-empty">No matching features.</div>}
        {matches.length > visible.length && (
          <div className="panel-note">
            Refine the search to view the remaining {matches.length - visible.length} features.
          </div>
        )}
      </div>
    </div>
  );
}

export function Sidebar({
  activeTab,
  open,
  onTabChange,
  onClose,
  activePath,
  openingPath,
  onOpenSource,
  overview,
  items,
  itemsTotal,
  itemsLoading,
  seriesQuery,
  selectedSeries,
  onSeriesQueryChange,
  onSelectSeries,
  onLoadMore,
  selectedFeatures,
  onToggleFeature,
  onSelectDefaultFeatures,
}) {
  return (
    <>
      <aside
        id="data-sidebar"
        className={`sidebar ${open ? "is-open" : ""}`}
        aria-label="Data navigation"
      >
        <SidebarTabs active={activeTab} onChange={onTabChange} />
        <SourceBrowser
          activePath={activePath}
          openingPath={openingPath}
          onOpen={onOpenSource}
          hidden={activeTab !== "sources"}
        />
        {activeTab === "series" && (
          <SeriesBrowser
            overview={overview}
            items={items}
            itemsTotal={itemsTotal}
            itemsLoading={itemsLoading}
            query={seriesQuery}
            selectedSeries={selectedSeries}
            onQueryChange={onSeriesQueryChange}
            onSelect={onSelectSeries}
            onLoadMore={onLoadMore}
          />
        )}
        {activeTab === "features" && (
          <FeatureBrowser
            overview={overview}
            selected={selectedFeatures}
            onToggle={onToggleFeature}
            onSelectDefaults={onSelectDefaultFeatures}
          />
        )}
      </aside>
      {open && <button type="button" className="sidebar-backdrop" aria-label="Close navigation" onClick={onClose} />}
    </>
  );
}

import { useEffect, useMemo, useState } from "react";
import {
  ArrowUp,
  Check,
  ChevronRight,
  Database,
  FileArchive,
  FileSpreadsheet,
  Folder,
  Layers3,
  Search,
} from "lucide-react";

import { api } from "./api.js";
import { TRACK_COLORS } from "./renderer.js";

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

function Breadcrumbs({ path, onChange }) {
  const parts = path ? path.split("/") : [];
  return (
    <nav className="breadcrumbs" aria-label="Source directory">
      <button type="button" onClick={() => onChange("")} title="Dataset root">
        Root
      </button>
      {parts.map((part, index) => {
        const target = parts.slice(0, index + 1).join("/");
        return (
          <span key={target}>
            <ChevronRight size={12} aria-hidden="true" />
            <button type="button" onClick={() => onChange(target)} title={target}>
              {part}
            </button>
          </span>
        );
      })}
    </nav>
  );
}

function SourceBrowser({ activePath, openingPath, onOpen }) {
  const [path, setPath] = useState("");
  const [listing, setListing] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    setError("");
    api("/api/tree", { path }, controller.signal)
      .then(setListing)
      .catch((requestError) => {
        if (requestError.name !== "AbortError") setError(requestError.message);
      });
    return () => controller.abort();
  }, [path]);

  const parent = path.split("/").slice(0, -1).join("/");
  return (
    <div className="sidebar-panel" role="tabpanel">
      <div className="panel-heading">
        <div>
          <span className="eyebrow">LOCAL WORKSPACE</span>
          <h2>Data sources</h2>
        </div>
      </div>
      <Breadcrumbs path={path} onChange={setPath} />
      <div className="source-list" aria-busy={!listing && !error}>
        {path && (
          <button type="button" className="source-row" onClick={() => setPath(parent)}>
            <span className="source-icon"><ArrowUp size={16} /></span>
            <span><strong>Parent directory</strong><small>Move one level up</small></span>
          </button>
        )}
        {error && <div className="panel-error" role="alert">{error}</div>}
        {!error && !listing && <div className="panel-empty">Reading directory…</div>}
        {listing?.entries.map((entry) => {
          const Icon = entry.dir
            ? Folder
            : entry.format === "csv"
              ? FileSpreadsheet
              : FileArchive;
          const isActive = entry.path === activePath;
          const busy = entry.path === openingPath;
          return (
            <button
              type="button"
              className={`source-row ${isActive ? "is-active" : ""}`}
              key={entry.path}
              disabled={!entry.dir && !entry.loadable}
              onClick={() => (entry.dir ? setPath(entry.path) : onOpen(entry.path))}
              title={entry.path}
            >
              <span className="source-icon"><Icon size={16} /></span>
              <span>
                <strong>{entry.name}</strong>
                <small>
                  {busy
                    ? "Opening…"
                    : entry.dir
                      ? "Directory"
                      : entry.loadable
                        ? entry.format === "csv"
                          ? "CSV contract v1"
                          : "SeriesData package"
                        : "Unsupported file"}
                </small>
              </span>
              {entry.loadable && <span className="source-format">{entry.format}</span>}
            </button>
          );
        })}
        {listing?.entries.length === 0 && (
          <div className="panel-empty">This directory is empty.</div>
        )}
        {listing?.truncated && (
          <div className="panel-note">Only the first 5,000 entries are shown.</div>
        )}
      </div>
      <div className="contract-note">
        CSV: <code>series_id</code>, <code>timestamp</code>, and one or more{" "}
        <code>feature.*</code> columns.
      </div>
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
      <aside className={`sidebar ${open ? "is-open" : ""}`} aria-label="Data navigation">
        <SidebarTabs active={activeTab} onChange={onTabChange} />
        {activeTab === "sources" && (
          <SourceBrowser activePath={activePath} openingPath={openingPath} onOpen={onOpenSource} />
        )}
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

import {
  ArrowDown,
  ArrowUp,
  FlipHorizontal2,
  FlipVertical2,
  Focus,
  RotateCcw,
  X,
} from "lucide-react";

import { TRACK_COLORS } from "./renderer.js";

function NumberControl({ label, value, min, max, step, suffix = "", onChange }) {
  return (
    <label className="number-control">
      <span>{label}</span>
      <div>
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(event) => onChange(Number(event.target.value))}
        />
        <output>{Number(value).toFixed(step < 1 ? 2 : 0)}{suffix}</output>
      </div>
    </label>
  );
}

function TransformPanel({ transform, onChange, onReset }) {
  return (
    <section className="inspector-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">2D GEOMETRY</span>
          <h2>Transform</h2>
        </div>
        <button type="button" className="text-icon-button" onClick={onReset} title="Reset transform">
          <RotateCcw size={14} aria-hidden="true" /> Reset
        </button>
      </div>
      <p className="section-copy">
        Each feature is rendered from an interleaved 2D point buffer before the affine transform is applied.
      </p>
      <NumberControl
        label="Rotation"
        value={transform.rotation}
        min={-180}
        max={180}
        step={1}
        suffix="°"
        onChange={(rotation) => onChange({ ...transform, rotation })}
      />
      <NumberControl
        label="Horizontal scale"
        value={transform.scaleX}
        min={0.25}
        max={2}
        step={0.05}
        onChange={(scaleX) => onChange({ ...transform, scaleX })}
      />
      <NumberControl
        label="Vertical scale"
        value={transform.scaleY}
        min={0.25}
        max={4}
        step={0.05}
        onChange={(scaleY) => onChange({ ...transform, scaleY })}
      />
      <div className="segmented full-width" aria-label="Axis reflection">
        <button
          type="button"
          className={transform.flipX ? "is-active" : ""}
          aria-pressed={transform.flipX}
          onClick={() => onChange({ ...transform, flipX: !transform.flipX })}
        >
          <FlipHorizontal2 size={14} /> Flip X
        </button>
        <button
          type="button"
          className={transform.flipY ? "is-active" : ""}
          aria-pressed={transform.flipY}
          onClick={() => onChange({ ...transform, flipY: !transform.flipY })}
        >
          <FlipVertical2 size={14} /> Flip Y
        </button>
      </div>
    </section>
  );
}

function TracksPanel({ features, onMove, onRemove, onSolo }) {
  return (
    <section className="inspector-section tracks-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">TIMELINE STACK</span>
          <h2>Tracks</h2>
        </div>
        <span className="count-badge">{features.length}</span>
      </div>
      <div className="track-stack">
        {features.map((feature, position) => (
          <div className="track-card" key={feature.index}>
            <span
              className="track-number"
              style={{ "--track-color": TRACK_COLORS[feature.index % TRACK_COLORS.length] }}
            >
              {String(position + 1).padStart(2, "0")}
            </span>
            <span className="track-name" title={feature.name}>{feature.name}</span>
            <span className="track-actions">
              <button
                type="button"
                aria-label={`Solo ${feature.name}`}
                title="Solo track"
                onClick={() => onSolo(feature.index)}
              >
                <Focus size={13} />
              </button>
              <button
                type="button"
                aria-label={`Move ${feature.name} up`}
                title="Move up"
                disabled={position === 0}
                onClick={() => onMove(position, -1)}
              >
                <ArrowUp size={13} />
              </button>
              <button
                type="button"
                aria-label={`Move ${feature.name} down`}
                title="Move down"
                disabled={position === features.length - 1}
                onClick={() => onMove(position, 1)}
              >
                <ArrowDown size={13} />
              </button>
              <button
                type="button"
                aria-label={`Remove ${feature.name}`}
                title="Remove track"
                disabled={features.length <= 1}
                onClick={() => onRemove(feature.index)}
              >
                <X size={13} />
              </button>
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function DetailsPanel({ overview, data }) {
  const details = [
    ["Series", data?.series.id ?? "None"],
    ["Points", data?.series.points?.toLocaleString() ?? "0"],
    ["Sampled", data?.sampled_points?.toLocaleString() ?? "0"],
    ["Format", overview?.manifest?.format ?? "SeriesData"],
  ];
  return (
    <section className="inspector-section">
      <div className="section-heading">
        <div>
          <span className="eyebrow">SELECTION</span>
          <h2>Details</h2>
        </div>
      </div>
      <dl className="detail-grid">
        {details.map(([label, value]) => (
          <div key={label}><dt>{label}</dt><dd>{value}</dd></div>
        ))}
      </dl>
      {data?.series.meta && Object.keys(data.series.meta).length > 0 && (
        <details className="metadata-details">
          <summary>Series metadata</summary>
          <pre>{JSON.stringify(data.series.meta, null, 2)}</pre>
        </details>
      )}
    </section>
  );
}

export function Inspector({
  open,
  features,
  transform,
  overview,
  data,
  onClose,
  onTransformChange,
  onTransformReset,
  onMoveTrack,
  onRemoveTrack,
  onSoloTrack,
}) {
  return (
    <aside className={`inspector ${open ? "is-open" : ""}`} aria-label="Track inspector">
      <div className="inspector-mobile-head">
        <strong>Inspector</strong>
        <button type="button" aria-label="Close inspector" onClick={onClose}><X size={16} /></button>
      </div>
      <TransformPanel transform={transform} onChange={onTransformChange} onReset={onTransformReset} />
      <TracksPanel features={features} onMove={onMoveTrack} onRemove={onRemoveTrack} onSolo={onSoloTrack} />
      <DetailsPanel overview={overview} data={data} />
      <section className="inspector-section shortcut-section">
        <span className="eyebrow">KEYBOARD</span>
        <dl className="shortcut-grid">
          <div><dt><kbd>Z</kbd></dt><dd>Box zoom</dd></div>
          <div><dt><kbd>V</kbd></dt><dd>Pan</dd></div>
          <div><dt><kbd>Space</kbd></dt><dd>Temporary pan</dd></div>
          <div><dt><kbd>+</kbd> <kbd>−</kbd></dt><dd>Zoom</dd></div>
          <div><dt><kbd>←</kbd> <kbd>→</kbd></dt><dd>Move timeline</dd></div>
          <div><dt><kbd>[</kbd> <kbd>]</kbd></dt><dd>Rotate 15°</dd></div>
          <div><dt><kbd>H</kbd></dt><dd>Flip X</dd></div>
          <div><dt><kbd>0</kbd></dt><dd>Fit all</dd></div>
        </dl>
      </section>
    </aside>
  );
}

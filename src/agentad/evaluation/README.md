# Evaluation

This package evaluates ordinary time-series anomaly scores with project-owned,
validated NumPy APIs. Repeated threshold and event scans use serial Numba
kernels over contiguous numeric arrays; Python event dictionaries and
per-threshold object allocation are kept out of hot paths.

## Entry points

- `evaluate(y_true, y_score, ...)` evaluates one series.
- `evaluate_collection(test, train, ...)` reconstructs matching splits and
  evaluates every series from a detector callback or score mapping.
- `point.py`, `range.py`, `vus.py` and `protocols.py` expose focused functions
  for callers that need structured results instead of flat metric columns.
- `AVAILABLE_METRICS` is the complete selectable catalog. `DEFAULT_METRICS` is
  the recommended core suite, while `TSB_AD_METRICS` preserves the nine-column
  compatibility result returned by `get_metrics(...)`.

`evaluate()` follows `(y_true, y_score)`. Supplying `y_pred` makes every
threshold-dependent family use the same fixed decision. Without `y_pred`,
standard point metrics use the exact threshold maximizing point F1; PA,
composite-event, range, affiliation, interval and tolerance metrics use the
configured deterministic threshold grid. Oracle values compare score ranking
and are not deployable threshold evidence.

`VUS-Precision`, `VUS-Recall` and `VUS-F` describe one fixed decision and
therefore require `y_pred` (or a `predictions` mapping in collection mode).

## Metric catalog

| Family | Selectable metrics | Main configuration |
|---|---|---|
| Point ranking | `AUC-PR`, `AUC-ROC`, `Precision@K` | `precision_k` |
| Range curve | `R-AUC-PR`, `R-AUC-ROC` | `sliding_window`, `vus_threshold_count` |
| Volume surface | `VUS-PR`, `VUS-ROC` | `sliding_window`, `vus_threshold_count` |
| Standard point | accuracy, precision, recall, F0.5, F1, MCC, TP/FP/FN/TN | `threshold_count` |
| Point adjustment | accuracy, precision, recall, F1, exact PRF, `PA-AUC-PR` | `threshold_count` |
| Delay adjustment | `K-Delay-PA-F1` | `k_delay` |
| Event-scaled PA | `Event-PA-F1` | `event_scale`, `event_base` |
| Composite event | precision, recall, F1 | `threshold_count` |
| Tatbul range | precision, recall, F1 | `range_alpha`, `threshold_count` |
| Affiliation | precision, recall, F1 | `threshold_count` |
| Fixed VUS | precision, recall, F1 | `sliding_window`, fixed prediction |
| Interval overlap | precision, recall, F1 | `threshold_count` |
| Tolerance PA | precision, recall, F1 | `tolerance`, `threshold_count` |
| Early/delayed PATE | `PATE`, `PATE-F1` | early/delayed buffers, splits, thresholds |
| First hit | rank, fraction, `Hit@3%`, `Hit@10%` | none |

Names use a common `Family-Statistic` convention. The exact 51 strings are
available from `AVAILABLE_METRICS`; code should use that constant instead of
duplicating the catalog.

### Range-AUC and VUS

One numeric surface computation produces both families. Range-AUC is the curve
at the requested final window. VUS is the mean area across every integer window
from zero through that window. `VUSResult.roc_by_window` and
`VUSResult.pr_by_window` expose the per-window areas without recomputation.

Fixed-decision VUS averages precision, recall and F1 across the same windows.
Every window starts from the immutable input labels, so one window cannot alter
the next window's result.

### Delay, event and tolerance protocols

`K-Delay-PA-F1` only accepts a hit within the first `k_delay` points of an
event before filling that event. `Event-PA-F1` replaces each labelled event by
its maximum score and weights its duration using `squeeze`, `log`, `sqrt` or
`raw` scaling. `Tolerance-PA-*` first dilates each ground-truth event on both
sides and then applies ordinary point adjustment.

`PA-Exact-*` reconstructs event scores and searches the exact score ordering;
ordinary `PA-*` retains the deterministic TSB-style threshold grid.

### Interval overlap and PATE

Interval metrics operate on contiguous runs. A predicted run contributes one
true-positive count for every ground-truth run it overlaps, matching the
interval-output comparison protocol while avoiding interval objects.

PATE assigns weighted TP/FP/FN mass to early, true, delayed and outside zones.
Its threshold scan and buffer-pair averaging are compiled and self-contained;
it does not depend on scikit-learn private APIs or joblib workers. `PATE-F1`
uses a supplied fixed prediction when available and otherwise reports the best
score-threshold result averaged across buffer pairs.

## Collection evaluation and aggregation

`evaluate_collection(...)` accepts:

- exactly one of `score_fn` and `scores`;
- an optional `predictions={series_id: binary_array}` mapping;
- the same metric parameters as `evaluate()`;
- `sliding_window="auto"` for ACF-based range/VUS window selection.

The result contains per-series metric rows. `summary("mean"|"median")` performs
macro aggregation and skips undefined values. `micro_summary()` sums standard
point TP/FP/FN/TN across successful series and derives exact micro point
metrics. Counts are computed internally whenever any standard point metric is
selected. `anomalous_summary()` reproduces protocols that macro-average only
series containing labelled anomalies; `n_anomalies` and `n_events` remain in
the per-series table for explicit filtering and audit.

When train and test data are consumed, their feature annotations, feature attrs
and NumPy dtypes must match exactly. Precomputed-score evaluation that does not
read data or estimate an automatic period does not impose this requirement.

`on_error="nan"|"zero"` contains score, prediction, label and per-series metric
failures in the corresponding row. Invalid collection structure and global
configuration are raised before detector execution.

## Boundary semantics

- Events use start-inclusive, stop-exclusive runs internally and include a
  positive final sample.
- Fixed affiliation with labelled events and no predicted event is `(0, 0, 0)`.
- Metrics undefined for a one-class label vector return `NaN` rather than
  inventing perfect performance.
- Empty-label first-hit rank/fraction and hit flags are `NaN` in flat metric
  output.
- Series remain independent during aggregation; event ranges never merge
  across series boundaries.

Root-cause localization, causal-graph recovery and forecasting-error measures
require different inputs and remain outside this ordinary TSAD score/label
module.

Protocol provenance and compatibility boundaries are recorded in the module
docstrings alongside the independent AgentAD implementations.

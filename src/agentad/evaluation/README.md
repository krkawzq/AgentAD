# Evaluation

This package evaluates ordinary time-series anomaly scores. It follows the
TSB-AD metric definitions while providing validated NumPy APIs, compiled metric
kernels and `SeriesData` collection orchestration.

## Public entry points

- `evaluate(y_true, y_score, ...)` evaluates one series. Without `y_pred`, each
  threshold-dependent F1 metric selects its own oracle-best threshold. With
  `y_pred`, all threshold-dependent metrics use that fixed binary decision.
- `evaluate_collection(test, train, ...)` reconstructs matching train/test
  splits, obtains full-series scores from either `score_fn` or a score mapping,
  and returns per-series rows plus macro summary helpers.
- `reconstruct_series(test, train)` yields independent arrays. Mutating a
  reconstructed result or detector input does not mutate the source
  `SeriesData` collection.
- `volume_under_surface(...)` computes VUS-ROC and VUS-PR together with their
  per-window surfaces. `generate_curve(...)` is the validated compatibility
  representation used by TSB-AD callers.

When train and test data are consumed, their feature annotations, feature attrs
and NumPy dtypes must match exactly. Precomputed-score evaluation that does not
read data or estimate an automatic period does not impose this requirement.

`evaluate_collection(..., on_error="nan"|"zero")` contains scoring, label and
per-series metric failures in the corresponding result row. Invalid collection
structure and global evaluation parameters are configuration errors and are
raised before any detector is run.

## Metrics and module boundaries

- `point.py`: point accuracy, precision, recall, F1, MCC, ROC AUC and average
  precision.
- `range.py`: point-adjusted, event, Tatbul range and affiliation metrics.
- `vus.py`: range-AUC volume-under-surface metrics.
- `period.py`: autocorrelation-based automatic VUS window selection.
- `collection.py`: split reconstruction, detector execution, score persistence
  and macro aggregation.
- `_validation.py`, `_types.py` and `_kernels.py`: internal validation, result
  containers and compiled kernels.

## TSB-AD compatibility

Regular benchmark cases are numerically checked against the vendored TSB-AD
reference. AgentAD intentionally defines two boundary cases explicitly:

- An event that reaches the final sample includes that sample. Detecting only
  the final sample therefore detects the event; this avoids the reference
  implementation's terminal off-by-one behavior.
- If ground-truth events exist but a fixed prediction contains no events,
  affiliation precision, recall and F1 are `0.0`. This prevents a detector that
  predicts nothing from disappearing from macro aggregation as an undefined
  value.

Metrics that are mathematically undefined for a one-class label vector, such as
ROC AUC, remain `NaN`.

See `THIRD_PARTY_NOTICES.md` for source attribution and license information.

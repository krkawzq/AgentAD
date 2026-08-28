# Ordinary time-series corpus

This repository normalizes ordinary numeric time-series datasets while preserving
their original time coordinates, feature identities, labels, annotations, and
official split semantics. VLM4TS/Sintel-Orion is intentionally outside this build.

## Canonical format

Every source owns its namespace, so datasets with the same name never overwrite one
another:

```text
data/processed/<source>/<dataset>/
  train[__schemaN].zarr.zip
  val[__schemaN].zarr.zip
  test[__schemaN].zarr.zip
  eval[__schemaN].zarr.zip
  data[__schemaN].zarr.zip
```

Each archive is one packed `SeriesData` collection for a single split, feature
schema, data dtype, and label mode. It stores dense values and int64 timestamps in
Zarr, feature/label tables in Parquet, and per-series provenance, annotations, and
the dataset-level `source_metadata` in compressed JSON inside the archive. No
sidecar files live next to the archives; every dataset directory contains only
`.zarr.zip` packages, and a loader discovers them with a simple glob. Reversible
datetime strings use epoch nanoseconds plus an exact format descriptor; other
timestamps that cannot be reconstructed losslessly from int64 use packed offsets
while their original values remain in the label table.
Every item records the timestamp conversion contract. M4 uses the same packed
train/test representation, so it does not create one file per series.
Monash's univariate files share the package feature `value`; each original column
name remains in that series' `source_feature_names` provenance field.

Per-series source records contain path, size, modification time, and SHA-256.
LangTime keeps every named dataset configuration independently, including settings
that differ between training and evaluation configurations.

## Python API

```python
from agentad import SeriesData, read, write

sdata = read("data/processed/source/dataset/train.zarr.zip")
item = sdata["series-id"]             # id or integer position
subset = sdata.select(["series-id"]) # independent packed subset
standalone = item.copy()              # independent SeriesItem

for series_id, item in sdata.items():
    ...

write(subset, "subset.zarr.zip")
```

Use `SeriesData.empty(...)` when an empty collection must retain feature and label
schemas. `keys()`, `values()`, `items()`, `get()`, and `index()` provide mapping-like
access; `reorder()` requires a complete permutation, while `select()` accepts any
unique subset, including an empty one. Feature and label column labels may use
pandas-supported scalar or MultiIndex schemas, but must be unique; default integer
column labels are supported. Pass `create_parents=True` to `write()` when output
parent directories should be created explicitly.

## Evaluation API

The ordinary time-series metric suite is available without importing it into the
lightweight top-level package:

```python
from agentad.evaluation import evaluate, evaluate_collection

values = evaluate(labels, anomaly_scores)
fixed_values = evaluate(labels, anomaly_scores, y_pred=predictions)

report = evaluate_collection(
    test,
    train,
    scores={"series-id": full_series_scores},
    predictions={"series-id": full_series_predictions},  # optional
)
macro_average = report.summary()
micro_point_metrics = report.micro_summary()
```

`evaluate()` follows the `(y_true, y_score)` convention. Without `y_pred`, each
threshold-dependent family uses its documented oracle threshold; with `y_pred`,
every threshold-dependent metric evaluates the same fixed decision. The full
catalog includes point, range-AUC, VUS, PA, event, affiliation, interval,
delay-aware, PATE and first-hit protocols. See
`src/agentad/evaluation/README.md` for the metric survey and module boundaries.

## Build

Run full preprocessing only on a valid PJLab pod, never on the dev box.
`build_all.sh` is the single build entry point: it stages every source
processor in parallel, verifies the build inputs did not change mid-build,
and publishes `data/processed` atomically with a backup of the previous tree.

```bash
scripts/preprocess/build_all.sh --jobs 4
```

A failed build is never published; its staging tree is retained for
inspection. The output destination must resolve to a dedicated child of the
project root; `build_all.sh` rejects the project itself, its ancestors, and
other external destinations. Run logs land in
`logs/preprocess/<timestamp>-<pid>/` and never inside the published data tree.

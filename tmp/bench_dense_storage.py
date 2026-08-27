"""Benchmark: dense packed label storage vs one-file-per-item.

Scenario: 100k items x 1000 points, one 0/1 int8 label column (1e8 labels,
100 MB raw int8).

  per_item - 100k individual parquet.gzip files (1000 rows each)
  packed   - one parquet.gzip holding all 1e8 rows, i.e. the layout the
             project's ``agentad.series.write`` uses for labels

Measures write time, logical/disk size, full-read time and single-item
read latency. The per-item phase does 100k filesystem operations and is
expected to run for many minutes: run on a pod, not the dev box.
"""

import os
import shutil
import statistics
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

N_ITEMS = 100_000
SEQ_LEN = 1000
LABEL_COL = "label"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_storage")


def make_labels() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 2, (N_ITEMS, SEQ_LEN), dtype=np.int8)


def disk_usage_bytes(path: str) -> int:
    # st_blocks captures filesystem block rounding, like du
    if os.path.isfile(path):
        return os.stat(path).st_blocks * 512
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total += os.stat(os.path.join(root, name)).st_blocks * 512
    return total


def logical_bytes(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(
        os.path.getsize(os.path.join(root, name))
        for root, _dirs, files in os.walk(path)
        for name in files
    )


def bench_per_item(labels: np.ndarray) -> dict:
    out = os.path.join(ROOT, "per_item")
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)

    t0 = time.perf_counter()
    for i in range(N_ITEMS):
        frame = pd.DataFrame({LABEL_COL: labels[i]})
        frame.to_parquet(
            os.path.join(out, f"item_{i:06d}.parquet.gzip"),
            compression="gzip",
            index=False,
        )
        if (i + 1) % 10_000 == 0:
            print(f"  per_item write {i + 1}/{N_ITEMS}", flush=True)
    write_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    for i in range(N_ITEMS):
        pd.read_parquet(os.path.join(out, f"item_{i:06d}.parquet.gzip"))
        if (i + 1) % 10_000 == 0:
            print(f"  per_item read {i + 1}/{N_ITEMS}", flush=True)
    read_all_s = time.perf_counter() - t0

    rng = np.random.default_rng(1)
    picks = rng.choice(N_ITEMS, 50, replace=False)
    latencies = []
    for i in picks:
        t0 = time.perf_counter()
        pd.read_parquet(os.path.join(out, f"item_{i:06d}.parquet.gzip"))
        latencies.append(time.perf_counter() - t0)

    return {
        "write_s": write_s,
        "read_all_s": read_all_s,
        "read_one_ms_median": statistics.median(latencies) * 1e3,
        "logical_mb": logical_bytes(out) / 1e6,
        "disk_mb": disk_usage_bytes(out) / 1e6,
        "files": len(os.listdir(out)),
    }


def bench_packed(labels: np.ndarray) -> dict:
    path = os.path.join(ROOT, "labels_packed.parquet.gzip")
    if os.path.exists(path):
        os.remove(path)

    flat = pd.DataFrame({LABEL_COL: labels.reshape(-1)})
    t0 = time.perf_counter()
    flat.to_parquet(path, compression="gzip", index=False)
    write_s = time.perf_counter() - t0
    del flat

    t0 = time.perf_counter()
    whole = pd.read_parquet(path)
    read_all_s = time.perf_counter() - t0
    assert len(whole) == N_ITEMS * SEQ_LEN
    del whole

    parquet = pq.ParquetFile(path)
    n_groups = parquet.metadata.num_row_groups
    rows_per_group = parquet.metadata.row_group(0).num_rows

    # Packed random access pays row-group granularity: one item may force
    # decompressing the whole row group that contains it.
    rng = np.random.default_rng(1)
    picks = rng.choice(N_ITEMS, 50, replace=False)
    latencies = []
    for i in picks:
        t0 = time.perf_counter()
        group = parquet.read_row_group(i * SEQ_LEN // rows_per_group)
        _ = group.column(LABEL_COL)
        latencies.append(time.perf_counter() - t0)

    return {
        "write_s": write_s,
        "read_all_s": read_all_s,
        "read_one_ms_median": statistics.median(latencies) * 1e3,
        "logical_mb": logical_bytes(path) / 1e6,
        "disk_mb": disk_usage_bytes(path) / 1e6,
        "files": 1,
        "row_groups": n_groups,
        "rows_per_row_group": rows_per_group,
    }


def main() -> None:
    os.makedirs(ROOT, exist_ok=True)
    raw_mb = N_ITEMS * SEQ_LEN / 1e6
    print(
        f"labels: {N_ITEMS} items x {SEQ_LEN} points, int8 0/1 "
        f"({raw_mb:.0f} MB raw)",
        flush=True,
    )
    labels = make_labels()

    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("per_item", "both"):
        print("== per_item: 100k files ==", flush=True)
        for key, value in bench_per_item(labels).items():
            print(f"  {key}: {value}", flush=True)
    if which in ("packed", "both"):
        print("== packed: 1 dense parquet.gzip ==", flush=True)
        for key, value in bench_packed(labels).items():
            print(f"  {key}: {value}", flush=True)


if __name__ == "__main__":
    main()

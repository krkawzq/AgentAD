"""Benchmark: 100k small per-item label objects holding 0/1 labels, seq len 1000.

Each mode runs in its own process so ru_maxrss deltas stay clean:
  numpy_uint8   - raw numpy arrays, no pandas (floor)
  series_int8   - pd.Series, int8, default RangeIndex
  frame_int8    - pd.DataFrame (1000x1), int8, default RangeIndex
  frame_int8_ts - frame_int8 plus an int64 positional index per item
  frame_int64   - pd.DataFrame (1000x1), int64 (naive 0/1), default RangeIndex
  one_big_frame - single (100k x 1000) int8 frame, the packed-series design
"""

import argparse
import gc
import resource
import time

import numpy as np
import pandas as pd


def rss_mb() -> float:
    # ru_maxrss is KiB on Linux
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build(mode: str, n: int, seq_len: int) -> list:
    rng = np.random.default_rng(0)
    keep = []
    for i in range(n):
        if mode == "numpy_uint8":
            keep.append(rng.integers(0, 2, seq_len, dtype=np.uint8))
        elif mode == "series_int8":
            keep.append(pd.Series(rng.integers(0, 2, seq_len, dtype=np.int8)))
        elif mode == "frame_int8":
            keep.append(pd.DataFrame(rng.integers(0, 2, (seq_len, 1), dtype=np.int8)))
        elif mode == "frame_int8_ts":
            keep.append(
                pd.DataFrame(
                    rng.integers(0, 2, (seq_len, 1), dtype=np.int8),
                    index=np.arange(i, i + seq_len, dtype=np.int64),
                )
            )
        elif mode == "frame_int64":
            keep.append(pd.DataFrame(rng.integers(0, 2, (seq_len, 1), dtype=np.int64)))
        elif mode == "one_big_frame":
            keep = [pd.DataFrame(rng.integers(0, 2, (n, seq_len), dtype=np.int8))]
            break
        else:
            raise ValueError(mode)
    return keep


def self_reported_mb(mode: str, objs: list, n: int) -> float:
    if mode == "numpy_uint8":
        per = sum(a.nbytes for a in objs[:2000]) / len(objs[:2000])
        return per * n / 1e6
    if mode == "one_big_frame":
        return float(objs[0].memory_usage(deep=True).sum()) / 1e6

    # Series.memory_usage returns a scalar; DataFrame's returns a per-column Series
    vals = []
    for m in objs[:2000]:
        usage = m.memory_usage(deep=True)
        vals.append(float(usage.sum() if hasattr(usage, "sum") else usage))
    return sum(vals) / len(vals) * n / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True)
    parser.add_argument("--n", type=int, default=100_000)
    parser.add_argument("--seq-len", type=int, default=1000)
    args = parser.parse_args()

    gc.collect()
    base = rss_mb()
    t0 = time.perf_counter()
    objs = build(args.mode, args.n, args.seq_len)
    dt = time.perf_counter() - t0
    peak = rss_mb()
    delta = peak - base

    print(f"mode={args.mode} n={args.n} seq_len={args.seq_len}")
    print(f"  alloc_time   : {dt:.2f} s")
    print(f"  peak_rss     : {peak:.0f} MB (delta {delta:.0f} MB)")
    if args.mode != "one_big_frame":
        print(f"  per_item_rss : {delta * 1e6 / args.n:.0f} B")
    print(f"  self_reported: {self_reported_mb(args.mode, objs, args.n):.0f} MB")


if __name__ == "__main__":
    main()

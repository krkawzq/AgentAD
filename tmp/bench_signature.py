"""Benchmark label-frame signature implementations."""

import hashlib
import pickle
import time

import numpy as np
import pandas as pd

from agentad.series._data import _frame_signature as hybrid_sig


def pickle_sig(frame: pd.DataFrame) -> bytes | None:
    try:
        values = frame.to_numpy(copy=True)
        payload = pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL)
    except (TypeError, ValueError, pickle.PickleError):
        return None
    digest = hashlib.blake2b(digest_size=16)
    digest.update(payload)
    digest.update(repr(tuple(frame.columns)).encode("utf-8"))
    digest.update(repr(tuple(str(dtype) for dtype in frame.dtypes)).encode("utf-8"))
    digest.update(str(frame.shape).encode("ascii"))
    return digest.digest()


def bench(name, fn, frame, repeat=5):
    fn(frame)  # warmup
    start = time.perf_counter()
    for _ in range(repeat):
        fn(frame)
    elapsed = (time.perf_counter() - start) / repeat
    print(f"{name:28s} {elapsed * 1000:8.2f} ms/call")
    return elapsed


numeric = pd.DataFrame(
    {
        "is_anomaly": np.zeros(1_000_000, dtype=np.int64),
        "score": np.random.default_rng(0).normal(size=1_000_000),
        "label": np.zeros(1_000_000, dtype=np.int64),
    }
)
print(f"numeric frame: {numeric.shape}, {numeric.memory_usage(deep=False).sum() / 1e6:.0f} MB data")
bench("pickle (old)", pickle_sig, numeric)
bench("hybrid (new)", hybrid_sig, numeric)

# stability + mutation detection on the hybrid path
assert hybrid_sig(numeric) == hybrid_sig(numeric)
mutated = numeric.copy()
mutated.iloc[0, 0] = 1
assert hybrid_sig(mutated) != hybrid_sig(numeric)

# unhashable values: pickle fallback now yields a working fingerprint
lists = pd.DataFrame({"v": [[1], [2], [3]] * 1000})
print(f"list-valued frame: {lists.shape}")
print("  pickle sig is None:", pickle_sig(lists) is None)
print("  hybrid sig is None:", hybrid_sig(lists) is None, "(fallback covers it)")
assert hybrid_sig(lists) == hybrid_sig(lists)
print("OK")

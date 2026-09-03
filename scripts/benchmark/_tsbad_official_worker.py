#!/usr/bin/env python
"""Run a complete TSB-AD Eva split inside the isolated reference environment."""

from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile

import numpy as np
import pandas as pd
import torch


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _atomic_save(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(dir=path.parent, suffix=".npy")
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, values, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--source", choices=["U", "M"], required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--file-list", type=Path, required=True)
    parser.add_argument("--score-dir", type=Path, required=True)
    parser.add_argument("--score-scope", choices=["full", "test"], required=True)
    parser.add_argument("--contract-file", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    reference_root = args.reference_root.resolve()
    sys.path.insert(0, str(reference_root))
    wrapper = importlib.import_module("TSB_AD.model_wrapper")
    contract = json.loads(args.contract_file.read_text(encoding="utf-8"))
    hp_contract = contract.get("hp")
    if not isinstance(hp_contract, dict):
        raise TypeError("contract file must contain an hp object")
    is_semisupervised = args.method in wrapper.Semisupervise_AD_Pool
    is_unsupervised = args.method in wrapper.Unsupervise_AD_Pool
    if is_semisupervised == is_unsupervised:
        raise RuntimeError(
            f"{args.method!r} must occur in exactly one official detector pool"
        )

    files = pd.read_csv(args.file_list)["file_name"].astype(str).tolist()
    _seed(args.seed)
    for position, filename in enumerate(files, start=1):
        frame = pd.read_csv(args.raw_dir / filename).dropna()
        data = frame.iloc[:, :-1].to_numpy(dtype=float)
        train_length = int(Path(filename).stem.split("_")[-3])
        data_train = data[:train_length]
        target = data[train_length:] if args.score_scope == "test" else data
        hp = dict(hp_contract)
        if hp.get("win_size") == "min(10000, floor(400000 / n_features))":
            hp["win_size"] = min(10000, 400000 // data.shape[1])
        print(f"[{position}/{len(files)}] {args.method}: {filename}", flush=True)
        if is_semisupervised:
            output = wrapper.run_Semisupervise_AD(
                args.method, data_train, target, **hp
            )
        else:
            output = wrapper.run_Unsupervise_AD(args.method, target, **hp)
        if not isinstance(output, np.ndarray):
            raise RuntimeError(f"official wrapper failed for {filename}: {output}")
        score = np.asarray(output, dtype=np.float64).reshape(-1)
        if score.shape != (len(target),) or not np.isfinite(score).all():
            raise ValueError(
                f"official wrapper returned {score.shape} for {filename}; "
                f"expected {(len(target),)} finite values"
            )
        _atomic_save(args.score_dir / f"{Path(filename).stem}.npy", score)


if __name__ == "__main__":
    main()

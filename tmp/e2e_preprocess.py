"""End-to-end preprocessing check on miniature synthetic raw data."""

from __future__ import annotations

import json
import pickle
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# The shared-storage mount rejects mmap, so raw fixtures live on a local
# temporary filesystem; only the processed output stays in the project.
RAW = Path(tempfile.mkdtemp(prefix="agentad-e2e-raw-"))
OUT = ROOT / "tmp" / "e2e_processed"
SCRIPTS = ROOT / "scripts" / "preprocess"

PROCESSES = (
    ("process_anomllm.py", ["--input-dir", str(RAW / "anomllm" / "data" / "synthetic")]),
    ("process_crossad.py", ["--input-dir", str(RAW / "CrossAD" / "dataset"),
                            "--meta-path", str(RAW / "crossad_meta.csv")]),
    ("process_dada.py", ["--input-dir", str(RAW / "DADA")]),
    ("process_dcdetector.py", ["--input-dir", str(RAW / "DCDetector" / "benchmark")]),
    ("process_easytsad.py", ["--input-dir", str(RAW / "EasyTSAD")]),
    ("process_langtime.py", ["--input-dir", str(RAW / "LangTime" / "dataset"),
                             "--config", str(RAW / "langtime_config.json")]),
    ("process_sintel_orion.py", ["--input-dir", str(RAW / "Sintel-Orion")]),
    ("process_tsbad.py", ["--input-dir", str(RAW / "TSB-AD"),
                          "--file-list-dir", str(RAW / "tsbad_lists")]),
)


def build_fixtures() -> None:
    shutil.rmtree(RAW, ignore_errors=True)
    shutil.rmtree(OUT, ignore_errors=True)

    # anomllm: pickle of {"series": [...], "anom": [[[start, end]]]}
    for split in ("train", "eval"):
        path = RAW / "anomllm" / "data" / "synthetic" / "typeA" / split
        path.mkdir(parents=True)
        payload = {
            "series": [np.arange(10, dtype=np.float64).reshape(-1, 1),
                       np.ones((4, 2))],
            "anom": [[[[2, 4]]], []],
        }
        with (path / "data.pkl").open("wb") as handle:
            pickle.dump(payload, handle)

    # crossad long CSV + UCR text
    long_dir = RAW / "CrossAD" / "dataset" / "data"
    long_dir.mkdir(parents=True)
    pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02",
                 "2024-01-03", "2024-01-03"],
        "data": [1.0, 0, 2.0, 0, 3.0, 1],
        "cols": ["value", "label", "value", "label", "value", "label"],
    }).to_csv(long_dir / "longA.csv", index=False)
    (RAW / "crossad_meta.csv").write_text("file_name,train_lens\nlongA.csv,2\n")
    ucr_dir = RAW / "CrossAD" / "dataset" / "UCR_Anomaly_FullData"
    ucr_dir.mkdir(parents=True)
    (ucr_dir / "UCR_001_2_3_4.txt").write_text(
        "\n".join(str(float(value)) for value in range(6)) + "\n")

    # dada eval (long csv + meta.npy) and monash csv
    eval_dir = RAW / "DADA" / "eval" / "dataset" / "evaluation_dataset"
    eval_dir.mkdir(parents=True)
    pd.DataFrame({
        "date": ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02",
                 "2024-01-03", "2024-01-03"],
        "data": [5.0, 0, 6.0, 0, 7.0, 1],
        "cols": ["value", "label", "value", "label", "value", "label"],
    }).to_csv(eval_dir / "A.csv", index=False)
    meta = {
        "data_info": {
            "file_name": {0: "A.csv"},
            "train_lens": {0: 2},
            "dataset_name": {0: "DadaA"},
        },
        "channel_info": {"DadaA": ["value"]},
    }
    np.save(eval_dir / "meta.npy", meta, allow_pickle=True)
    monash_dir = RAW / "DADA" / "monash" / "dataset" / "Monash"
    monash_dir.mkdir(parents=True)
    pd.DataFrame({"date": ["2024-01-01", "2024-01-02"],
                  "usage": [1.5, 2.5]}).to_csv(monash_dir / "m1.csv", index=False)

    # dcdetector paired .out files
    pair_dir = RAW / "DCDetector" / "benchmark" / "D1"
    pair_dir.mkdir(parents=True)
    (pair_dir / "s1.train.out").write_text("1.0,0\n2.0,0\n3.0,1\n")
    (pair_dir / "s1.test.out").write_text("4.0,0\n5.0,1\n")

    # easytsad npy quartet
    series_dir = RAW / "EasyTSAD" / "MTS" / "E1" / "all"
    series_dir.mkdir(parents=True)
    np.save(series_dir / "train.npy", np.arange(4, dtype=np.float32).reshape(-1, 1))
    np.save(series_dir / "test.npy", np.arange(4, 7, dtype=np.float32).reshape(-1, 1))
    np.save(series_dir / "train_label.npy", np.array([0, 0, 1, 0], dtype=np.int8))
    np.save(series_dir / "test_label.npy", np.array([1, 0, 0], dtype=np.int8))
    (series_dir / "info.json").write_text('{"freq": "1min"}')

    uts_dir = RAW / "EasyTSAD" / "UTS" / "U1" / "all"
    uts_dir.mkdir(parents=True)
    np.save(uts_dir / "train.npy", np.arange(3, dtype=np.float64))
    np.save(uts_dir / "test.npy", np.arange(3, 5, dtype=np.float64))
    np.save(uts_dir / "train_label.npy", np.array([0, 1, 0], dtype=np.int8))
    np.save(uts_dir / "test_label.npy", np.array([0, 1], dtype=np.int8))

    # langtime standard csv + m4 + PEMS + solar
    ett_dir = RAW / "LangTime" / "dataset" / "ETT-h1"
    ett_dir.mkdir(parents=True)
    pd.DataFrame({
        "date": ["2024-01-01 00:00:00", "2024-01-02 00:00:00",
                 "2024-01-03 00:00:00"],
        "HUFL": [1.0, 2.0, 3.0],
        "HULL": [4.0, 5.0, 6.0],
    }).to_csv(ett_dir / "ett1.csv", index=False)
    (RAW / "langtime_config.json").write_text(json.dumps({
        "ett1": {"data_path": "ETT-h1/ett1.csv", "target": "HUFL", "freq": "h"},
    }))
    m4_dir = RAW / "LangTime" / "dataset" / "m4"
    m4_dir.mkdir(parents=True)
    pd.DataFrame({"V1": ["N1", "D1"], "SP": [1, 24]}).to_csv(
        m4_dir / "M4-info.csv", index=False)
    for name, series_list in (
        ("training.npz", [np.array([1., 2, 3]), np.array([4., 5])]),
        ("test.npz", [np.array([6.]), np.array([7., 8])]),
    ):
        with (m4_dir / name).open("wb") as handle:
            np.save(handle, np.array(series_list, dtype=object))
    pems_dir = RAW / "LangTime" / "dataset" / "PEMS"
    pems_dir.mkdir(parents=True)
    np.savez(pems_dir / "03.npz",
             data=np.arange(10, dtype=np.float64).reshape(5, 2, 1))
    solar_dir = RAW / "LangTime" / "dataset" / "solar"
    solar_dir.mkdir(parents=True)
    (solar_dir / "solar_AL.txt").write_text("1,2\n3,4\n5,6\n")

    # sintel-orion signal csv + anomalies
    ds_dir = RAW / "Sintel-Orion" / "DS1"
    ds_dir.mkdir(parents=True)
    pd.DataFrame({"timestamp": [1, 2, 3, 4], "value": [1.0, 2.0, 3.0, 4.0]}
                 ).to_csv(ds_dir / "sig1.csv", index=False)
    pd.DataFrame({"dataset": ["DS1"], "signal": ["sig1"],
                  "start": [2], "end": [3]}).to_csv(
        RAW / "Sintel-Orion" / "anomalies.csv", index=False)

    # tsbad csv + file list
    tsb_dir = RAW / "TSB-AD" / "TSB-AD-U"
    tsb_dir.mkdir(parents=True)
    rows = 6
    pd.DataFrame({
        "value": np.arange(rows, dtype=np.float64),
        "label": ([0] * 4) + [1, 0],
    }).to_csv(tsb_dir / "1_Machine_id_5_tr_4_1st_4.csv", index=False)
    lists = RAW / "tsbad_lists"
    lists.mkdir(parents=True)
    (lists / "TSB-AD-U.csv").write_text(
        "file_name\n1_Machine_id_5_tr_4_1st_4.csv\n")
    # Two multivariate series with different channel counts, exercising the
    # per-schema package split.
    tsb_m_dir = RAW / "TSB-AD" / "TSB-AD-M"
    tsb_m_dir.mkdir(parents=True)
    pd.DataFrame({
        "v1": np.arange(5, dtype=np.float64),
        "v2": np.arange(5, dtype=np.float64) * 2,
        "label": [0, 0, 1, 0, 0],
    }).to_csv(tsb_m_dir / "2_MachineA_id_5_tr_3_1st_4.csv", index=False)
    pd.DataFrame({
        "v1": np.arange(4, dtype=np.float64),
        "v2": np.arange(4, dtype=np.float64) * 2,
        "v3": np.arange(4, dtype=np.float64) * 3,
        "label": [0, 1, 0, 0],
    }).to_csv(tsb_m_dir / "3_MachineB_id_5_tr_2_1st_4.csv", index=False)
    (lists / "TSB-AD-M.csv").write_text(
        "file_name\n2_MachineA_id_5_tr_3_1st_4.csv\n"
        "3_MachineB_id_5_tr_2_1st_4.csv\n")


def run_all() -> None:
    environment = {"PYTHONDONTWRITEBYTECODE": "1"}
    for script, extra in PROCESSES:
        command = [sys.executable, str(SCRIPTS / script),
                   "--output-dir", str(OUT), *extra]
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                capture_output=True, text=True)
        status = "ok" if result.returncode == 0 else "FAIL"
        print(f"{script}: {status}")
        if result.returncode != 0:
            print(result.stdout[-3000:])
            print(result.stderr[-3000:])
            raise SystemExit(1)
        tail = [line for line in result.stdout.strip().splitlines() if line]
        for line in tail[-2:]:
            print(f"    {line}")


def validate() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "validate_processed.py"), str(OUT),
         "--verify-hashes"],
        cwd=ROOT, capture_output=True, text=True)
    print(result.stdout.strip())
    if result.returncode != 0:
        print(result.stderr[-3000:])
        raise SystemExit(1)


def main() -> None:
    build_fixtures()
    run_all()
    validate()
    print("E2E OK")


if __name__ == "__main__":
    main()

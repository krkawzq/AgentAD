"""Convert Sintel-Orion signals into SeriesData packages.

Input:  data/raw/Sintel-Orion/<DS>/*.csv          (timestamp,value)
        data/raw/Sintel-Orion/anomalies.csv       (anomaly intervals)
Output: data/processed/sintel-orion/<DS>/<split>.zarr.zip
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .common import (
        DatasetWriter,
        SplitData,
        staged_source_directory,
        strict_int,
    )
else:
    from common import (
        DatasetWriter,
        SplitData,
        staged_source_directory,
        strict_int,
    )


SOURCE = "sintel-orion"


def load_anomalies(csv_path: Path) -> dict[str, list[tuple[int, int]]]:
    """Load Orion's ``signal,events`` table with inclusive event endpoints."""
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    table = pd.read_csv(csv_path, dtype={"signal": "string", "events": "string"})
    required = {"signal", "events"}
    if not required <= set(table.columns):
        raise ValueError(
            f"invalid Orion anomalies table; missing {sorted(required - set(table.columns))}"
        )
    result: dict[str, set[tuple[int, int]]] = {}
    for row_index, row in table.iterrows():
        if pd.isna(row["signal"]):
            raise ValueError(f"invalid anomaly row {row_index} in {csv_path}")
        signal = str(row["signal"]).strip()
        if not signal or pd.isna(row["events"]):
            raise ValueError(f"invalid anomaly row {row_index} in {csv_path}")
        try:
            raw_events = ast.literal_eval(str(row["events"]))
        except (SyntaxError, ValueError) as error:
            raise ValueError(
                f"invalid events for signal {signal!r} in {csv_path}"
            ) from error
        if not isinstance(raw_events, (list, tuple)):
            raise ValueError(f"events for signal {signal!r} are not a sequence")
        events = result.setdefault(signal, set())
        for event_index, event in enumerate(raw_events):
            if not isinstance(event, (list, tuple)) or len(event) != 2:
                raise ValueError(
                    f"invalid event {event_index} for signal {signal!r}: {event!r}"
                )
            start = strict_int(event[0], name=f"{signal} event start")
            end = strict_int(event[1], name=f"{signal} event end")
            if start > end:
                raise ValueError(f"reversed event for signal {signal!r}: {event!r}")
            events.add((start, end))
    return {signal: sorted(events) for signal, events in result.items()}


def mark_labels(
    frame: pd.DataFrame, anomalies: dict[str, list[tuple[int, int]]], signal: str
) -> pd.DataFrame:
    """Mark points whose timestamp falls inside an anomaly interval."""
    frame = frame.copy()
    frame["label"] = 0
    if signal not in anomalies:
        raise KeyError(f"no anomaly metadata for Orion signal {signal!r}")
    for start, end in anomalies[signal]:
        frame.loc[
            (frame["timestamp"] >= start) & (frame["timestamp"] <= end), "label"
        ] = 1
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Process Sintel-Orion dataset")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw/Sintel-Orion"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()

    project_root = Path.cwd().resolve()
    input_root = args.input_dir.resolve()
    anomaly_path = input_root / "anomalies.csv"
    anomalies = load_anomalies(anomaly_path)
    with staged_source_directory(args.output_dir.resolve(), SOURCE) as source_dir:
        dataset_dirs = sorted(path for path in input_root.iterdir() if path.is_dir())
        if not dataset_dirs:
            raise FileNotFoundError(f"no Orion datasets under {input_root}")
        for ds_dir in dataset_dirs:
            files = sorted(ds_dir.glob("*.csv"))
            if not files:
                print(f"{SOURCE}: skipped {ds_dir.name}: no CSV files")
                continue
            writer = DatasetWriter(
                source_dir / ds_dir.name,
                source=SOURCE,
                dataset=ds_dir.name,
                task="anomaly_detection",
                project_root=project_root,
                data_dtype=np.float64,
            )
            for path in files:
                frame = pd.read_csv(path)
                required = {"timestamp", "value"}
                if frame.empty or not required <= set(frame.columns):
                    raise ValueError(f"invalid Orion signal CSV: {path}")
                frame = frame.loc[:, ["timestamp", "value"]].copy()
                for column in ("timestamp", "value"):
                    frame[column] = pd.to_numeric(frame[column], errors="coerce")
                source_rows = len(frame)
                frame = frame.dropna(subset=["timestamp", "value"])
                frame = frame.sort_values("timestamp", kind="stable").reset_index(
                    drop=True
                )
                if frame.empty:
                    raise ValueError(f"Orion signal has no numeric rows: {path}")
                duplicate_timestamp_rows = int(frame["timestamp"].duplicated().sum())
                frame = mark_labels(frame, anomalies, path.stem)
                annotations = []
                timestamps = frame["timestamp"].to_numpy()
                for start, end in anomalies[path.stem]:
                    matches = np.flatnonzero(
                        (timestamps >= start) & (timestamps <= end)
                    )
                    event = {
                        "split": "data",
                        "start": start,
                        "end": end,
                        "label": 1,
                        "source_end_is_inclusive": True,
                    }
                    if len(matches):
                        event.update(
                            {
                                "start_index": int(matches[0]),
                                "end_index": int(matches[-1]),
                            }
                        )
                    else:
                        print(
                            f"{SOURCE}/{ds_dir.name}/{path.stem}: event [{start}, "
                            f"{end}] contains no sampled points"
                        )
                    annotations.append(event)
                writer.add_series(
                    series_id=path.stem,
                    splits={
                        "data": SplitData(
                            frame[["value"]],
                            frame["timestamp"],
                            frame["label"],
                            ["value"],
                            timestamp_kind="source",
                        )
                    },
                    source_files=[path, anomaly_path],
                    source_metadata={
                        "source_rows": int(source_rows),
                        "dropped_rows": int(source_rows - len(frame)),
                        "duplicate_timestamp_rows": duplicate_timestamp_rows,
                    },
                    annotations=annotations,
                )
            summary = writer.finalize(source_metadata={"domain": "mixed", "dims": 1})
            print(f"  {ds_dir.name}: {summary['num_series']} series")


if __name__ == "__main__":
    sys.exit(main())

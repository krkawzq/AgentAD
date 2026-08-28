# AgentAD Visualizer CSV Contract

The visualizer accepts UTF-8 CSV files that follow the
`agentad.visualize.csv.v1` column contract.

## Columns

| Column | Required | Meaning |
| --- | --- | --- |
| `series_id` | Yes | Non-empty series identifier. Rows for one identifier must be contiguous. |
| `timestamp` | Yes | Signed 64-bit integer or ISO-8601 datetime. Use one encoding per file. Row order is preserved. |
| `feature.<name>` | At least one | Numeric feature rendered as a polyline. |
| `label.<name>` | No | Point-level label. Binary labels are available as chart overlays. |
| `meta.<name>` | No | Per-series value. It must be constant within each series block. |

Column names must be unique. Empty names and columns outside the contract are
rejected. ISO-8601 timestamps without an explicit offset are interpreted as
UTC. The default service-side file-size limit is 512 MiB and can be changed
with `--max-csv-mib`.

## Example

```csv
series_id,timestamp,feature.temperature,feature.pressure,label.is_anomaly,meta.split
sample-01,2026-01-01T00:00:00Z,21.2,1008.4,0,test
sample-01,2026-01-01T00:01:00Z,21.8,1008.1,1,test
sample-02,2026-01-02T00:00:00Z,18.7,1012.0,0,train
sample-02,2026-01-02T00:01:00Z,18.9,1011.8,0,train
```

# TSB-AD leaderboard reproduction

The current TSB-AD leaderboard is not the unchanged result table from the
NeurIPS 2024 paper. It combines the original benchmark with later community
submissions. The official repository began accepting community submissions in
2026 and stores their submitted per-series results under
`benchmark_exp/leaderboard_results/`.

AgentAD therefore has two deliberately separate benchmark paths:

- Existing method scripts run AgentAD-native implementations. Their output is
  useful for implementation experiments, but it must not be presented as an
  exact reproduction of the public TSB-AD leaderboard.
- `scripts/benchmark/tsbad_official.py` is the strict leaderboard path. It runs
  the official dispatcher and selected HP in an isolated environment, preserves
  the official Eva-file order and seed, imports the scores, then checks every
  local VUS-PR against the official per-series result CSV.

## Pinned reference

The strict path accepts only TSB-AD commit
`e0975a5f7d3e65ab77e9fab24d1b5b51acda8f48`, the source snapshot containing the
July 1, 2026 leaderboard update. Download it with:

```bash
python scripts/download/download_tsbad_reference.py
```

Create a separate environment for that checkout. This separation is required
because the reference project pins NumPy 1.x while AgentAD uses NumPy 2.x.
Install the official requirements plus any method-specific dependencies listed
in `forks/TSB-AD/TSB_AD/models/README.md`. TSPulse is stricter: its official
README recommends Python 3.12.9 and granite-tsfm 0.3.2 for exact reproduction,
and the strict runner enforces both versions before starting that method.

## Protocol provenance

| Public row | Official runner | Provenance | Score/evaluation scope |
|---|---|---|---|
| PCA | `Sub_PCA` on U, `PCA` on M | 2024 benchmark | full CSV |
| MCD | `Sub_MCD` on U, `MCD` on M | 2024 benchmark | full CSV |
| OCSVM | `Sub_OCSVM` on U, `OCSVM` on M | 2024 benchmark | full CSV |
| HBOS | `Sub_HBOS` on U, `HBOS` on M | 2024 benchmark | full CSV |
| KNN | `Sub_KNN` on U, `KNN` on M | 2024 benchmark | full CSV |
| KMeansAD | `KMeansAD_U` on U, `KMeansAD` on M | 2024 benchmark | full CSV |
| TSPulse (ZS/FT) | `TSPulse_ZS` / `TSPulse_FT` | community submission | full CSV |
| xLSTMAD | `xLSTMAD` | community submission | full CSV |
| MMPAD | `MMPAD` | community submission | full CSV |
| CHARM | `CHARM` | community submission | full CSV |
| StreamVAE | `StreamVAE` | community submission | full CSV |
| Time-RCD | `Time_RCD` | community submission | test suffix |
| Time-RCD+MAFT | `TimeRCD_MAFT` | community submission | full CSV |

Time-RCD is the important exception. Its archived integration script removes the
filename-encoded normal prefix before inference and evaluates the suffix, while
computing the VUS sliding window from the original full CSV. The generic current
`Run_Detector_U/M.py` does not preserve this submission detail, so AgentAD records
it as an explicit method profile. The submitted univariate window is 15000; the
multivariate submission uses
`min(10000, floor(400000 / number_of_channels))`, as recorded row by row in the
official `Multi_Time_RCD.csv` result. TSPulse-M is also a submitted profile because
the current official HP list omits its multivariate entries; it uses window 96,
time mode, and (for FT) learning rate `1e-4` from the integrated wrapper.

## Run and verify

Run the complete Eva split in one process; do not split or parallelize it because
the official benchmark seeds once before walking the file list.

```bash
python scripts/benchmark/tsbad_official.py \
  --method PCA \
  --source U \
  --reference-python /path/to/tsbad-env/bin/python
```

Examples for later submissions:

```bash
python scripts/benchmark/tsbad_official.py --method MMPAD --source M \
  --reference-python /path/to/tsbad-env/bin/python
python scripts/benchmark/tsbad_official.py --method Time-RCD --source U \
  --reference-python /path/to/tsbad-env/bin/python
```

Raw wrapper scores are retained under
`outputs/tsb-ad-reference-scores/<commit>/`. Imported scores and metrics use the
separate method namespace `TSB-AD-official/`; native results are never silently
overwritten. Every result artifact gets a `parity.json`. A run fails if any
series differs from the official VUS-PR by more than the configured tolerance.
The default `5.1e-6` accounts only for TSPulse result CSVs stored to five decimal
places. Use `--import-only` to verify an already completed raw-score run.

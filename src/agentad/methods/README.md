# Neural methods

This package contains independent, from-scratch PyTorch implementations of eight
published anomaly and root-cause detectors. The code keeps the algorithms' model
and scoring semantics, while using AgentAD's common tensor convention:

```text
input series:   [batch, time, feature]
point scores:   [batch, time]
```

The methods are intentionally separate from dataset loading and metric evaluation.
Use `SeriesData` to obtain arrays, create model-specific training windows, and pass
the resulting tensors to a method. Materialize the project environment first:

```bash
uv sync
```

| Package | Training signal | Score |
|---|---|---|
| `AERCA` | reconstruction, sparse/smooth dynamic Granger coefficients, residual KL | decoder error plus signed residual root-cause scores |
| `CARLA` | injected-anomaly contrastive pretext and neighbor self-classification | inverse probability of the calibrated normal cluster |
| `CrossAD` | cross-scale reconstruction plus context quantization | summed, upsampled reconstruction error |
| `DADA` | released masked reconstruction/MoE components | variance across complementary masked reconstructions |
| `KanAD` | next-value forecasting | absolute forecast error |
| `Left` | spectral multiscale, time-frequency cycle, and prototype consistency | fused multiscale/cycle error with prototype uncertainty |
| `PaAno` | local-positive triplet and temporal pretext losses | top-k cosine distance to a normal patch bank |
| `ScatterAD` | scattering, temporal smoothness, cross-view consistency | inverse center distance plus temporal inconsistency |

Each package exports a frozen dataclass config and one `nn.Module`. All detectors
expose `score(...)`; KAN-AD accepts
`score(windows, targets)` because its published criterion is a one-step forecast
error rather than a reconstruction at every point in the input window.

The three multi-stage additions keep state changes explicit:

```python
# AERCA: calibrate normal residuals before root-cause scoring.
aerca.calibrate_root_causes(normal_series)
root_scores = aerca.root_cause_score(test_series)

# CARLA: pretrain with injected anomalies, mine neighbors, then calibrate the normal cluster.
injected = inject_anomalies(train_windows)
pretext_loss = carla.pretext_loss(train_windows, nearby_windows, injected)
nearest, furthest = carla.mine_neighbors(train_windows)
carla.calibrate_normal_clusters(train_windows)

# Left: update prototypes after backward/optimizer.step(), never during forward.
output = left(train_windows)
loss = left.loss_from_output(output).total
loss.backward()
optimizer.step()
left.update_prototypes(train_windows, output)
```

DADA's public repository does not disclose its pretraining procedure. This package
does not invent a replacement objective. It implements the released architecture,
masked-copy inference, adaptive bottleneck, normal/adversarial decoders, gradient
reversal, and `load_reference_checkpoint(...)` for the published checkpoint.

Algorithm references and source material used for semantic verification:

- AERCA, ICLR 2025, `hanxiao0607/AERCA`.
- CARLA, Pattern Recognition 2025, `zamanzadeh/CARLA`.
- CrossAD, NeurIPS 2025, `decisionintelligence/CrossAD`.
- DADA, ICLR 2025, `iambowen/DADA`.
- KAN-AD, ICML 2025, `CSTCloudOps/KAN-AD`.
- Left, `DezhengWang/Left`.
- PaAno, ICLR 2026, `jinnnju/PaAno`.
- ScatterAD, `jk-sounds/ScatterAD`.
